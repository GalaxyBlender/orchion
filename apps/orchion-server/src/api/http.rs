use crate::api::openai::{
    ApiError, ModelList, ModelObject, ModelSubtype, ModelType, OcrApiFormat, OcrJsonResponse,
    SpeechFormat, SpeechRequest, TranscriptionFormat, TranscriptionJson, TranscriptionVerboseJson,
    content_type_for,
};
use crate::api::srt::format_srt;
use crate::api::{docs, ui};
use crate::infrastructure::orchion::AppState;
use crate::settings::{parse_asr_model, parse_tts_model};
use axum::body::Body;
use axum::extract::{DefaultBodyLimit, FromRequest, Multipart, Request, State};
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE, LOCATION};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use orchion::{
    AsrOptions, AsrTimestampGranularity, AudioOutputFormat, ModelId, OcrOptions, OcrResponseFormat,
    OcrTask, OrchionError, TtsAudio, decode_audio_bytes, encode_tts_audio,
};
use orchion_core::{KnownOcrModel, OcrModelKind};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;
use tempfile::{Builder as TempFileBuilder, NamedTempFile};
use tokio::io::AsyncWriteExt;
use tower_http::trace::{DefaultMakeSpan, DefaultOnRequest, DefaultOnResponse, TraceLayer};
use tracing::Level;

pub fn router(state: Arc<AppState>) -> Router {
    router_with_ui_routes(state, ui::routes())
}

pub fn router_with_ui_routes(state: Arc<AppState>, ui_routes: Router<Arc<AppState>>) -> Router {
    let max_upload_size = state.config.server.max_upload_size;
    let mut router = Router::new()
        .route("/", get(root_redirect))
        .route("/healthz", get(healthz))
        .route("/v1/models", get(list_models));

    if state.config.services.tts.enabled {
        router = router.route("/v1/audio/speech", post(create_speech));
    }
    if state.config.services.asr.enabled {
        router = router.route("/v1/audio/transcriptions", post(create_transcription));
    }
    if state.config.services.ocr.active() || state.config.services.ocr_vl.active() {
        router = router.route("/v1/ocr", post(create_ocr));
    }

    router
        .merge(ui_routes)
        .merge(docs::swagger_ui())
        .layer(DefaultBodyLimit::max(max_upload_size))
        .with_state(state)
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(DefaultMakeSpan::new().level(Level::DEBUG))
                .on_request(DefaultOnRequest::new().level(Level::DEBUG))
                .on_response(DefaultOnResponse::new().level(Level::INFO)),
        )
}

async fn root_redirect() -> impl IntoResponse {
    (
        StatusCode::FOUND,
        [(LOCATION, HeaderValue::from_static("/ui"))],
    )
}

async fn healthz() -> &'static str {
    "ok"
}

async fn list_models(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<ModelList>, ApiError> {
    authorize(&state, &headers)?;
    let mut data = Vec::new();
    if state.config.services.asr.enabled {
        data.extend(
            state
                .config
                .services
                .asr
                .available_models
                .iter()
                .copied()
                .map(|model| ModelObject::new(model, ModelType::Asr, None)),
        );
    }
    if state.config.services.tts.enabled {
        data.extend(
            state
                .config
                .services
                .tts
                .available_models
                .iter()
                .copied()
                .map(|model| {
                    ModelObject::new(model, ModelType::Tts, Some(tts_model_subtype(model)))
                }),
        );
    }
    if state.config.services.ocr.active() {
        data.extend(
            state
                .config
                .services
                .ocr
                .available_models
                .iter()
                .map(|id| {
                    ModelObject::from_id(id.as_str(), ModelType::Ocr, Some(ModelSubtype::Standard))
                }),
        );
        data.extend(
            state
                .config
                .services
                .ocr
                .layout_available_models
                .iter()
                .map(|id| {
                    ModelObject::from_id(id.as_str(), ModelType::Ocr, Some(ModelSubtype::Layout))
                }),
        );
    }
    if state.config.services.ocr_vl.active() {
        data.extend(
            state
                .config
                .services
                .ocr_vl
                .available_models
                .iter()
                .map(|id| {
                    ModelObject::from_id(id.as_str(), ModelType::Ocr, Some(ModelSubtype::Vl))
                }),
        );
        data.extend(
            state
                .config
                .services
                .ocr_vl
                .layout_available_models
                .iter()
                .map(|id| {
                    ModelObject::from_id(id.as_str(), ModelType::Ocr, Some(ModelSubtype::Layout))
                }),
        );
    }
    dedupe_model_objects(&mut data);
    Ok(Json(ModelList {
        object: "list",
        data,
    }))
}

fn tts_model_subtype(model: orchion::TtsModel) -> ModelSubtype {
    if model.supports_preset_speakers() {
        ModelSubtype::PresetVoice
    } else if model.supports_voice_design() {
        ModelSubtype::VoiceDesign
    } else {
        ModelSubtype::VoiceClone
    }
}

fn dedupe_model_objects(models: &mut Vec<ModelObject>) {
    let mut seen = HashSet::new();
    models.retain(|model| seen.insert(model.id.clone()));
}

async fn create_speech(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    request: Request,
) -> Result<Response, ApiError> {
    authorize(&state, &headers)?;
    if is_multipart(&headers) {
        let multipart = Multipart::from_request(request, &state)
            .await
            .map_err(|error| {
                ApiError::invalid_request(error.to_string(), None, Some("invalid_multipart"))
            })?;
        return create_speech_multipart(state, multipart).await;
    }

    let Json(request) = Json::<SpeechRequest>::from_request(request, &state)
        .await
        .map_err(|error| {
            ApiError::invalid_request(error.to_string(), None, Some("invalid_json"))
        })?;
    if request.is_voice_clone() {
        return Err(ApiError::invalid_request(
            "voice clone requires multipart/form-data with a reference_audio file upload",
            Some("voice"),
            Some("unsupported_voice_input"),
        ));
    }
    create_speech_from_request(state, request).await
}

async fn create_speech_multipart(
    state: Arc<AppState>,
    mut multipart: Multipart,
) -> Result<Response, ApiError> {
    let mut model = None;
    let mut input = None;
    let mut voice = None;
    let mut response_format = None;
    let mut speed = None;
    let mut language = None;
    let mut reference_audio = None;
    let mut reference_text = None;
    let mut seed = None;
    let mut temperature = None;
    let mut top_k = None;
    let mut top_p = None;
    let mut repetition_penalty = None;
    let mut max_length = None;

    while let Some(field) = multipart.next_field().await.map_err(|error| {
        ApiError::invalid_request(error.to_string(), None, Some("invalid_multipart"))
    })? {
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "reference_audio" => {
                reference_audio = Some(
                    field
                        .bytes()
                        .await
                        .map_err(|error| {
                            ApiError::invalid_request(
                                error.to_string(),
                                Some("reference_audio"),
                                Some("invalid_file"),
                            )
                        })?
                        .to_vec(),
                );
            }
            "model" => model = Some(read_text_field(field, "model").await?),
            "input" => input = Some(read_text_field(field, "input").await?),
            "voice" => voice = Some(read_text_field(field, "voice").await?),
            "response_format" => {
                response_format = Some(SpeechFormat::try_from(
                    read_text_field(field, "response_format").await?.as_str(),
                )?);
            }
            "speed" => speed = Some(parse_multipart_value(field, "speed").await?),
            "language" => language = Some(read_text_field(field, "language").await?),
            "reference_text" => {
                reference_text = Some(read_text_field(field, "reference_text").await?);
            }
            "seed" => seed = Some(parse_multipart_value(field, "seed").await?),
            "temperature" => temperature = Some(parse_multipart_value(field, "temperature").await?),
            "top_k" => top_k = Some(parse_multipart_value(field, "top_k").await?),
            "top_p" => top_p = Some(parse_multipart_value(field, "top_p").await?),
            "repetition_penalty" => {
                repetition_penalty =
                    Some(parse_multipart_value(field, "repetition_penalty").await?);
            }
            "max_length" => max_length = Some(parse_multipart_value(field, "max_length").await?),
            _ => {
                let _ = field.bytes().await;
            }
        }
    }

    let voice = required_multipart_field(voice, "voice")?;
    if voice.trim().to_ascii_lowercase() != "clone" {
        return Err(ApiError::invalid_request(
            "multipart speech is only supported for voice clone requests",
            Some("voice"),
            Some("unsupported_voice_input"),
        ));
    }
    let reference_audio = reference_audio.ok_or_else(|| {
        ApiError::invalid_request(
            "voice clone requires `reference_audio` file upload",
            Some("reference_audio"),
            Some("missing_required_parameter"),
        )
    })?;
    let model = required_multipart_field(model, "model")?;
    let input = required_multipart_field(input, "input")?;
    let request = SpeechRequest {
        model,
        input,
        voice,
        response_format,
        speed: speed.unwrap_or(1.0),
        language,
        reference_audio: Some(String::new()),
        reference_text,
        voice_prompt: None,
        seed,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        max_length,
    };
    request.validate()?;
    parse_tts_model(&request.model).map_err(|_| ApiError::model_not_available(&request.model))?;
    let reference_file = transcode_reference_audio_to_wav_file(reference_audio).await?;
    let reference_path = reference_file
        .path()
        .to_str()
        .ok_or_else(|| ApiError::internal("reference audio path is not valid UTF-8"))?
        .to_string();
    let request = SpeechRequest {
        reference_audio: Some(reference_path),
        ..request
    };
    let response = create_speech_from_request(state, request).await;
    drop(reference_file);
    response
}

async fn transcode_reference_audio_to_wav_file(
    reference_audio: Vec<u8>,
) -> Result<NamedTempFile, ApiError> {
    let decoded = decode_audio_bytes(reference_audio)
        .await
        .map_err(reference_audio_error)?;
    let audio = TtsAudio::new(decoded.samples, decoded.sample_rate);
    let encoded = encode_tts_audio(&audio, AudioOutputFormat::Wav)
        .await
        .map_err(reference_audio_error)?;
    let reference_file = TempFileBuilder::new()
        .suffix(".wav")
        .tempfile()
        .map_err(|error| ApiError::internal(error.to_string()))?;
    tokio::fs::write(reference_file.path(), encoded.bytes)
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    Ok(reference_file)
}

fn reference_audio_error(error: OrchionError) -> ApiError {
    match error {
        OrchionError::InvalidAudio { reason } => {
            ApiError::invalid_request(reason, Some("reference_audio"), Some("invalid_audio"))
        }
        other => ApiError::internal(other.to_string()),
    }
}

async fn create_speech_from_request(
    state: Arc<AppState>,
    request: SpeechRequest,
) -> Result<Response, ApiError> {
    tracing::debug!(
        model = %request.model,
        voice = %request.voice,
        format = ?request.response_format,
        has_language = request.language.is_some(),
        "speech request received"
    );
    request.validate()?;
    let requested = parse_tts_model(&request.model)
        .map_err(|_| ApiError::model_not_available(&request.model))?;
    let tts = state
        .tts(requested)
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?
        .ok_or_else(|| ApiError::model_not_available(&request.model))?;
    let format = request
        .response_format
        .map(Ok)
        .unwrap_or_else(|| SpeechFormat::try_from(state.config.services.tts.format.as_str()))?;
    let voice = request.to_tts_voice()?;
    let synthesis_started = Instant::now();
    tracing::debug!("speech synthesis started");
    let options = request.to_tts_options();
    let audio = tts.synthesize_with(request.input, voice, options).await?;
    let synthesis_elapsed = synthesis_started.elapsed();
    tracing::debug!(
        samples = audio.samples.len(),
        sample_rate = audio.sample_rate,
        elapsed_ms = synthesis_elapsed.as_millis(),
        "speech synthesis completed"
    );
    let encode_started = Instant::now();
    tracing::debug!(format = %format, "speech audio encode started");
    let encoded = encode_tts_audio(&audio, AudioOutputFormat::from(format)).await?;
    let encode_elapsed = encode_started.elapsed();
    tracing::info!(format = %format, "speech request completed");
    tracing::debug!(
        bytes = encoded.bytes.len(),
        format = %format,
        elapsed_ms = encode_elapsed.as_millis(),
        "speech response encoded"
    );
    Response::builder()
        .status(StatusCode::OK)
        .header(
            CONTENT_TYPE,
            HeaderValue::from_static(content_type_for(format)),
        )
        .body(Body::from(encoded.bytes))
        .map_err(|error| ApiError::internal(error.to_string()))
}

async fn create_transcription(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Result<Response, ApiError> {
    authorize(&state, &headers)?;
    let mut audio_file = None;
    let mut model = None;
    let mut language = None;
    let mut response_format = TranscriptionFormat::default();
    let mut timestamp_granularities = Vec::new();

    while let Some(field) = multipart.next_field().await.map_err(|error| {
        ApiError::invalid_request(error.to_string(), None, Some("invalid_multipart"))
    })? {
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "file" => {
                audio_file = Some(write_multipart_file_to_temp_file(field, "file").await?);
            }
            "model" => model = Some(read_text_field(field, "model").await?),
            "language" => language = Some(read_text_field(field, "language").await?),
            "response_format" => {
                let value = read_text_field(field, "response_format").await?;
                response_format = TranscriptionFormat::try_from(value.as_str())?;
            }
            "timestamp_granularities[]" | "timestamp_granularities" => {
                timestamp_granularities
                    .push(read_text_field(field, "timestamp_granularities").await?);
            }
            "prompt" | "temperature" => {
                let _ = field.text().await;
            }
            _ => {
                let _ = field.text().await;
            }
        }
    }

    let segment_timestamps = parse_timestamp_granularities(&timestamp_granularities)?;
    let use_segments = segment_timestamps || matches!(response_format, TranscriptionFormat::Srt);

    let model = model.ok_or_else(|| {
        ApiError::invalid_request(
            "`model` is required",
            Some("model"),
            Some("missing_required_parameter"),
        )
    })?;
    let requested = parse_asr_model(&model).map_err(|_| ApiError::model_not_available(&model))?;
    let asr = state
        .asr(requested)
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?
        .ok_or_else(|| ApiError::model_not_available(&model))?;
    let (audio_file, audio_bytes) = audio_file.ok_or_else(|| {
        ApiError::invalid_request(
            "`file` is required",
            Some("file"),
            Some("missing_required_parameter"),
        )
    })?;
    if audio_bytes == 0 {
        return Err(ApiError::invalid_request(
            "uploaded audio file is empty",
            Some("file"),
            Some("invalid_file"),
        ));
    }
    let audio_path = audio_file.path().to_path_buf();
    tracing::debug!(
        model = %model,
        language = ?language,
        response_format = ?response_format,
        audio_bytes,
        "transcription request received"
    );
    let options = AsrOptions {
        language,
        ..Default::default()
    };
    let transcript = if use_segments {
        asr.transcribe_audio_file_with_segments(audio_path, options)
            .await?
    } else {
        asr.transcribe_audio_file_with(audio_path, options).await?
    };
    tracing::info!(format = ?response_format, "transcription request completed");

    Ok(match response_format {
        TranscriptionFormat::Json => Json(TranscriptionJson {
            text: transcript.text,
        })
        .into_response(),
        TranscriptionFormat::VerboseJson => Json(TranscriptionVerboseJson {
            text: transcript.text,
            language: transcript.language,
            raw_output: transcript.raw_output,
            segments: segment_timestamps.then_some(transcript.segments),
        })
        .into_response(),
        TranscriptionFormat::Srt => (
            [(
                CONTENT_TYPE,
                HeaderValue::from_static("text/plain; charset=utf-8"),
            )],
            format_srt(&transcript),
        )
            .into_response(),
        TranscriptionFormat::Text => (
            [(
                CONTENT_TYPE,
                HeaderValue::from_static("text/plain; charset=utf-8"),
            )],
            transcript.text,
        )
            .into_response(),
    })
}

async fn create_ocr(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Result<Response, ApiError> {
    authorize(&state, &headers)?;
    let mut image_file = None;
    let mut model = None;
    let mut response_format = None;
    let mut task = OcrTask::Ocr;
    let mut layout_model = None;
    let mut max_tokens = None;

    while let Some(field) = multipart.next_field().await.map_err(|error| {
        ApiError::invalid_request(error.to_string(), None, Some("invalid_multipart"))
    })? {
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "file" => {
                image_file = Some(write_multipart_file_to_temp_file(field, "file").await?);
            }
            "model" => model = Some(read_text_field(field, "model").await?),
            "response_format" => {
                let value = read_text_field(field, "response_format").await?;
                response_format = Some(OcrApiFormat::try_from(value.as_str())?);
            }
            "task" => {
                let value = read_text_field(field, "task").await?;
                task = parse_ocr_task(&value)?;
            }
            "layout_model" => {
                let value = read_text_field(field, "layout_model").await?;
                layout_model = Some(parse_ocr_model_id(&value, "layout_model")?);
            }
            "max_tokens" => {
                let value: usize = parse_multipart_value(field, "max_tokens").await?;
                if value == 0 {
                    return Err(ApiError::invalid_request(
                        "`max_tokens` must be greater than 0",
                        Some("max_tokens"),
                        Some("invalid_multipart_field"),
                    ));
                }
                max_tokens = Some(value);
            }
            _ => {
                let _ = field.text().await;
            }
        }
    }

    let (image_file, image_bytes) = image_file.ok_or_else(|| {
        ApiError::invalid_request(
            "`file` is required",
            Some("file"),
            Some("missing_required_parameter"),
        )
    })?;
    if image_bytes == 0 {
        return Err(ApiError::invalid_request(
            "uploaded OCR file is empty",
            Some("file"),
            Some("invalid_file"),
        ));
    }

    let choice = resolve_ocr_service_choice(&state, model.as_deref(), response_format)?;
    let response_format = resolve_ocr_response_format(&state, choice, response_format);
    validate_ocr_parameters(
        choice,
        response_format,
        task,
        layout_model.as_ref(),
        max_tokens,
        &state.config.services.ocr,
        &state.config.services.ocr_vl,
    )?;
    let ocr = match choice {
        OcrServiceChoice::Ocr { model } => state.ocr(model).await,
        OcrServiceChoice::OcrVl { model } => state.ocr_vl(model).await,
    }
    .map_err(|error| {
        tracing::error!(error = %format_args!("{error:#}"), "failed to load OCR runtime");
        ApiError::internal(error.to_string())
    })?
    .ok_or_else(|| ApiError::model_not_available(choice.model().id()))?;

    let options = OcrOptions {
        response_format: OcrResponseFormat::from(response_format),
        task,
        layout_model: resolve_ocr_layout_model(&state, choice, layout_model),
        max_tokens,
    };
    let result = ocr.recognize_file_with(image_file.path(), options).await?;

    Ok(match response_format {
        OcrApiFormat::Json => Json(OcrJsonResponse {
            model: result.model.to_string(),
            format: response_format,
            text: result.text,
            markdown: result.markdown,
            html: result.html,
            regions: result.regions,
            layout_blocks: result.layout_blocks,
            usage: result.usage,
        })
        .into_response(),
        OcrApiFormat::Text => (
            [(
                CONTENT_TYPE,
                HeaderValue::from_static("text/plain; charset=utf-8"),
            )],
            result.text,
        )
            .into_response(),
        OcrApiFormat::Markdown => (
            [(
                CONTENT_TYPE,
                HeaderValue::from_static("text/markdown; charset=utf-8"),
            )],
            result.markdown.unwrap_or(result.text),
        )
            .into_response(),
        OcrApiFormat::Html => (
            [(
                CONTENT_TYPE,
                HeaderValue::from_static("text/html; charset=utf-8"),
            )],
            result.html.unwrap_or(result.text),
        )
            .into_response(),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OcrServiceChoice {
    Ocr { model: KnownOcrModel },
    OcrVl { model: KnownOcrModel },
}

impl OcrServiceChoice {
    fn ocr(model: KnownOcrModel) -> Result<Self, ApiError> {
        if model.kind() == OcrModelKind::OcrVl {
            return Err(invalid_ocr_model_kind(model, "traditional OCR"));
        }
        Ok(Self::Ocr { model })
    }

    fn ocr_vl(model: KnownOcrModel) -> Result<Self, ApiError> {
        if model.kind() != OcrModelKind::OcrVl {
            return Err(invalid_ocr_model_kind(model, "OCR-VL"));
        }
        Ok(Self::OcrVl { model })
    }

    const fn model(self) -> KnownOcrModel {
        match self {
            Self::Ocr { model } | Self::OcrVl { model } => model,
        }
    }

    const fn is_ocr_vl(self) -> bool {
        matches!(self, Self::OcrVl { .. })
    }
}

fn resolve_ocr_service_choice(
    state: &AppState,
    model: Option<&str>,
    response_format: Option<OcrApiFormat>,
) -> Result<OcrServiceChoice, ApiError> {
    if let Some(model) = model {
        return resolve_explicit_ocr_model(state, model);
    }

    let ocr_active = state.config.services.ocr.active();
    let ocr_vl_active = state.config.services.ocr_vl.active();
    match (ocr_active, ocr_vl_active) {
        (true, false) => OcrServiceChoice::ocr(default_ocr_choice(state, false)?),
        (false, true) => OcrServiceChoice::ocr_vl(default_ocr_choice(state, true)?),
        (true, true) => resolve_default_ocr_model(state, response_format),
        (false, false) => Err(ApiError::invalid_request(
            "`model` is required",
            Some("model"),
            Some("missing_required_parameter"),
        )),
    }
}

fn resolve_explicit_ocr_model(state: &AppState, model: &str) -> Result<OcrServiceChoice, ApiError> {
    let model_id = ModelId::parse(model).map_err(|_| ApiError::model_not_available(model))?;
    let ocr_match = state.config.services.ocr.active()
        && state
            .config
            .services
            .ocr
            .available_models
            .contains(&model_id);
    let ocr_vl_match = state.config.services.ocr_vl.active()
        && state
            .config
            .services
            .ocr_vl
            .available_models
            .contains(&model_id);

    match (ocr_match, ocr_vl_match) {
        (true, true) => Err(ApiError::invalid_request(
            format!("model `{model}` is configured for both OCR services"),
            Some("model"),
            Some("ambiguous_model"),
        )),
        (true, false) => OcrServiceChoice::ocr(known_ocr_model(&model_id, model)?),
        (false, true) => OcrServiceChoice::ocr_vl(known_ocr_model(&model_id, model)?),
        (false, false) => Err(ApiError::model_not_available(model)),
    }
}

fn resolve_default_ocr_model(
    state: &AppState,
    response_format: Option<OcrApiFormat>,
) -> Result<OcrServiceChoice, ApiError> {
    let prefer_ocr_vl = matches!(
        response_format,
        Some(OcrApiFormat::Markdown | OcrApiFormat::Html)
    );
    if prefer_ocr_vl {
        if effective_default_ocr_model(state, true).is_some() {
            OcrServiceChoice::ocr_vl(default_ocr_choice(state, true)?)
        } else {
            OcrServiceChoice::ocr(default_ocr_choice(state, false)?)
        }
    } else {
        if effective_default_ocr_model(state, false).is_some() {
            OcrServiceChoice::ocr(default_ocr_choice(state, false)?)
        } else {
            OcrServiceChoice::ocr_vl(default_ocr_choice(state, true)?)
        }
    }
}

fn resolve_ocr_response_format(
    state: &AppState,
    choice: OcrServiceChoice,
    response_format: Option<OcrApiFormat>,
) -> OcrApiFormat {
    response_format.unwrap_or_else(|| match choice {
        OcrServiceChoice::Ocr { .. } => OcrApiFormat::from(state.config.services.ocr.format),
        OcrServiceChoice::OcrVl { .. } => OcrApiFormat::from(state.config.services.ocr_vl.format),
    })
}

fn resolve_ocr_layout_model(
    _state: &AppState,
    _choice: OcrServiceChoice,
    layout_model: Option<ModelId>,
) -> Option<ModelId> {
    layout_model
}

impl From<OcrResponseFormat> for OcrApiFormat {
    fn from(format: OcrResponseFormat) -> Self {
        match format {
            OcrResponseFormat::Json => Self::Json,
            OcrResponseFormat::Text => Self::Text,
            OcrResponseFormat::Markdown => Self::Markdown,
            OcrResponseFormat::Html => Self::Html,
        }
    }
}

fn default_ocr_choice(state: &AppState, ocr_vl: bool) -> Result<KnownOcrModel, ApiError> {
    let Some(default_model) = effective_default_ocr_model(state, ocr_vl) else {
        return Err(ApiError::invalid_request(
            "`model` is required",
            Some("model"),
            Some("missing_required_parameter"),
        ));
    };
    known_ocr_model(default_model, default_model.as_str())
}

fn effective_default_ocr_model(state: &AppState, ocr_vl: bool) -> Option<&ModelId> {
    if ocr_vl {
        let service = &state.config.services.ocr_vl;
        if !service.active() {
            return None;
        }
        service
            .default_model
            .as_ref()
            .or_else(|| service.available_models.first())
    } else {
        let service = &state.config.services.ocr;
        if !service.active() {
            return None;
        }
        service
            .default_model
            .as_ref()
            .or_else(|| service.available_models.first())
    }
}

fn known_ocr_model(model_id: &ModelId, raw_model: &str) -> Result<KnownOcrModel, ApiError> {
    KnownOcrModel::from_model_id(model_id).map_err(|_| ApiError::model_not_available(raw_model))
}

fn invalid_ocr_model_kind(model: KnownOcrModel, service: &str) -> ApiError {
    ApiError::invalid_request(
        format!(
            "model `{}` is not compatible with the {service} service",
            model.id()
        ),
        Some("model"),
        Some("invalid_ocr_model_kind"),
    )
}

fn validate_ocr_parameters(
    choice: OcrServiceChoice,
    response_format: OcrApiFormat,
    task: OcrTask,
    layout_model: Option<&ModelId>,
    max_tokens: Option<usize>,
    ocr_service: &crate::settings::OcrServiceSection,
    ocr_vl_service: &crate::settings::OcrVlServiceSection,
) -> Result<(), ApiError> {
    let model = choice.model();
    if matches!(response_format, OcrApiFormat::Markdown | OcrApiFormat::Html)
        && !model.supports_markdown()
        && layout_model.is_none()
    {
        return Err(ApiError::invalid_request(
            "selected OCR model does not support structured response format",
            Some("response_format"),
            Some("unsupported_response_format"),
        ));
    }
    if choice.is_ocr_vl() {
        if let Some(layout_model) = layout_model {
            validate_ocr_layout_model(layout_model)?;
            validate_configured_layout_model(
                &ocr_vl_service.layout_available_models,
                layout_model,
                "OCR-VL",
            )?;
        }
        return Ok(());
    }
    if let Some(layout_model) = layout_model {
        validate_ocr_layout_model(layout_model)?;
        validate_configured_layout_model(
            &ocr_service.layout_available_models,
            layout_model,
            "OCR",
        )?;
    }
    if task != OcrTask::Ocr || max_tokens.is_some() {
        return Err(ApiError::invalid_request(
            "selected OCR model does not support OCR-VL parameters",
            None,
            Some("unsupported_ocr_parameter"),
        ));
    }
    Ok(())
}

fn validate_ocr_layout_model(layout_model: &ModelId) -> Result<(), ApiError> {
    let model = known_ocr_model(layout_model, layout_model.as_str())?;
    if model == KnownOcrModel::PpDocLayoutV3 {
        return Ok(());
    }
    Err(ApiError::invalid_request(
        "`layout_model` must be PaddlePaddle/PP-DocLayoutV3",
        Some("layout_model"),
        Some("invalid_ocr_model_kind"),
    ))
}

fn validate_configured_layout_model(
    available_models: &[ModelId],
    layout_model: &ModelId,
    service_name: &str,
) -> Result<(), ApiError> {
    if available_models.contains(layout_model) {
        return Ok(());
    }
    Err(ApiError::invalid_request(
        format!("`layout_model` is not configured for the {service_name} service"),
        Some("layout_model"),
        Some("model_not_available"),
    ))
}

fn parse_ocr_task(value: &str) -> Result<OcrTask, ApiError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "ocr" => Ok(OcrTask::Ocr),
        "table" => Ok(OcrTask::Table),
        "formula" => Ok(OcrTask::Formula),
        "chart" => Ok(OcrTask::Chart),
        "spotting" => Ok(OcrTask::Spotting),
        "seal" => Ok(OcrTask::Seal),
        _ => Err(ApiError::invalid_request(
            "unsupported OCR task; supported tasks are ocr, table, formula, chart, spotting, and seal",
            Some("task"),
            Some("unsupported_ocr_parameter"),
        )),
    }
}

fn parse_ocr_model_id(value: &str, param: &'static str) -> Result<ModelId, ApiError> {
    ModelId::parse(value).map_err(|_| {
        ApiError::invalid_request(
            format!("invalid `{param}`; expected vendor/name"),
            Some(param),
            Some("invalid_multipart_field"),
        )
    })
}

fn parse_timestamp_granularities(values: &[String]) -> Result<bool, ApiError> {
    let mut wants_segments = false;
    for value in values {
        match value.parse::<AsrTimestampGranularity>().map_err(|error| {
            ApiError::invalid_request(
                error,
                Some("timestamp_granularities"),
                Some("unsupported_timestamp_granularity"),
            )
        })? {
            AsrTimestampGranularity::Segment => wants_segments = true,
            AsrTimestampGranularity::Word => {
                return Err(ApiError::invalid_request(
                    "word timestamp granularity is not supported",
                    Some("timestamp_granularities"),
                    Some("unsupported_timestamp_granularity"),
                ));
            }
        }
    }
    Ok(wants_segments)
}

fn authorize(state: &AppState, headers: &HeaderMap) -> Result<(), ApiError> {
    let Some(api_key) = state.config.auth.api_key.as_deref() else {
        return Ok(());
    };
    let Some(header) = headers
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
    else {
        return Err(ApiError::invalid_api_key());
    };
    let Some(token) = header.strip_prefix("Bearer ") else {
        return Err(ApiError::invalid_api_key());
    };
    if token == api_key {
        Ok(())
    } else {
        Err(ApiError::invalid_api_key())
    }
}

fn is_multipart(headers: &HeaderMap) -> bool {
    headers
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| {
            value
                .to_ascii_lowercase()
                .starts_with("multipart/form-data")
        })
}

fn required_multipart_field(
    value: Option<String>,
    param: &'static str,
) -> Result<String, ApiError> {
    value.ok_or_else(|| {
        ApiError::invalid_request(
            format!("`{param}` is required"),
            Some(param),
            Some("missing_required_parameter"),
        )
    })
}

async fn parse_multipart_value<T>(
    field: axum::extract::multipart::Field<'_>,
    param: &'static str,
) -> Result<T, ApiError>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let value = read_text_field(field, param).await?;
    value.trim().parse().map_err(|error| {
        ApiError::invalid_request(
            format!("invalid `{param}`: {error}"),
            Some(param),
            Some("invalid_multipart_field"),
        )
    })
}

async fn read_text_field(
    field: axum::extract::multipart::Field<'_>,
    param: &str,
) -> Result<String, ApiError> {
    field.text().await.map_err(|error| {
        ApiError::invalid_request(
            error.to_string(),
            Some(param),
            Some("invalid_multipart_field"),
        )
    })
}

async fn write_multipart_file_to_temp_file(
    mut field: axum::extract::multipart::Field<'_>,
    param: &'static str,
) -> Result<(NamedTempFile, u64), ApiError> {
    let suffix = multipart_file_suffix(field.content_type());
    let audio_file = TempFileBuilder::new()
        .prefix("orchion-upload-")
        .suffix(suffix)
        .tempfile()
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let mut writer = tokio::fs::File::create(audio_file.path())
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let mut bytes_written = 0_u64;

    while let Some(chunk) = field.chunk().await.map_err(|error| {
        ApiError::invalid_request(error.to_string(), Some(param), Some("invalid_file"))
    })? {
        writer
            .write_all(&chunk)
            .await
            .map_err(|error| ApiError::internal(error.to_string()))?;
        bytes_written += u64::try_from(chunk.len()).map_err(|error| {
            ApiError::internal(format!("uploaded file chunk size overflowed u64: {error}"))
        })?;
    }
    writer
        .flush()
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;

    Ok((audio_file, bytes_written))
}

fn multipart_file_suffix(content_type: Option<&str>) -> &'static str {
    match content_type
        .and_then(|value| value.split(';').next())
        .map(str::trim)
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("image/png") => ".png",
        Some("image/jpeg") | Some("image/jpg") => ".jpg",
        Some("image/webp") => ".webp",
        Some("image/bmp") => ".bmp",
        Some("image/tiff") => ".tiff",
        Some("application/pdf") => ".pdf",
        Some("video/mp4") => ".mp4",
        Some("video/quicktime") => ".mov",
        Some("video/webm") => ".webm",
        Some("video/x-matroska") => ".mkv",
        _ => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::application::model_cache::{
        AsrModelCache, GlobalModelCacheLimiter, OcrModelCache, OcrVlModelCache, TtsModelCache,
    };
    use crate::settings::ServerConfig;
    use axum::body::Body;
    use http_body_util::BodyExt;
    use orchion::{AsrModel, ModelId, TtsModel};
    use orchion_core::KnownOcrModel;
    use serde_json::Value;
    use std::time::Duration;
    use tower::ServiceExt;

    #[test]
    fn parse_timestamp_granularities_accepts_segment() {
        let values = vec!["segment".to_string()];

        assert!(parse_timestamp_granularities(&values).unwrap());
    }

    #[test]
    fn parse_timestamp_granularities_rejects_word() {
        let values = vec!["segment".to_string(), "word".to_string()];

        let error = parse_timestamp_granularities(&values).unwrap_err();

        assert_eq!(
            error.error.code.as_deref(),
            Some("unsupported_timestamp_granularity")
        );
    }

    #[test]
    fn parse_timestamp_granularities_rejects_unknown_value() {
        let values = vec!["sentence".to_string()];

        let error = parse_timestamp_granularities(&values).unwrap_err();

        assert_eq!(
            error.error.param.as_deref(),
            Some("timestamp_granularities")
        );
    }

    #[test]
    fn multipart_file_suffix_uses_supported_mime_type() {
        assert_eq!(multipart_file_suffix(Some("image/png")), ".png");
        assert_eq!(multipart_file_suffix(Some("image/jpeg")), ".jpg");
        assert_eq!(multipart_file_suffix(Some("application/pdf")), ".pdf");
        assert_eq!(multipart_file_suffix(Some("video/mp4")), ".mp4");
        assert_eq!(multipart_file_suffix(Some("text/plain")), "");
        assert_eq!(multipart_file_suffix(None), "");
    }

    #[tokio::test]
    async fn ocr_route_is_absent_when_ocr_services_are_inactive() {
        let response = router_with_ui_routes(test_state(false, false), Router::new())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/ocr")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn multipart_ocr_requires_file() {
        let boundary = "orchion-ocr-missing-file";
        let body = multipart_body(boundary, &[("model", "PaddlePaddle/PP-OCRv6_tiny")], None);

        let response = post_ocr(test_state(true, false), boundary, body).await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = json_body(response).await;
        assert_eq!(body["error"]["code"], "missing_required_parameter");
        assert_eq!(body["error"]["param"], "file");
    }

    #[tokio::test]
    async fn multipart_ocr_rejects_empty_file() {
        let boundary = "orchion-ocr-empty-file";
        let body = multipart_body(
            boundary,
            &[("model", "PaddlePaddle/PP-OCRv6_tiny")],
            Some(("file", "empty.png", b"")),
        );

        let response = post_ocr(test_state(true, false), boundary, body).await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = json_body(response).await;
        assert_eq!(body["error"]["code"], "invalid_file");
        assert_eq!(body["error"]["param"], "file");
    }

    #[tokio::test]
    async fn multipart_ocr_rejects_invalid_response_format() {
        let boundary = "orchion-ocr-invalid-format";
        let body = multipart_body(
            boundary,
            &[("response_format", "verbose_json")],
            Some(("file", "document.png", b"image")),
        );

        let response = post_ocr(test_state(true, false), boundary, body).await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = json_body(response).await;
        assert_eq!(body["error"]["code"], "unsupported_response_format");
        assert_eq!(body["error"]["param"], "response_format");
    }

    #[tokio::test]
    async fn multipart_ocr_rejects_unknown_model() {
        let boundary = "orchion-ocr-unknown-model";
        let body = multipart_body(
            boundary,
            &[("model", "Acme/Experimental-OCR")],
            Some(("file", "document.png", b"image")),
        );

        let response = post_ocr(test_state(true, false), boundary, body).await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = json_body(response).await;
        assert_eq!(body["error"]["code"], "model_not_available");
        assert_eq!(body["error"]["param"], "model");
    }

    #[test]
    fn resolve_ocr_service_choice_rejects_ambiguous_explicit_model() {
        let mut state = test_state(true, true);
        let model = ModelId::parse("PaddlePaddle/PaddleOCR-VL-1.6").unwrap();
        Arc::get_mut(&mut state)
            .unwrap()
            .config
            .services
            .ocr
            .available_models = vec![model];

        let error = resolve_ocr_service_choice(
            &state,
            Some("PaddlePaddle/PaddleOCR-VL-1.6"),
            Some(OcrApiFormat::Json),
        )
        .unwrap_err();

        assert_eq!(error.error.code.as_deref(), Some("ambiguous_model"));
    }

    #[test]
    fn resolve_ocr_service_choice_selects_ocr_vl_for_default_markdown() {
        let state = test_state(true, true);

        let choice =
            resolve_ocr_service_choice(&state, None, Some(OcrApiFormat::Markdown)).unwrap();

        assert_eq!(choice.model(), KnownOcrModel::PaddleOcrVl16);
        assert!(choice.is_ocr_vl());
    }

    #[test]
    fn resolve_ocr_service_choice_selects_ocr_vl_for_default_html() {
        let state = test_state(true, true);

        let choice = resolve_ocr_service_choice(&state, None, Some(OcrApiFormat::Html)).unwrap();

        assert_eq!(choice.model(), KnownOcrModel::PaddleOcrVl16);
        assert!(choice.is_ocr_vl());
    }

    #[test]
    fn resolve_ocr_service_choice_defaults_ocr_vl_format_when_only_ocr_vl_is_active() {
        let state = test_state(false, true);

        let choice = resolve_ocr_service_choice(&state, None, None).unwrap();
        let response_format = resolve_ocr_response_format(&state, choice, None);

        assert_eq!(choice.model(), KnownOcrModel::PaddleOcrVl16);
        assert!(choice.is_ocr_vl());
        assert_eq!(response_format, OcrApiFormat::Markdown);
    }

    #[test]
    fn resolve_ocr_service_choice_defaults_to_ocr_config_format_when_ocr_is_active() {
        let mut state = test_state(true, true);
        Arc::get_mut(&mut state).unwrap().config.services.ocr.format = OcrResponseFormat::Text;

        let choice = resolve_ocr_service_choice(&state, None, None).unwrap();
        let response_format = resolve_ocr_response_format(&state, choice, None);

        assert_eq!(choice.model(), KnownOcrModel::PpOcrV6Tiny);
        assert!(!choice.is_ocr_vl());
        assert_eq!(response_format, OcrApiFormat::Text);
    }

    #[test]
    fn resolve_ocr_service_choice_keeps_explicit_markdown_preference_for_ocr_vl() {
        let state = test_state(true, true);

        let choice =
            resolve_ocr_service_choice(&state, None, Some(OcrApiFormat::Markdown)).unwrap();
        let response_format =
            resolve_ocr_response_format(&state, choice, Some(OcrApiFormat::Markdown));

        assert_eq!(choice.model(), KnownOcrModel::PaddleOcrVl16);
        assert!(choice.is_ocr_vl());
        assert_eq!(response_format, OcrApiFormat::Markdown);
    }

    #[test]
    fn resolve_ocr_service_choice_selects_ocr_vl_fallback_for_default_markdown() {
        let mut state = test_state(true, true);
        let ocr_vl_model = ModelId::parse("PaddlePaddle/PaddleOCR-VL-1.6").unwrap();
        let state_mut = Arc::get_mut(&mut state).unwrap();
        state_mut.config.services.ocr_vl.default_model = None;
        state_mut.config.services.ocr_vl.available_models = vec![ocr_vl_model];

        let choice =
            resolve_ocr_service_choice(&state, None, Some(OcrApiFormat::Markdown)).unwrap();

        assert_eq!(choice.model(), KnownOcrModel::PaddleOcrVl16);
        assert!(choice.is_ocr_vl());
    }

    #[test]
    fn resolve_ocr_service_choice_selects_ocr_fallback_for_default_format() {
        let mut state = test_state(true, true);
        Arc::get_mut(&mut state)
            .unwrap()
            .config
            .services
            .ocr
            .default_model = None;

        let choice = resolve_ocr_service_choice(&state, None, None).unwrap();

        assert_eq!(choice.model(), KnownOcrModel::PpOcrV6Tiny);
        assert!(!choice.is_ocr_vl());
    }

    #[test]
    fn resolve_ocr_layout_model_does_not_use_ocr_vl_layout_default_without_request_value() {
        let mut state = test_state(false, true);
        let layout_model = ModelId::parse("PaddlePaddle/PP-DocLayoutV3").unwrap();
        Arc::get_mut(&mut state)
            .unwrap()
            .config
            .services
            .ocr_vl
            .layout_default_model = Some(layout_model.clone());
        let choice = resolve_ocr_service_choice(&state, None, None).unwrap();

        let resolved = resolve_ocr_layout_model(&state, choice, None);

        assert_eq!(resolved, None);
    }

    #[test]
    fn resolve_ocr_layout_model_does_not_use_ocr_layout_default_without_request_value() {
        let mut state = test_state(true, false);
        let layout_model = ModelId::parse("PaddlePaddle/PP-DocLayoutV3").unwrap();
        Arc::get_mut(&mut state)
            .unwrap()
            .config
            .services
            .ocr
            .layout_default_model = Some(layout_model.clone());
        let choice = resolve_ocr_service_choice(&state, None, None).unwrap();

        let resolved = resolve_ocr_layout_model(&state, choice, None);

        assert_eq!(resolved, None);
    }

    #[test]
    fn validate_ocr_parameters_allows_markdown_for_traditional_layout() {
        let mut state = test_state(true, false);
        let layout_model = ModelId::parse("PaddlePaddle/PP-DocLayoutV3").unwrap();
        Arc::get_mut(&mut state)
            .unwrap()
            .config
            .services
            .ocr
            .layout_available_models = vec![layout_model.clone()];
        let choice = OcrServiceChoice::Ocr {
            model: KnownOcrModel::PpOcrV6Tiny,
        };

        validate_ocr_parameters(
            choice,
            OcrApiFormat::Markdown,
            OcrTask::Ocr,
            Some(&layout_model),
            None,
            &state.config.services.ocr,
            &state.config.services.ocr_vl,
        )
        .unwrap();
    }

    #[test]
    fn validate_ocr_parameters_rejects_unconfigured_traditional_layout() {
        let state = test_state(true, false);
        let layout_model = ModelId::parse("PaddlePaddle/PP-DocLayoutV3").unwrap();
        let choice = OcrServiceChoice::Ocr {
            model: KnownOcrModel::PpOcrV6Tiny,
        };

        let error = validate_ocr_parameters(
            choice,
            OcrApiFormat::Json,
            OcrTask::Ocr,
            Some(&layout_model),
            None,
            &state.config.services.ocr,
            &state.config.services.ocr_vl,
        )
        .unwrap_err();

        assert_eq!(error.error.code.as_deref(), Some("model_not_available"));
        assert_eq!(error.error.param.as_deref(), Some("layout_model"));
    }

    #[test]
    fn resolve_ocr_service_choice_rejects_unknown_explicit_model() {
        let state = test_state(true, false);

        let error = resolve_ocr_service_choice(
            &state,
            Some("Acme/Experimental-OCR"),
            Some(OcrApiFormat::Json),
        )
        .unwrap_err();

        assert_eq!(error.error.code.as_deref(), Some("model_not_available"));
    }

    #[test]
    fn resolve_ocr_service_choice_rejects_traditional_model_not_configured_for_ocr_vl_service() {
        let state = test_state(false, true);

        let error = resolve_ocr_service_choice(
            &state,
            Some("PaddlePaddle/PP-OCRv6_tiny"),
            Some(OcrApiFormat::Json),
        )
        .unwrap_err();

        assert_eq!(error.error.code.as_deref(), Some("model_not_available"));
        assert_eq!(error.error.param.as_deref(), Some("model"));
    }

    #[test]
    fn resolve_ocr_service_choice_rejects_ocr_vl_model_in_traditional_service() {
        let mut state = test_state(true, false);
        let state_mut = Arc::get_mut(&mut state).unwrap();
        let ocr_vl_model = ModelId::parse("PaddlePaddle/PaddleOCR-VL-1.6").unwrap();
        state_mut.config.services.ocr.default_model = Some(ocr_vl_model.clone());
        state_mut.config.services.ocr.available_models = vec![ocr_vl_model];

        let error = resolve_ocr_service_choice(
            &state,
            Some("PaddlePaddle/PaddleOCR-VL-1.6"),
            Some(OcrApiFormat::Json),
        )
        .unwrap_err();

        assert_eq!(error.error.code.as_deref(), Some("invalid_ocr_model_kind"));
        assert_eq!(error.error.param.as_deref(), Some("model"));
    }

    async fn post_ocr(state: Arc<AppState>, boundary: &str, body: Vec<u8>) -> Response {
        router_with_ui_routes(state, Router::new())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/ocr")
                    .header(
                        CONTENT_TYPE,
                        format!("multipart/form-data; boundary={boundary}"),
                    )
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap()
    }

    async fn json_body(response: Response) -> Value {
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice(&bytes).unwrap()
    }

    fn multipart_body(
        boundary: &str,
        fields: &[(&str, &str)],
        file: Option<(&str, &str, &[u8])>,
    ) -> Vec<u8> {
        let mut body = Vec::new();
        for (name, value) in fields {
            body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
            body.extend_from_slice(
                format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
            );
            body.extend_from_slice(value.as_bytes());
            body.extend_from_slice(b"\r\n");
        }
        if let Some((field_name, file_name, file_bytes)) = file {
            body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
            body.extend_from_slice(
                format!(
                    "Content-Disposition: form-data; name=\"{field_name}\"; filename=\"{file_name}\"\r\nContent-Type: image/png\r\n\r\n"
                )
                .as_bytes(),
            );
            body.extend_from_slice(file_bytes);
            body.extend_from_slice(b"\r\n");
        }
        body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        body
    }

    fn test_state(ocr_active: bool, ocr_vl_active: bool) -> Arc<AppState> {
        let mut config = ServerConfig::default_for_exe(std::path::Path::new("/tmp/orchion-server"));
        config.services.asr.enabled = false;
        config.services.tts.enabled = false;
        config.services.ocr.enabled = ocr_active;
        config.services.ocr.default_model =
            Some(ModelId::parse("PaddlePaddle/PP-OCRv6_tiny").unwrap());
        config.services.ocr.available_models = vec![
            ModelId::parse("PaddlePaddle/PP-OCRv6_tiny").unwrap(),
            ModelId::parse("PaddlePaddle/PP-OCRv6_small").unwrap(),
        ];
        config.services.ocr_vl.enabled = ocr_vl_active;
        config.services.ocr_vl.default_model =
            Some(ModelId::parse("PaddlePaddle/PaddleOCR-VL-1.6").unwrap());
        config.services.ocr_vl.available_models =
            vec![ModelId::parse("PaddlePaddle/PaddleOCR-VL-1.6").unwrap()];
        config.services.asr.available_models = vec![AsrModel::Qwen3Asr06B];
        config.services.tts.available_models = vec![TtsModel::Qwen3Tts06BCustomVoice];
        config.services.asr.idle_timeout = Duration::from_secs(600);
        config.services.tts.idle_timeout = Duration::from_secs(600);

        let asr_models = AsrModelCache::new(
            "asr",
            config.services.asr.available_models.clone(),
            config.services.asr.idle_timeout,
            config.services.asr.max_loaded,
            config.models.dir.clone(),
        );
        let tts_models = TtsModelCache::new(
            "tts",
            config.services.tts.available_models.clone(),
            config.services.tts.idle_timeout,
            config.services.tts.max_loaded,
            config.models.dir.clone(),
        );
        let ocr_models = OcrModelCache::new(
            "ocr",
            vec![KnownOcrModel::PpOcrV6Tiny, KnownOcrModel::PpOcrV6Small],
            config.services.ocr.idle_timeout,
            config.services.ocr.max_loaded,
            config.models.dir.clone(),
        );
        let ocr_vl_models = OcrVlModelCache::new(
            "ocr-vl",
            vec![KnownOcrModel::PpOcrV6Tiny, KnownOcrModel::PaddleOcrVl16],
            config.services.ocr_vl.idle_timeout,
            config.services.ocr_vl.max_loaded,
            config.models.dir.clone(),
        );
        Arc::new(AppState {
            config,
            asr_models,
            tts_models,
            ocr_models,
            ocr_vl_models,
            global_models: GlobalModelCacheLimiter::new(2),
        })
    }
}
