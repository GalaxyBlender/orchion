use crate::api::docs;
use crate::api::openai::{
    ApiError, ModelList, ModelObject, SpeechFormat, SpeechRequest, TranscriptionFormat,
    TranscriptionJson, TranscriptionVerboseJson, content_type_for,
};
use crate::infrastructure::orchion::AppState;
use crate::settings::{parse_asr_model, parse_tts_model};
use axum::body::Body;
use axum::extract::{DefaultBodyLimit, FromRequest, Multipart, Request, State};
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use orchion::{AsrOptions, AudioOutputFormat, encode_tts_audio};
use std::sync::Arc;
use std::time::Instant;
use tempfile::NamedTempFile;
use tower_http::trace::{DefaultMakeSpan, DefaultOnRequest, DefaultOnResponse, TraceLayer};
use tracing::Level;

pub fn router(state: Arc<AppState>) -> Router {
    let max_upload_size = state.config.server.max_upload_size;
    Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/models", get(list_models))
        .route("/v1/audio/speech", post(create_speech))
        .route("/v1/audio/transcriptions", post(create_transcription))
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

async fn healthz() -> &'static str {
    "ok"
}

async fn list_models(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<ModelList>, ApiError> {
    authorize(&state, &headers)?;
    let mut data = Vec::with_capacity(
        state.config.models.asr.available.len() + state.config.models.tts.available.len(),
    );
    data.extend(
        state
            .config
            .models
            .asr
            .available
            .iter()
            .copied()
            .map(ModelObject::new),
    );
    data.extend(
        state
            .config
            .models
            .tts
            .available
            .iter()
            .copied()
            .map(ModelObject::new),
    );
    Ok(Json(ModelList {
        object: "list",
        data,
    }))
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
    let reference_file =
        NamedTempFile::new().map_err(|error| ApiError::internal(error.to_string()))?;
    tokio::fs::write(reference_file.path(), reference_audio)
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let reference_path = reference_file
        .path()
        .to_str()
        .ok_or_else(|| ApiError::internal("reference audio path is not valid UTF-8"))?
        .to_string();
    let request = SpeechRequest {
        model: required_multipart_field(model, "model")?,
        input: required_multipart_field(input, "input")?,
        voice,
        response_format,
        speed: speed.unwrap_or(1.0),
        language,
        reference_audio: Some(reference_path),
        reference_text,
        voice_prompt: None,
        seed,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        max_length,
    };
    let response = create_speech_from_request(state, request).await;
    drop(reference_file);
    response
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
        .unwrap_or_else(|| SpeechFormat::try_from(state.config.defaults.tts.format.as_str()))?;
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
    let mut audio_bytes = None;
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
                audio_bytes = Some(
                    field
                        .bytes()
                        .await
                        .map_err(|error| {
                            ApiError::invalid_request(
                                error.to_string(),
                                Some("file"),
                                Some("invalid_file"),
                            )
                        })?
                        .to_vec(),
                );
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

    if !timestamp_granularities.is_empty() {
        return Err(ApiError::invalid_request(
            "timestamp granularities are not currently supported",
            Some("timestamp_granularities"),
            Some("unsupported_timestamp_granularity"),
        ));
    }

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
    let audio_bytes = audio_bytes.ok_or_else(|| {
        ApiError::invalid_request(
            "`file` is required",
            Some("file"),
            Some("missing_required_parameter"),
        )
    })?;
    tracing::debug!(
        model = %model,
        language = ?language,
        response_format = ?response_format,
        audio_bytes = audio_bytes.len(),
        "transcription request received"
    );
    let transcript = asr
        .transcribe_audio_bytes_with(
            audio_bytes,
            AsrOptions {
                language,
                ..Default::default()
            },
        )
        .await?;
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
        })
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
