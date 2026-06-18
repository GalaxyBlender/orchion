use crate::audio::{encode_wav, write_temp_file};
use crate::config::{parse_asr_model, parse_tts_model};
use crate::docs;
use crate::openai::{
    ApiError, SpeechRequest, TranscriptionFormat, TranscriptionJson, TranscriptionVerboseJson,
    content_type_for,
};
use crate::state::AppState;
use axum::body::Body;
use axum::extract::{Multipart, State};
use axum::http::header::CONTENT_TYPE;
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use orchion::AsrOptions;
use std::sync::Arc;
use tower_http::trace::TraceLayer;

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/audio/speech", post(create_speech))
        .route("/v1/audio/transcriptions", post(create_transcription))
        .merge(docs::swagger_ui())
        .with_state(state)
        .layer(TraceLayer::new_for_http())
}

async fn healthz() -> &'static str {
    "ok"
}

async fn create_speech(
    State(state): State<Arc<AppState>>,
    Json(request): Json<SpeechRequest>,
) -> Result<Response, ApiError> {
    request.validate()?;
    let requested =
        parse_tts_model(&request.model).map_err(|_| ApiError::model_not_loaded(&request.model))?;
    if requested != state.tts.model() {
        return Err(ApiError::model_not_loaded(&request.model));
    }
    let format = request.response_format;
    let voice = request.to_tts_voice()?;
    let audio = state.tts.synthesize(request.input, voice).await?;
    let bytes = encode_wav(&audio).map_err(|error| ApiError::internal(error.to_string()))?;
    Response::builder()
        .status(StatusCode::OK)
        .header(
            CONTENT_TYPE,
            HeaderValue::from_static(content_type_for(format)),
        )
        .body(Body::from(bytes))
        .map_err(|error| ApiError::internal(error.to_string()))
}

async fn create_transcription(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<Response, ApiError> {
    let mut audio_bytes = None;
    let mut audio_suffix = ".wav".to_string();
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
                if let Some(file_name) = field.file_name() {
                    if let Some(extension) = std::path::Path::new(file_name).extension() {
                        audio_suffix = format!(".{}", extension.to_string_lossy());
                    }
                }
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
    let requested = parse_asr_model(&model).map_err(|_| ApiError::model_not_loaded(&model))?;
    if requested != state.asr.model() {
        return Err(ApiError::model_not_loaded(&model));
    }
    let audio_bytes = audio_bytes.ok_or_else(|| {
        ApiError::invalid_request(
            "`file` is required",
            Some("file"),
            Some("missing_required_parameter"),
        )
    })?;
    let upload = write_temp_file(&audio_bytes, &audio_suffix)
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let transcript = state
        .asr
        .transcribe_file_with(
            &upload,
            AsrOptions {
                language,
                ..Default::default()
            },
        )
        .await?;

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
