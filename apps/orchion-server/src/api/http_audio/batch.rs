use crate::api::activity::ActivityContext;
use crate::api::http_shared::{
    authorize, read_text_field, run_owned, write_multipart_file_to_temp_file,
};
use crate::api::openai::{
    ApiError, TranscriptionFormat, TranscriptionJson, TranscriptionVerboseJson,
};
use crate::api::srt::format_srt;
use crate::application::ServerApplication;
use crate::application::transcription::{TranscriptionCommand, transcribe};
use axum::Json;
use axum::extract::{Extension, Multipart, State};
use axum::http::header::CONTENT_TYPE;
use axum::http::{HeaderMap, HeaderValue};
use axum::response::{IntoResponse, Response};
use orchion::{AsrModel, AsrTimestampGranularity};
use std::sync::Arc;

#[allow(
    clippy::too_many_lines,
    reason = "multipart parsing and cleanup form one request transaction"
)]
pub(in crate::api) async fn create_transcription<S>(
    State(state): State<Arc<S>>,
    headers: HeaderMap,
    activity: Option<Extension<ActivityContext>>,
    mut multipart: Multipart,
) -> Result<Response, ApiError>
where
    S: ServerApplication,
{
    authorize(state.as_ref(), &headers)?;
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
                return Err(ApiError::invalid_request(
                    format!("`{name}` is not supported for batch transcription"),
                    Some(name.as_str()),
                    Some("unsupported_parameter"),
                ));
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
    if let Some(Extension(activity)) = &activity {
        if let Ok(model_id) = AsrModel::parse(&model)
            && state
                .api_policy()
                .asr
                .as_ref()
                .is_some_and(|policy| policy.models.contains(&model_id))
        {
            activity.set_model(model_id.to_string());
        }
        activity.set_input_bytes(audio_bytes);
    }
    let audio_path = audio_file.path().to_path_buf();
    tracing::debug!(
        model = %model,
        language = ?language,
        response_format = ?response_format,
        audio_bytes,
        "transcription request received"
    );
    let operation_state = Arc::clone(&state);
    let result = run_owned(async move {
        let _audio_file = audio_file;
        transcribe(
            operation_state.as_ref(),
            TranscriptionCommand {
                audio_path,
                model,
                language,
                with_segments: use_segments,
            },
        )
        .await
        .map_err(ApiError::from)
    })
    .await?;
    let transcript = result.transcript;
    let duration = result.duration;
    tracing::info!(format = ?response_format, "transcription request completed");

    Ok(match response_format {
        TranscriptionFormat::Json => Json(TranscriptionJson {
            text: transcript.text,
        })
        .into_response(),
        TranscriptionFormat::VerboseJson => Json(TranscriptionVerboseJson {
            text: transcript.text,
            language: transcript.language,
            duration,
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

pub(in crate::api) fn parse_timestamp_granularities(values: &[String]) -> Result<bool, ApiError> {
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
