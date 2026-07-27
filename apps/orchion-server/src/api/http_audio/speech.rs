use crate::api::http_shared::{
    authorize, is_multipart, parse_multipart_value, read_text_field, required_multipart_field,
    run_inference_owned, write_multipart_file_to_temp_file,
};
use crate::api::openai::{ApiError, SpeechFormat, SpeechRequest};
use crate::application::ServerApplication;
use crate::application::speech::{SpeechCommand, synthesize};
use axum::Json;
use axum::body::Body;
use axum::extract::{FromRequest, Multipart, Request, State};
use axum::http::header::CONTENT_TYPE;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::Response;
use orchion::AudioOutputFormat;
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::{Builder as TempFileBuilder, NamedTempFile};

pub(in crate::api) async fn create_speech<S>(
    State(state): State<Arc<S>>,
    headers: HeaderMap,
    request: Request,
) -> Result<Response, ApiError>
where
    S: ServerApplication,
{
    authorize(state.as_ref(), &headers)?;
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
    create_speech_from_command(state, speech_command(request), Vec::new()).await
}

#[allow(
    clippy::too_many_lines,
    reason = "multipart parsing and temporary-file ownership form one request transaction"
)]
async fn create_speech_multipart<S>(
    state: Arc<S>,
    mut multipart: Multipart,
) -> Result<Response, ApiError>
where
    S: ServerApplication,
{
    let mut model = None;
    let mut input = None;
    let mut voice = None;
    let mut response_format = None;
    let mut speed = None;
    let mut language = None;
    let mut reference_audio_file = None;
    let mut reference_text = None;
    let mut random_seed = None;
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
                reference_audio_file =
                    Some(write_multipart_file_to_temp_file(field, "reference_audio").await?);
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
            "seed" => random_seed = Some(parse_multipart_value(field, "seed").await?),
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
    if !voice.trim().eq_ignore_ascii_case("clone") {
        return Err(ApiError::invalid_request(
            "multipart speech is only supported for voice clone requests",
            Some("voice"),
            Some("unsupported_voice_input"),
        ));
    }
    let (reference_audio_file, reference_audio_size) = reference_audio_file.ok_or_else(|| {
        ApiError::invalid_request(
            "voice clone requires `reference_audio` file upload",
            Some("reference_audio"),
            Some("missing_required_parameter"),
        )
    })?;
    if reference_audio_size == 0 {
        return Err(ApiError::invalid_request(
            "uploaded reference audio is empty",
            Some("reference_audio"),
            Some("invalid_file"),
        ));
    }
    let reference_wav_file = TempFileBuilder::new()
        .suffix(".wav")
        .tempfile()
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let command = SpeechCommand {
        model: required_multipart_field(model, "model")?,
        input: required_multipart_field(input, "input")?,
        voice,
        output_format: response_format.map(AudioOutputFormat::from),
        speed: speed.unwrap_or(1.0),
        language,
        reference_audio: Some(reference_audio_file.path().to_path_buf()),
        reference_audio_output: Some(reference_wav_file.path().to_path_buf()),
        reference_text,
        voice_prompt: None,
        seed: random_seed,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        max_length,
    };
    create_speech_from_command(
        state,
        command,
        vec![reference_audio_file, reference_wav_file],
    )
    .await
}

async fn create_speech_from_command<S>(
    state: Arc<S>,
    command: SpeechCommand,
    temporary_files: Vec<NamedTempFile>,
) -> Result<Response, ApiError>
where
    S: ServerApplication,
{
    tracing::debug!(
        model = %command.model,
        voice = %command.voice,
        format = ?command.output_format,
        has_language = command.language.is_some(),
        "speech request received"
    );
    let operation_state = Arc::clone(&state);
    let result = run_inference_owned(state.try_acquire_inference(), async move {
        let _temporary_files = temporary_files;
        synthesize(operation_state.as_ref(), command)
            .await
            .map_err(ApiError::from)
    })
    .await?;
    tracing::info!(format = %result.format, "speech request completed");
    Response::builder()
        .status(StatusCode::OK)
        .header(
            CONTENT_TYPE,
            HeaderValue::from_static(result.format.content_type()),
        )
        .body(Body::from(result.bytes))
        .map_err(|error| ApiError::internal(error.to_string()))
}

fn speech_command(request: SpeechRequest) -> SpeechCommand {
    SpeechCommand {
        model: request.model,
        input: request.input,
        voice: request.voice,
        output_format: request.response_format.map(AudioOutputFormat::from),
        speed: request.speed,
        language: request.language,
        reference_audio: request.reference_audio.map(PathBuf::from),
        reference_audio_output: None,
        reference_text: request.reference_text,
        voice_prompt: request.voice_prompt,
        seed: request.seed,
        temperature: request.temperature,
        top_k: request.top_k,
        top_p: request.top_p,
        repetition_penalty: request.repetition_penalty,
        max_length: request.max_length,
    }
}
