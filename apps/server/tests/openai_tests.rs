use axum::http::StatusCode;
use orchion::{TtsLanguage, TtsSpeaker, TtsVoice};
use orchion_server::openai::{ApiError, SpeechFormat, SpeechRequest};

#[test]
fn error_response_uses_openai_shape() {
    let error = ApiError::invalid_request(
        "unsupported voice",
        Some("voice"),
        Some("unsupported_voice"),
    );
    let (status, body) = error.into_status_body();

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body.error.message, "unsupported voice");
    assert_eq!(body.error.error_type, "invalid_request_error");
    assert_eq!(body.error.param.as_deref(), Some("voice"));
    assert_eq!(body.error.code.as_deref(), Some("unsupported_voice"));
}

#[test]
fn speech_preset_voice_maps_to_tts_voice() {
    let request = SpeechRequest {
        model: "qwen3-tts-0.6b-custom-voice".to_string(),
        input: "Hello".to_string(),
        voice: "ryan".to_string(),
        response_format: SpeechFormat::Wav,
        speed: 1.0,
        language: Some("english".to_string()),
        reference_audio: None,
        reference_text: None,
        voice_prompt: None,
    };

    assert_eq!(
        request.to_tts_voice().unwrap(),
        TtsVoice::Preset {
            speaker: TtsSpeaker::Ryan,
            language: TtsLanguage::English,
        }
    );
}

#[test]
fn speech_clone_voice_requires_reference_audio() {
    let request = SpeechRequest {
        model: "qwen3-tts-0.6b-base".to_string(),
        input: "Hello".to_string(),
        voice: "clone".to_string(),
        response_format: SpeechFormat::Wav,
        speed: 1.0,
        language: Some("english".to_string()),
        reference_audio: None,
        reference_text: Some("Reference text".to_string()),
        voice_prompt: None,
    };

    let error = request.to_tts_voice().unwrap_err();

    assert_eq!(error.error.param.as_deref(), Some("reference_audio"));
}

#[test]
fn speech_design_voice_maps_prompt() {
    let request = SpeechRequest {
        model: "qwen3-tts-1.7b-voice-design".to_string(),
        input: "Hello".to_string(),
        voice: "design".to_string(),
        response_format: SpeechFormat::Wav,
        speed: 1.0,
        language: Some("japanese".to_string()),
        reference_audio: None,
        reference_text: None,
        voice_prompt: Some("warm narrator".to_string()),
    };

    assert_eq!(
        request.to_tts_voice().unwrap(),
        TtsVoice::Design {
            prompt: "warm narrator".to_string(),
            language: TtsLanguage::Japanese,
        }
    );
}

#[test]
fn unsupported_speech_format_is_rejected() {
    let error = SpeechFormat::try_from("mp3").unwrap_err();

    assert_eq!(error.error.param.as_deref(), Some("response_format"));
    assert_eq!(
        error.error.code.as_deref(),
        Some("unsupported_audio_format")
    );
}
