use axum::http::StatusCode;
use orchion::{TtsLanguage, TtsOptions, TtsSpeaker, TtsVoice};
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
        response_format: Some(SpeechFormat::Wav),
        speed: 1.0,
        language: Some("english".to_string()),
        reference_audio: None,
        reference_text: None,
        voice_prompt: None,
        seed: None,
        temperature: None,
        top_k: None,
        top_p: None,
        repetition_penalty: None,
        max_length: None,
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
        response_format: Some(SpeechFormat::Wav),
        speed: 1.0,
        language: Some("english".to_string()),
        reference_audio: None,
        reference_text: Some("Reference text".to_string()),
        voice_prompt: None,
        seed: None,
        temperature: None,
        top_k: None,
        top_p: None,
        repetition_penalty: None,
        max_length: None,
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
        response_format: Some(SpeechFormat::Wav),
        speed: 1.0,
        language: Some("japanese".to_string()),
        reference_audio: None,
        reference_text: None,
        voice_prompt: Some("warm narrator".to_string()),
        seed: None,
        temperature: None,
        top_k: None,
        top_p: None,
        repetition_penalty: None,
        max_length: None,
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
fn speech_options_default_to_seed_42_and_upstream_sampling_defaults() {
    let request = SpeechRequest {
        model: "qwen3-tts-0.6b-custom-voice".to_string(),
        input: "Hello".to_string(),
        voice: "ryan".to_string(),
        response_format: Some(SpeechFormat::Wav),
        speed: 1.0,
        language: Some("english".to_string()),
        reference_audio: None,
        reference_text: None,
        voice_prompt: None,
        seed: None,
        temperature: None,
        top_k: None,
        top_p: None,
        repetition_penalty: None,
        max_length: None,
    };

    let options = request.to_tts_options();
    let upstream_defaults = TtsOptions::default();

    assert_eq!(options.seed, Some(42));
    assert_eq!(options.max_length, upstream_defaults.max_length);
    assert_eq!(options.temperature, upstream_defaults.temperature);
    assert_eq!(options.top_k, upstream_defaults.top_k);
    assert_eq!(options.top_p, upstream_defaults.top_p);
    assert_eq!(
        options.repetition_penalty,
        upstream_defaults.repetition_penalty
    );
}

#[test]
fn speech_options_accept_qwen3_tts_sampling_overrides() {
    let request = SpeechRequest {
        model: "qwen3-tts-1.7b".to_string(),
        input: "Hello".to_string(),
        voice: "ryan".to_string(),
        response_format: Some(SpeechFormat::Wav),
        speed: 1.0,
        language: Some("english".to_string()),
        reference_audio: None,
        reference_text: None,
        voice_prompt: None,
        seed: Some(7),
        temperature: Some(0.6),
        top_k: Some(30),
        top_p: Some(0.8),
        repetition_penalty: Some(1.1),
        max_length: Some(256),
    };

    let options = request.to_tts_options();

    assert_eq!(options.seed, Some(7));
    assert_eq!(options.max_length, 256);
    assert_eq!(options.temperature, 0.6);
    assert_eq!(options.top_k, 30);
    assert_eq!(options.top_p, 0.8);
    assert_eq!(options.repetition_penalty, 1.1);
}

#[test]
fn speech_options_reject_invalid_sampling_values() {
    let mut request = SpeechRequest {
        model: "qwen3-tts-0.6b-custom-voice".to_string(),
        input: "Hello".to_string(),
        voice: "ryan".to_string(),
        response_format: Some(SpeechFormat::Wav),
        speed: 1.0,
        language: Some("english".to_string()),
        reference_audio: None,
        reference_text: None,
        voice_prompt: None,
        seed: None,
        temperature: Some(0.0),
        top_k: None,
        top_p: None,
        repetition_penalty: None,
        max_length: None,
    };

    let error = request.validate().unwrap_err();
    assert_eq!(error.error.param.as_deref(), Some("temperature"));

    request.temperature = Some(0.7);
    request.top_k = Some(0);
    let error = request.validate().unwrap_err();
    assert_eq!(error.error.param.as_deref(), Some("top_k"));

    request.top_k = Some(50);
    request.top_p = Some(1.5);
    let error = request.validate().unwrap_err();
    assert_eq!(error.error.param.as_deref(), Some("top_p"));

    request.top_p = Some(0.9);
    request.repetition_penalty = Some(0.0);
    let error = request.validate().unwrap_err();
    assert_eq!(error.error.param.as_deref(), Some("repetition_penalty"));

    request.repetition_penalty = Some(1.05);
    request.max_length = Some(0);
    let error = request.validate().unwrap_err();
    assert_eq!(error.error.param.as_deref(), Some("max_length"));
}

#[test]
fn unsupported_speech_format_is_rejected() {
    let error = SpeechFormat::try_from("ogg").unwrap_err();

    assert_eq!(error.error.param.as_deref(), Some("response_format"));
    assert_eq!(
        error.error.code.as_deref(),
        Some("unsupported_audio_format")
    );
}

#[test]
fn speech_formats_match_openai_values() {
    let cases = [
        ("wav", SpeechFormat::Wav, "audio/wav"),
        ("mp3", SpeechFormat::Mp3, "audio/mpeg"),
        ("aac", SpeechFormat::Aac, "audio/aac"),
        ("opus", SpeechFormat::Opus, "audio/ogg"),
        ("flac", SpeechFormat::Flac, "audio/flac"),
        ("pcm", SpeechFormat::Pcm, "audio/pcm"),
    ];

    for (value, expected, content_type) in cases {
        let format = SpeechFormat::try_from(value).unwrap();
        assert_eq!(format, expected);
        assert_eq!(
            orchion_server::openai::content_type_for(format),
            content_type
        );
    }
}
