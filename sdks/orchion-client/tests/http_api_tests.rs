#[cfg(feature = "models")]
#[tokio::test]
async fn list_models_sends_auth_and_decodes_typed_models() {
    use orchion_client::models::{ModelSubtype, ModelType};
    use orchion_client::{Client, ClientConfig};
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .and(header("Authorization", "Bearer secret"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "object": "list",
            "data": [{
                "id": "Qwen/Qwen3-ASR-Flash",
                "object": "model",
                "created": 0,
                "owned_by": "orchion",
                "type": "asr",
                "subtype": "standard"
            }]
        })))
        .mount(&server)
        .await;

    let config = ClientConfig::new(server.uri())
        .unwrap()
        .with_api_key("secret");
    let client = Client::from_config(config).unwrap();

    let models = client.models().list().await.unwrap();

    assert_eq!(models.object, "list");
    assert_eq!(models.data[0].id, "Qwen/Qwen3-ASR-Flash");
    assert_eq!(models.data[0].model_type, ModelType::Asr);
    assert_eq!(models.data[0].subtype, Some(ModelSubtype::Standard));
}

#[cfg(feature = "models")]
#[tokio::test]
async fn list_models_preserves_the_base_url_path_prefix() {
    use orchion_client::Client;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/orchion/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "object": "list",
            "data": []
        })))
        .expect(1)
        .mount(&server)
        .await;
    let client = Client::new(format!("{}/orchion/", server.uri())).unwrap();

    let models = client.models().list().await.unwrap();

    assert!(models.data.is_empty());
}

#[cfg(feature = "asr")]
#[tokio::test]
async fn transcribe_file_posts_multipart_and_decodes_json() {
    use orchion_client::asr::{TranscriptionFormat, TranscriptionRequest, TranscriptionResponse};
    use orchion_client::{Client, ClientConfig};
    use wiremock::matchers::{body_string_contains, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .and(header("Authorization", "Bearer secret"))
        .and(body_string_contains("Qwen/Qwen3-ASR-Flash"))
        .and(body_string_contains("response_format"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "text": "hello world"
        })))
        .mount(&server)
        .await;

    let config = ClientConfig::new(server.uri())
        .unwrap()
        .with_api_key("secret");
    let client = Client::from_config(config).unwrap();
    let request = TranscriptionRequest::new("Qwen/Qwen3-ASR-Flash", "audio.wav")
        .with_file_bytes(b"fake wav".to_vec())
        .with_response_format(TranscriptionFormat::Json);

    let response = client.asr().transcribe(request).await.unwrap();

    assert_eq!(
        response,
        TranscriptionResponse::Json {
            text: "hello world".to_string()
        }
    );
}

#[cfg(feature = "asr")]
#[tokio::test]
async fn transcribe_text_format_returns_text_response() {
    use orchion_client::asr::{TranscriptionFormat, TranscriptionRequest, TranscriptionResponse};
    use orchion_client::{Client, ClientConfig};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(200).set_body_string("plain transcript"))
        .mount(&server)
        .await;

    let config = ClientConfig::new(server.uri()).unwrap();
    let client = Client::from_config(config).unwrap();
    let request = TranscriptionRequest::new("Qwen/Qwen3-ASR-Flash", "audio.wav")
        .with_file_bytes(b"fake wav".to_vec())
        .with_response_format(TranscriptionFormat::Text);

    let response = client.asr().transcribe(request).await.unwrap();

    assert_eq!(
        response,
        TranscriptionResponse::Text("plain transcript".to_string())
    );
}

#[cfg(feature = "asr")]
#[tokio::test]
async fn streaming_handshake_preserves_the_server_error_body() {
    use orchion_client::asr::{StreamingInputAudioFormat, StreamingStartRequest};
    use orchion_client::{Client, ClientError};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/audio/transcriptions/stream"))
        .respond_with(ResponseTemplate::new(401).set_body_json(serde_json::json!({
            "error": {
                "message": "invalid API key",
                "type": "invalid_request_error",
                "param": null,
                "code": "invalid_api_key"
            }
        })))
        .mount(&server)
        .await;
    let client = Client::new(server.uri()).unwrap();
    let request =
        StreamingStartRequest::new("Qwen/Qwen3-ASR-Flash", StreamingInputAudioFormat::Mp3);

    let Err(error) = client.asr().start_streaming(request).await else {
        panic!("streaming handshake unexpectedly succeeded");
    };

    match error {
        ClientError::Http {
            status,
            error: Some(error),
            ..
        } => {
            assert_eq!(status, reqwest::StatusCode::UNAUTHORIZED);
            assert_eq!(error.message, "invalid API key");
            assert_eq!(error.code.as_deref(), Some("invalid_api_key"));
        }
        unexpected => panic!("unexpected error variant: {unexpected:?}"),
    }
}

#[cfg(feature = "tts")]
#[tokio::test]
async fn create_speech_posts_json_and_returns_audio_bytes() {
    use orchion_client::tts::{SpeechFormat, SpeechRequest};
    use orchion_client::{Client, ClientConfig};
    use wiremock::matchers::{body_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .and(body_json(serde_json::json!({
            "model": "Qwen/Qwen3-TTS-Flash",
            "input": "hello",
            "voice": "Serena",
            "response_format": "wav",
            "speed": 1.0
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "audio/wav")
                .set_body_bytes(vec![1, 2, 3]),
        )
        .mount(&server)
        .await;

    let config = ClientConfig::new(server.uri()).unwrap();
    let client = Client::from_config(config).unwrap();
    let request = SpeechRequest::new("Qwen/Qwen3-TTS-Flash", "hello", "Serena")
        .with_response_format(SpeechFormat::Wav);

    let response = client.tts().create_speech(request).await.unwrap();

    assert_eq!(response.content_type.as_deref(), Some("audio/wav"));
    assert_eq!(response.bytes.as_ref(), &[1, 2, 3]);
}

#[cfg(feature = "tts")]
#[tokio::test]
async fn create_speech_rejects_clone_voice() {
    use orchion_client::tts::SpeechRequest;
    use orchion_client::{Client, ClientError};

    let client = Client::new("http://localhost:8080").unwrap();
    let request = SpeechRequest::new("Qwen/Qwen3-TTS-Flash", "hello", " CLONE ");

    let error = client.tts().create_speech(request).await.unwrap_err();

    assert!(matches!(error, ClientError::BuildRequest { .. }));
}

#[cfg(feature = "ocr")]
#[tokio::test]
async fn create_ocr_posts_multipart_and_decodes_json() {
    use orchion_client::Client;
    use orchion_client::ocr::{OcrRequest, OcrResponse};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/ocr"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "model": "PaddlePaddle/PP-OCRv6_tiny",
            "format": "json",
            "text": "invoice",
            "regions": [],
            "layout_blocks": [],
            "usage": {"input_pages": 1}
        })))
        .mount(&server)
        .await;

    let client = Client::new(server.uri()).unwrap();
    let request = OcrRequest::new("image.png")
        .with_file_bytes(vec![137, 80, 78, 71])
        .with_model("PaddlePaddle/PP-OCRv6_tiny");

    let response = client.ocr().recognize(request).await.unwrap();

    let OcrResponse::Json(body) = response else {
        panic!("expected JSON OCR response");
    };

    assert_eq!(body.text, "invoice");
    assert_eq!(body.usage.input_pages, 1);

    let requests = server.received_requests().await.unwrap();
    assert!(
        requests[0]
            .body
            .windows(b"PaddlePaddle/PP-OCRv6_tiny".len())
            .any(|window| window == b"PaddlePaddle/PP-OCRv6_tiny")
    );
    assert!(
        !requests[0]
            .body
            .windows(b"response_format".len())
            .any(|window| window == b"response_format"),
        "an omitted response format must use the server default"
    );
}

#[cfg(feature = "pdf")]
#[tokio::test]
async fn render_pdf_images_posts_multipart_and_returns_zip_bytes() {
    use orchion_client::Client;
    use orchion_client::pdf::{PdfImageFormat, PdfImagesRequest};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/pdf/images"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/zip")
                .set_body_bytes(vec![80, 75, 3, 4]),
        )
        .mount(&server)
        .await;

    let client = Client::new(server.uri()).unwrap();
    let request = PdfImagesRequest::new("doc.pdf")
        .with_file_bytes(b"%PDF".to_vec())
        .with_response_format(PdfImageFormat::Png)
        .with_pages("1,3-5")
        .with_scale(2.0);

    let response = client.pdf().render_images(request).await.unwrap();

    assert_eq!(response.content_type.as_deref(), Some("application/zip"));
    assert_eq!(response.bytes.as_ref(), &[80, 75, 3, 4]);

    let requests = server.received_requests().await.unwrap();
    assert!(
        requests[0]
            .body
            .windows(b"response_format".len())
            .any(|window| window == b"response_format")
    );
    assert!(
        requests[0]
            .body
            .windows(b"png".len())
            .any(|window| window == b"png")
    );
}

#[cfg(feature = "models")]
#[tokio::test]
async fn http_error_preserves_openai_style_error_body() {
    use orchion_client::Client;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(401).set_body_json(serde_json::json!({
            "error": {
                "message": "invalid API key",
                "type": "invalid_request_error",
                "param": null,
                "code": "invalid_api_key"
            }
        })))
        .mount(&server)
        .await;

    let client = Client::new(server.uri()).unwrap();
    let error = client.models().list().await.unwrap_err();

    match error {
        orchion_client::ClientError::Http {
            status,
            error: Some(error),
            ..
        } => {
            assert_eq!(status, reqwest::StatusCode::UNAUTHORIZED);
            assert_eq!(error.message, "invalid API key");
            assert_eq!(error.error_type, "invalid_request_error");
            assert_eq!(error.code.as_deref(), Some("invalid_api_key"));
        }
        unexpected => panic!("unexpected error variant: {unexpected:?}"),
    }
}
