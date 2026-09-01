#![cfg(feature = "llm")]

use orchion_client::llm::{
    ChatCompletionEvent, ChatCompletionRequest, ChatContentPart, ChatImageUrl, ChatMessage,
    ChatReasoningControlRequest, CompletionRequest, EmbeddingEncodingFormat, EmbeddingValue,
    EmbeddingsInput, EmbeddingsRequest, FinishReason, FunctionTool, ImageDetail, MessageRole,
    ReasoningEffort, ResponseInputContentPart, ResponseInputItem, ResponseInputMessage,
    ResponseStatus, ResponsesEvent, ResponsesInput, ResponsesInputTokensRequest, ResponsesRequest,
    ResponsesTextFormat, ToolChoice,
};
use orchion_client::{Client, ClientConfig, ClientError};
use serde_json::{Value, json};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;
use std::time::Duration;
use wiremock::matchers::{body_json, header, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn resumable_chat_start_and_resume_expose_transport_identity() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("x-orchion-resumable", "true"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .insert_header("x-orchion-stream-id", "strm_fixture")
                .insert_header(
                    "x-orchion-completion-id",
                    "chatcmpl_0123456789abcdefghijklmnopqrstuv",
                )
                .set_body_raw("id: 1\ndata: [DONE]\n\n", "text/event-stream"),
        )
        .mount(&server)
        .await;
    let client = Client::new(server.uri()).unwrap();
    let mut stream = client
        .llm()
        .start_resumable_chat_completion(ChatCompletionRequest::new(
            "qwen/test",
            vec![ChatMessage::user("hello")],
        ))
        .await
        .unwrap();
    assert_eq!(stream.stream_id(), Some("strm_fixture"));
    assert_eq!(
        stream.completion_id(),
        Some("chatcmpl_0123456789abcdefghijklmnopqrstuv")
    );
    assert!(stream.next_event().await.unwrap().is_none());
    assert_eq!(stream.last_event_id(), Some(1));

    Mock::given(method("GET"))
        .and(path("/v1/stream"))
        .and(query_param("stream_id", "strm_fixture"))
        .and(header("last-event-id", "1"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw("id: 2\ndata: [DONE]\n\n", "text/event-stream"),
        )
        .mount(&server)
        .await;
    let mut resumed = client
        .llm()
        .resume_chat_completion("strm_fixture", Some(1))
        .await
        .unwrap();
    assert!(resumed.next_event().await.unwrap().is_none());
    assert_eq!(resumed.last_event_id(), Some(2));
}

#[tokio::test]
async fn resumable_sse_requires_nonzero_contiguous_decimal_ids() {
    for body in [
        "data: [DONE]\n\n",
        "id: 0\ndata: [DONE]\n\n",
        "id: nope\ndata: [DONE]\n\n",
        "id: 2\ndata: [DONE]\n\n",
    ] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .insert_header("x-orchion-stream-id", "strm_fixture")
                    .set_body_raw(body, "text/event-stream"),
            )
            .mount(&server)
            .await;
        let client = Client::new(server.uri()).unwrap();
        let mut stream = client
            .llm()
            .start_resumable_completion(CompletionRequest::new("qwen/test", "hello"))
            .await
            .unwrap();
        assert!(matches!(
            stream.next_event().await,
            Err(ClientError::Decode { .. })
        ));
        assert_eq!(stream.last_event_id(), None);
    }

    for second_id in [1, 0, 3] {
        let server = MockServer::start().await;
        let chunk = json!({
            "id":"cmpl-1","object":"text_completion","created":1,"model":"qwen/test",
            "choices":[{"text":"x","index":0,"logprobs":null,"finish_reason":null}],
            "usage":null,"timings":null
        });
        let body = format!("id: 1\ndata: {chunk}\n\nid: {second_id}\ndata: [DONE]\n\n");
        Mock::given(method("POST"))
            .and(path("/v1/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .insert_header("x-orchion-stream-id", "strm_fixture")
                    .set_body_raw(body, "text/event-stream"),
            )
            .mount(&server)
            .await;
        let client = Client::new(server.uri()).unwrap();
        let mut stream = client
            .llm()
            .start_resumable_completion(CompletionRequest::new("qwen/test", "hello"))
            .await
            .unwrap();
        assert!(stream.next_event().await.unwrap().is_some());
        assert!(matches!(
            stream.next_event().await,
            Err(ClientError::Decode { .. })
        ));
        assert_eq!(stream.last_event_id(), Some(1));
    }
}

#[tokio::test]
async fn chat_reasoning_control_builder_and_typed_result_use_fixed_action() {
    let server = MockServer::start().await;
    let id = "chatcmpl_0123456789abcdefghijklmnopqrstuv";
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions/control"))
        .and(body_json(json!({
            "id":id,
            "action":"reasoning_end",
            "model":"qwen/test"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id":id,
            "action":"reasoning_end",
            "success":true,
            "message":null
        })))
        .mount(&server)
        .await;
    let client = Client::new(server.uri()).unwrap();
    let result = client
        .llm()
        .control_chat_reasoning(ChatReasoningControlRequest::new(id).with_model("qwen/test"))
        .await
        .unwrap();
    assert!(result.success);
    assert_eq!(result.action, "reasoning_end");

    let request = ChatCompletionRequest::new("qwen/test", vec![ChatMessage::user("hi")])
        .with_reasoning_control();
    assert!(matches!(
        client.llm().create_chat_completion(request).await,
        Err(ClientError::BuildRequest { .. })
    ));
}

#[tokio::test]
async fn resumed_responses_accepts_a_nonzero_first_payload_sequence() {
    let server = MockServer::start().await;
    let terminal = json!({
        "type":"response.completed",
        "response":response_object("completed"),
        "timings":timings(),
        "sequence_number":8
    });
    Mock::given(method("GET"))
        .and(path("/v1/stream"))
        .and(query_param("stream_id", "strm_fixture"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(
                    format!("id: 9\n{}", response_frame("response.completed", &terminal)),
                    "text/event-stream",
                ),
        )
        .mount(&server)
        .await;
    let client = Client::new(server.uri()).unwrap();
    let mut stream = client
        .llm()
        .resume_response("strm_fixture", Some(8))
        .await
        .unwrap();
    assert!(matches!(
        stream.next_event().await.unwrap(),
        Some(ResponsesEvent::Completed {
            sequence_number: 8,
            ..
        })
    ));
    assert_eq!(stream.last_event_id(), Some(9));
}

#[tokio::test]
async fn resumed_responses_validate_full_replay_and_inferred_lifecycle() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/stream"))
        .and(query_param("stream_id", "strm_full"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(
                    format!(
                        "id: 1\n{}",
                        response_frame(
                            "response.created",
                            &snapshot_payload("response.created", 1)
                        )
                    ),
                    "text/event-stream",
                ),
        )
        .mount(&server)
        .await;
    let client = Client::new(server.uri()).unwrap();
    for cursor in [None, Some(0)] {
        let mut stream = client
            .llm()
            .resume_response("strm_full", cursor)
            .await
            .unwrap();
        assert!(matches!(
            stream.next_event().await.unwrap_err(),
            ClientError::Decode { .. }
        ));
    }

    Mock::given(method("GET"))
        .and(path("/v1/stream"))
        .and(query_param("stream_id", "strm_partial"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(
                    format!(
                        "id: 4\n{}id: 5\n{}",
                        response_frame(
                            "response.created",
                            &snapshot_payload("response.created", 4),
                        ),
                        response_frame(
                            "response.created",
                            &snapshot_payload("response.created", 5),
                        ),
                    ),
                    "text/event-stream",
                ),
        )
        .mount(&server)
        .await;
    let mut stream = client
        .llm()
        .resume_response("strm_partial", Some(3))
        .await
        .unwrap();
    assert!(matches!(
        stream.next_event().await.unwrap(),
        Some(ResponsesEvent::Created { .. })
    ));
    assert!(matches!(
        stream.next_event().await.unwrap_err(),
        ClientError::Decode { .. }
    ));
}

#[tokio::test]
async fn resumable_lookup_and_delete_use_explicit_routes() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/streams/lookup"))
        .and(body_json(json!({"stream_ids":["strm_fixture"]})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"streams":[{
            "stream_id":"strm_fixture","protocol":"chat","status":"active",
            "last_event_id":3,"expires_in_seconds":300
        }]})))
        .mount(&server)
        .await;
    Mock::given(method("DELETE"))
        .and(path("/v1/stream"))
        .and(query_param("stream_id", "strm_fixture"))
        .respond_with(ResponseTemplate::new(204))
        .mount(&server)
        .await;
    let client = Client::new(server.uri()).unwrap();
    let lookup = client
        .llm()
        .lookup_streams(vec!["strm_fixture".into()])
        .await
        .unwrap();
    assert_eq!(lookup.streams[0].last_event_id, 3);
    client.llm().delete_stream("strm_fixture").await.unwrap();
}

#[test]
fn typed_image_parts_serialize_to_chat_and_responses_wire_shapes() {
    let data_url = "data:image/png;base64,AAAA".to_string();
    let chat = ChatCompletionRequest::new(
        "qwen/vision",
        vec![ChatMessage::user("").with_content_parts(vec![
            ChatContentPart::Text {
                text: "before".to_string(),
            },
            ChatContentPart::ImageUrl {
                image_url: ChatImageUrl {
                    url: data_url.clone(),
                    detail: Some(ImageDetail::Auto),
                },
            },
        ])],
    );
    let chat = serde_json::to_value(chat).unwrap();
    assert_eq!(chat["messages"][0]["content"][1]["type"], "image_url");
    assert_eq!(
        chat["messages"][0]["content"][1]["image_url"]["detail"],
        "auto"
    );

    let responses = ResponsesRequest::new(
        "qwen/vision",
        ResponsesInput::items(vec![ResponseInputItem::MessageParts {
            role: MessageRole::User,
            content: vec![ResponseInputContentPart::InputImage {
                image_url: data_url,
                detail: Some(ImageDetail::Auto),
            }],
        }]),
    );
    let responses = serde_json::to_value(responses).unwrap();
    assert_eq!(responses["input"][0]["type"], "message");
    assert_eq!(responses["input"][0]["content"][0]["type"], "input_image");
    assert_eq!(responses["input"][0]["content"][0]["detail"], "auto");
}

#[tokio::test]
async fn chat_json_sends_typed_request_auth_and_decodes_complete_response() {
    let server = MockServer::start().await;
    let request = ChatCompletionRequest::new(
        "qwen/test",
        vec![
            ChatMessage::system("rules"),
            ChatMessage::developer("concise"),
            ChatMessage::user("hello"),
            ChatMessage::assistant("prior"),
        ],
    )
    .with_max_completion_tokens(64)
    .with_temperature(0.7)
    .with_top_p(0.9)
    .with_presence_penalty(0.2)
    .with_frequency_penalty(-0.1)
    .with_stop_sequences(vec!["END".to_string(), "STOP".to_string()])
    .with_seed(42);
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("Authorization", "Bearer secret"))
        .and(body_json(json!({
            "model":"qwen/test",
            "messages":[
                {"role":"system","content":"rules"},
                {"role":"developer","content":"concise"},
                {"role":"user","content":"hello"},
                {"role":"assistant","content":"prior"}
            ],
            "max_completion_tokens":64,
            "temperature":0.7,
            "top_p":0.9,
            "presence_penalty":0.2,
            "frequency_penalty":-0.1,
            "stop":["END","STOP"],
            "seed":42,
            "stream":false
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_response()))
        .expect(1)
        .mount(&server)
        .await;
    let client = authenticated_client(&server);

    let response = client.llm().create_chat_completion(request).await.unwrap();

    assert_eq!(response.object, "chat.completion");
    assert_eq!(response.choices[0].message.role, MessageRole::Assistant);
    assert_eq!(response.choices[0].message.refusal, None);
    assert_eq!(response.choices[0].finish_reason, FinishReason::Stop);
    assert_eq!(response.usage.total_tokens, 3);
    assert_eq!(response.timings.predicted_n, 1);
}

#[tokio::test]
async fn chat_request_serializes_tools_reasoning_and_multiple_choice_options() {
    let server = MockServer::start().await;
    let request = ChatCompletionRequest::new("qwen/test", vec![ChatMessage::user("weather")])
        .with_tools(vec![FunctionTool::new(
            "weather",
            json!({"type":"object","properties":{},"required":[],"additionalProperties":false}),
        )])
        .with_tool_choice(ToolChoice::named("weather"))
        .with_parallel_tool_calls(true)
        .with_reasoning_effort(ReasoningEffort::Low)
        .with_logprobs(2)
        .with_choices(2);
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_json(json!({
            "model":"qwen/test","messages":[{"role":"user","content":"weather"}],
            "tools":[{"type":"function","function":{"name":"weather","parameters":{"type":"object","properties":{},"required":[],"additionalProperties":false},"strict":true}}],
            "tool_choice":{"type":"function","function":{"name":"weather"}},
            "parallel_tool_calls":true,"reasoning_effort":"low","logprobs":true,
            "top_logprobs":2,"n":2,"stream":false
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_response()))
        .mount(&server)
        .await;

    Client::new(server.uri())
        .unwrap()
        .llm()
        .create_chat_completion(request)
        .await
        .unwrap();
}

#[tokio::test]
async fn completion_json_stream_and_input_tokens_use_typed_contracts() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .and(body_json(json!({
            "model":"qwen/test","prompt":"hello","max_tokens":8,"stream":false
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id":"cmpl-1","object":"text_completion","created":1,"model":"qwen/test",
            "choices":[{"text":"world","index":0,"logprobs":null,"finish_reason":"stop"}],
            "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
        })))
        .expect(1)
        .mount(&server)
        .await;
    let response = Client::new(server.uri())
        .unwrap()
        .llm()
        .create_completion(CompletionRequest::new("qwen/test", "hello").with_max_tokens(8))
        .await
        .unwrap();
    assert_eq!(response.choices[0].text, "world");

    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .and(body_json(json!({"model":"qwen/test","prompt":"hello","stream":true})))
        .respond_with(sse_response(
            "data: {\"id\":\"cmpl-1\",\"object\":\"text_completion\",\"created\":1,\"model\":\"qwen/test\",\"choices\":[{\"text\":\"world\",\"index\":0,\"logprobs\":null,\"finish_reason\":null}],\"usage\":null}\n\ndata: [DONE]\n\n",
        ))
        .expect(1)
        .mount(&server)
        .await;
    let mut stream = Client::new(server.uri())
        .unwrap()
        .llm()
        .stream_completion(CompletionRequest::new("qwen/test", "hello"))
        .await
        .unwrap();
    assert_eq!(
        stream.next_event().await.unwrap().unwrap().choices[0].text,
        "world"
    );
    assert!(stream.next_event().await.unwrap().is_none());

    Mock::given(method("POST"))
        .and(path("/v1/responses/input_tokens"))
        .and(body_json(json!({
            "model":"qwen/test","input":"hello","instructions":"concise",
            "tools":[{"type":"function","function":{"name":"weather","parameters":{"type":"object"},"strict":true}}],
            "tool_choice":{"type":"function","function":{"name":"weather"}},
            "parallel_tool_calls":false,
            "reasoning":{"effort":"low"},
            "text":{"format":{"type":"json_object"}}
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object":"response.input_tokens","input_tokens":7
        })))
        .expect(1)
        .mount(&server)
        .await;
    let count = Client::new(server.uri())
        .unwrap()
        .llm()
        .count_response_input_tokens(
            ResponsesInputTokensRequest::new("qwen/test", ResponsesInput::text("hello"))
                .with_instructions("concise")
                .with_tools(vec![FunctionTool::new("weather", json!({"type":"object"}))])
                .with_tool_choice(ToolChoice::named("weather"))
                .with_parallel_tool_calls(false)
                .with_reasoning(ReasoningEffort::Low)
                .with_text_format(ResponsesTextFormat::JsonObject),
        )
        .await
        .unwrap();
    assert_eq!(count.input_tokens, 7);
}

#[tokio::test]
async fn responses_json_always_disables_store_and_decodes_typed_fields() {
    let server = MockServer::start().await;
    let request = ResponsesRequest::new(
        "qwen/test",
        ResponsesInput::messages(vec![ResponseInputMessage::text(MessageRole::User, "hello")]),
    )
    .with_instructions("be concise")
    .with_max_output_tokens(32)
    .with_temperature(1.0)
    .with_top_p(0.9);
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .and(body_json(json!({
            "model":"qwen/test",
            "input":[{"type":"message","role":"user","content":"hello"}],
            "instructions":"be concise",
            "max_output_tokens":32,
            "temperature":1.0,
            "top_p":0.9,
            "store":false,
            "stream":false
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(response_object("completed")))
        .expect(1)
        .mount(&server)
        .await;

    let response = Client::new(server.uri())
        .unwrap()
        .llm()
        .create_response(request)
        .await
        .unwrap();

    assert_eq!(response.status, ResponseStatus::Completed);
    assert_eq!(response.output_text, "hello");
    assert!(!response.store);
    assert_eq!(
        response.usage.unwrap().input_tokens_details.cached_tokens,
        0
    );
    assert_eq!(response.timings.unwrap().prompt_n, 2);
}

#[tokio::test]
async fn embeddings_send_all_options_and_decode_float_and_base64_values() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .and(body_json(json!({
            "model":"qwen/embed",
            "input":["first","second"],
            "dimensions":2,
            "encoding_format":"base64",
            "user":"ignored-user"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object":"list",
            "data":[
                {"object":"embedding","embedding":[0.6,0.8],"index":0},
                {"object":"embedding","embedding":"AACAPwAAAD8=","index":1}
            ],
            "model":"qwen/embed",
            "usage":{"prompt_tokens":4,"total_tokens":4}
        })))
        .expect(1)
        .mount(&server)
        .await;
    let response = Client::new(server.uri())
        .unwrap()
        .llm()
        .create_embeddings(
            EmbeddingsRequest::new(
                "qwen/embed",
                EmbeddingsInput::Texts(vec!["first".to_string(), "second".to_string()]),
            )
            .with_dimensions(2)
            .with_encoding_format(EmbeddingEncodingFormat::Base64)
            .with_user("ignored-user"),
        )
        .await
        .unwrap();
    assert_eq!(response.usage.total_tokens, 4);
    assert!(matches!(
        response.data[0].embedding,
        EmbeddingValue::Float(_)
    ));
    assert!(matches!(
        response.data[1].embedding,
        EmbeddingValue::Base64(_)
    ));
}

#[tokio::test]
async fn embeddings_reject_empty_inputs_before_network_io() {
    let client = Client::new("http://127.0.0.1:9").unwrap();
    let error = client
        .llm()
        .create_embeddings(EmbeddingsRequest::new(
            "qwen/embed",
            EmbeddingsInput::TokenBatches(Vec::new()),
        ))
        .await
        .unwrap_err();
    assert!(matches!(error, ClientError::BuildRequest { .. }));
}

#[tokio::test]
async fn chat_stream_forces_options_decodes_chunks_usage_and_done() {
    let body = format!(
        "data: {}\n\ndata: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
        chat_chunk(&json!({"role":"assistant"}), None, None, None),
        chat_chunk(&json!({"content":"hello"}), None, None, None),
        chat_chunk(&json!({}), None, Some(&chat_usage()), Some(&timings()))
    );
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_json(json!({
            "model":"qwen/test",
            "messages":[{"role":"user","content":"hello"}],
            "stream":true,
            "stream_options":{"include_usage":true}
        })))
        .respond_with(sse_response(&body))
        .expect(1)
        .mount(&server)
        .await;
    let client = Client::new(server.uri()).unwrap();
    let mut stream = client
        .llm()
        .stream_chat_completion(chat_request())
        .await
        .unwrap();

    let Some(ChatCompletionEvent::Chunk(role)) = stream.next_event().await.unwrap() else {
        panic!("missing role chunk");
    };
    assert_eq!(role.choices[0].delta.role, Some(MessageRole::Assistant));
    let Some(ChatCompletionEvent::Chunk(content)) = stream.next_event().await.unwrap() else {
        panic!("missing content chunk");
    };
    assert_eq!(content.choices[0].delta.content.as_deref(), Some("hello"));
    let Some(ChatCompletionEvent::Chunk(usage)) = stream.next_event().await.unwrap() else {
        panic!("missing usage chunk");
    };
    assert!(usage.choices.is_empty());
    assert_eq!(usage.usage.unwrap().total_tokens, 3);
    assert_eq!(usage.timings.unwrap().predicted_n, 1);
    assert!(stream.next_event().await.unwrap().is_none());
    assert!(stream.next_event().await.unwrap().is_none());
}

#[tokio::test]
async fn chat_stream_errors_are_structured_terminal_and_malformed_is_decode() {
    let cases = [
        (
            "data: {\"error\":{\"message\":\"deadline\",\"type\":\"invalid_request_error\",\"param\":null,\"code\":\"request_timeout\"}}\n\n",
            "server",
        ),
        ("data: not-json\n\n", "decode"),
    ];
    for (body, expected) in cases {
        let (client, server) = chat_stream_mock(body).await;
        let mut stream = client
            .llm()
            .stream_chat_completion(chat_request())
            .await
            .unwrap();
        let error = stream.next_event().await.unwrap_err();
        match (expected, error) {
            ("server", ClientError::StreamingServer { error }) => {
                assert_eq!(error.code.as_deref(), Some("request_timeout"));
                assert_eq!(error.param, None);
            }
            ("decode", ClientError::Decode { .. }) => {}
            (_, unexpected) => panic!("unexpected error: {unexpected:?}"),
        }
        assert!(stream.next_event().await.unwrap().is_none());
        drop(server);
    }
}

#[tokio::test]
async fn chat_stream_eof_before_done_is_explicit_and_terminal() {
    let (client, server) = chat_stream_mock(&format!(
        "data: {}\n\n",
        chat_chunk(&json!({"content":"partial"}), None, None, None)
    ))
    .await;
    let mut stream = client
        .llm()
        .stream_chat_completion(chat_request())
        .await
        .unwrap();
    assert!(stream.next_event().await.unwrap().is_some());
    assert!(matches!(
        stream.next_event().await,
        Err(ClientError::UnexpectedEof {
            stream: "chat completion"
        })
    ));
    assert!(stream.next_event().await.unwrap().is_none());
    drop(server);
}

#[tokio::test]
async fn responses_stream_exposes_complete_lifecycle_and_terminal() {
    let body = responses_lifecycle("response.completed", false);
    let (client, server) = responses_stream_mock(&body).await;
    let mut stream = client
        .llm()
        .stream_response(ResponsesRequest::new(
            "qwen/test",
            ResponsesInput::text("hello"),
        ))
        .await
        .unwrap();
    let mut names = Vec::new();
    while let Some(event) = stream.next_event().await.unwrap() {
        names.push(match event {
            ResponsesEvent::Created {
                sequence_number, ..
            } => {
                assert_eq!(sequence_number, 0);
                "created"
            }
            ResponsesEvent::InProgress { .. } => "in_progress",
            ResponsesEvent::OutputItemAdded { .. } => "item_added",
            ResponsesEvent::ContentPartAdded { .. } => "part_added",
            ResponsesEvent::OutputTextDelta { delta, .. } => {
                assert_eq!(delta, "hello");
                "delta"
            }
            ResponsesEvent::OutputTextDone { text, .. } => {
                assert_eq!(text, "hello");
                "text_done"
            }
            ResponsesEvent::ContentPartDone { .. } => "part_done",
            ResponsesEvent::OutputItemDone { .. } => "item_done",
            ResponsesEvent::Completed {
                response, timings, ..
            } => {
                assert_eq!(response.status, ResponseStatus::Completed);
                assert_eq!(timings.predicted_n, 1);
                "completed"
            }
            ResponsesEvent::Incomplete { .. } => "incomplete",
            _ => panic!("unexpected future Responses event"),
        });
    }
    assert_eq!(
        names,
        [
            "created",
            "in_progress",
            "item_added",
            "part_added",
            "delta",
            "text_done",
            "part_done",
            "item_done",
            "completed"
        ]
    );
    assert!(stream.next_event().await.unwrap().is_none());
    drop(server);
}

#[tokio::test]
async fn responses_known_and_forward_compatible_terminal_events_close_the_stream() {
    for (kind, status) in [
        ("response.failed", "failed"),
        ("response.cancelled", "cancelled"),
        ("response.future_terminal", "failed"),
    ] {
        let body = [
            response_frame("response.created", &snapshot_payload("response.created", 0)),
            response_frame(
                "response.in_progress",
                &snapshot_payload("response.in_progress", 1),
            ),
            response_frame(
                kind,
                &json!({"type":kind,"response":response_object(status),"sequence_number":2}),
            ),
        ]
        .concat();
        let (client, server) = responses_stream_mock(&body).await;
        let mut stream = client
            .llm()
            .stream_response(ResponsesRequest::new(
                "qwen/test",
                ResponsesInput::text("hello"),
            ))
            .await
            .unwrap();
        assert!(stream.next_event().await.unwrap().is_some());
        assert!(stream.next_event().await.unwrap().is_some());
        let terminal = stream.next_event().await.unwrap().unwrap();
        match kind {
            "response.failed" => assert!(matches!(terminal, ResponsesEvent::Failed { .. })),
            "response.cancelled" => assert!(matches!(terminal, ResponsesEvent::Cancelled { .. })),
            _ => assert!(matches!(terminal, ResponsesEvent::Unknown { .. })),
        }
        assert!(stream.next_event().await.unwrap().is_none());
        drop(server);
    }
}

#[tokio::test]
async fn responses_stream_decodes_dynamic_reasoning_and_function_events() {
    let events = [
        ("response.created", snapshot_payload("response.created", 0)),
        (
            "response.in_progress",
            snapshot_payload("response.in_progress", 1),
        ),
        (
            "response.reasoning_summary_text.delta",
            json!({"type":"response.reasoning_summary_text.delta","item_id":"rs-1","output_index":0,"summary_index":0,"delta":"think","sequence_number":2}),
        ),
        (
            "response.function_call_arguments.delta",
            json!({"type":"response.function_call_arguments.delta","item_id":"call-1","output_index":1,"delta":"{\"city\":","sequence_number":3}),
        ),
        (
            "response.function_call_arguments.done",
            json!({"type":"response.function_call_arguments.done","item_id":"call-1","output_index":1,"arguments":"{\"city\":\"Paris\"}","sequence_number":4}),
        ),
        (
            "response.completed",
            json!({"type":"response.completed","response":response_object("completed"),"timings":timings(),"sequence_number":5}),
        ),
    ];
    let body = events
        .iter()
        .map(|(name, payload)| response_frame(name, payload))
        .collect::<String>();
    let (client, server) = responses_stream_mock(&body).await;
    let mut stream = client
        .llm()
        .stream_response(ResponsesRequest::new(
            "qwen/test",
            ResponsesInput::text("hello"),
        ))
        .await
        .unwrap();
    let mut dynamic = Vec::new();
    while let Some(event) = stream.next_event().await.unwrap() {
        match event {
            ResponsesEvent::ReasoningSummaryTextDelta { delta, .. }
            | ResponsesEvent::FunctionCallArgumentsDelta { delta, .. } => dynamic.push(delta),
            ResponsesEvent::FunctionCallArgumentsDone { arguments, .. } => dynamic.push(arguments),
            _ => {}
        }
    }
    assert_eq!(dynamic, ["think", "{\"city\":", "{\"city\":\"Paris\"}"]);
    drop(server);
}

#[tokio::test]
async fn responses_incomplete_is_a_successful_terminal() {
    let body = responses_lifecycle("response.incomplete", true);
    let (client, server) = responses_stream_mock(&body).await;
    let mut stream = client
        .llm()
        .stream_response(ResponsesRequest::new(
            "qwen/test",
            ResponsesInput::text("hello"),
        ))
        .await
        .unwrap();
    let mut terminal = None;
    while let Some(event) = stream.next_event().await.unwrap() {
        if let ResponsesEvent::Incomplete { response, .. } = event {
            terminal = Some(response);
        }
    }
    let response = terminal.expect("missing incomplete event");
    assert_eq!(response.status, ResponseStatus::Incomplete);
    assert_eq!(
        response.incomplete_details.unwrap().reason,
        "max_output_tokens"
    );
    drop(server);
}

#[tokio::test]
async fn responses_flat_error_is_structured_and_terminal() {
    let body = "event: error\ndata: {\"type\":\"error\",\"sequence_number\":0,\"code\":\"internal_error\",\"message\":\"failed\",\"param\":\"input\"}\n\n";
    let (client, server) = responses_stream_mock(body).await;
    let mut stream = client
        .llm()
        .stream_response(ResponsesRequest::new(
            "qwen/test",
            ResponsesInput::text("hello"),
        ))
        .await
        .unwrap();
    let error = stream.next_event().await.unwrap_err();
    let ClientError::StreamingServerEvent { error } = error else {
        panic!("unexpected error: {error:?}");
    };
    assert_eq!(error.code.as_deref(), Some("internal_error"));
    assert_eq!(error.message, "failed");
    assert_eq!(error.param.as_deref(), Some("input"));
    assert!(stream.next_event().await.unwrap().is_none());
    drop(server);
}

#[tokio::test]
async fn responses_rejects_eof_sequence_mismatch_event_mismatch_and_duplicate() {
    let cases = [
        (
            response_frame("response.created", &snapshot_payload("response.created", 0)),
            "eof",
        ),
        (
            response_frame("response.created", &snapshot_payload("response.created", 1)),
            "decode",
        ),
        (
            response_frame(
                "response.in_progress",
                &snapshot_payload("response.created", 0),
            ),
            "decode",
        ),
        (
            format!(
                "{}{}",
                response_frame("response.created", &snapshot_payload("response.created", 0)),
                response_frame("response.created", &snapshot_payload("response.created", 1))
            ),
            "duplicate",
        ),
    ];
    for (body, expected) in cases {
        let (client, server) = responses_stream_mock(&body).await;
        let mut stream = client
            .llm()
            .stream_response(ResponsesRequest::new(
                "qwen/test",
                ResponsesInput::text("hello"),
            ))
            .await
            .unwrap();
        let error = if expected == "eof" || expected == "duplicate" {
            stream.next_event().await.unwrap();
            stream.next_event().await.unwrap_err()
        } else {
            stream.next_event().await.unwrap_err()
        };
        if expected == "eof" {
            assert!(matches!(error, ClientError::UnexpectedEof { .. }));
        } else {
            assert!(matches!(error, ClientError::Decode { .. }));
        }
        assert!(stream.next_event().await.unwrap().is_none());
        drop(server);
    }
}

#[tokio::test]
async fn cancelling_pending_next_event_does_not_mark_stream_terminal() {
    let frame = response_frame("response.created", &snapshot_payload("response.created", 0));
    let client = delayed_raw_sse_client(frame, Duration::from_millis(75));
    let mut stream = client
        .llm()
        .stream_response(ResponsesRequest::new(
            "qwen/test",
            ResponsesInput::text("hello"),
        ))
        .await
        .unwrap();

    assert!(
        tokio::time::timeout(Duration::from_millis(10), stream.next_event())
            .await
            .is_err()
    );
    assert!(matches!(
        stream.next_event().await.unwrap(),
        Some(ResponsesEvent::Created { .. })
    ));
}

#[tokio::test]
async fn non_success_llm_response_preserves_structured_server_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "error": {
                "message":"bad messages",
                "type":"invalid_request_error",
                "param":"messages",
                "code":"invalid_parameter"
            }
        })))
        .mount(&server)
        .await;
    let error = Client::new(server.uri())
        .unwrap()
        .llm()
        .create_chat_completion(chat_request())
        .await
        .unwrap_err();
    let ClientError::Http { status, error, .. } = error else {
        panic!("unexpected error: {error:?}");
    };
    assert_eq!(status.as_u16(), 400);
    assert_eq!(error.unwrap().param.as_deref(), Some("messages"));
}

fn authenticated_client(server: &MockServer) -> Client {
    Client::from_config(
        ClientConfig::new(server.uri())
            .unwrap()
            .with_api_key("secret"),
    )
    .unwrap()
}

fn chat_request() -> ChatCompletionRequest {
    ChatCompletionRequest::new("qwen/test", vec![ChatMessage::user("hello")])
}

async fn chat_stream_mock(body: &str) -> (Client, MockServer) {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(sse_response(body))
        .mount(&server)
        .await;
    (Client::new(server.uri()).unwrap(), server)
}

async fn responses_stream_mock(body: &str) -> (Client, MockServer) {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(sse_response(body))
        .mount(&server)
        .await;
    (Client::new(server.uri()).unwrap(), server)
}

fn sse_response(body: &str) -> ResponseTemplate {
    ResponseTemplate::new(200)
        .insert_header("content-type", "text/event-stream")
        .set_body_raw(body, "text/event-stream")
}

fn chat_response() -> Value {
    json!({
        "id":"chatcmpl-1","object":"chat.completion","created":1,"model":"qwen/test",
        "choices":[{"index":0,"message":{"role":"assistant","content":"hello","refusal":null},"finish_reason":"stop","logprobs":null}],
        "usage":chat_usage(),"timings":timings()
    })
}

fn chat_usage() -> Value {
    json!({"prompt_tokens":2,"completion_tokens":1,"total_tokens":3})
}

fn timings() -> Value {
    json!({
        "cache_n":0,"prompt_n":2,"prompt_ms":8.0,"prompt_per_token_ms":4.0,
        "prompt_per_second":250.0,"predicted_n":1,"predicted_ms":5.0,
        "predicted_per_token_ms":5.0,"predicted_per_second":200.0
    })
}

fn chat_chunk(
    delta: &Value,
    finish_reason: Option<&str>,
    usage: Option<&Value>,
    chunk_timings: Option<&Value>,
) -> Value {
    let choices = if usage.is_some() {
        json!([])
    } else {
        json!([{"index":0,"delta":delta,"logprobs":null,"finish_reason":finish_reason}])
    };
    let mut value = json!({
        "id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"qwen/test",
        "choices":choices,"usage":usage
    });
    if let Some(chunk_timings) = chunk_timings {
        value["timings"] = chunk_timings.clone();
    }
    value
}

fn response_object(status: &str) -> Value {
    let complete = status != "in_progress";
    let incomplete = status == "incomplete";
    json!({
        "id":"resp-1","object":"response","created_at":1,"status":status,"background":false,
        "error":null,"incomplete_details":if incomplete { json!({"reason":"max_output_tokens"}) } else { Value::Null },
        "instructions":"be concise","max_output_tokens":32,"model":"qwen/test",
        "output":if complete { json!([{"id":"msg-1","type":"message","status":status,"role":"assistant","content":[{"type":"output_text","text":"hello","annotations":[],"logprobs":[]}] }]) } else { json!([]) },
        "output_text":if complete { "hello" } else { "" },"parallel_tool_calls":false,
        "previous_response_id":null,"reasoning":{"effort":null,"summary":null},"store":false,
        "temperature":1.0,"text":{"format":{"type":"text"},"verbosity":"medium"},
        "tool_choice":"none","tools":[],"top_logprobs":0,"top_p":0.9,"truncation":"disabled",
        "metadata":{},"usage":if complete { responses_usage() } else { Value::Null },"timings":if complete { timings() } else { Value::Null }
    })
}

fn responses_usage() -> Value {
    json!({
        "input_tokens":2,"input_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},
        "output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":3
    })
}

fn snapshot_payload(kind: &str, sequence: u64) -> Value {
    json!({"type":kind,"response":response_object("in_progress"),"sequence_number":sequence})
}

fn response_frame(event: &str, payload: &Value) -> String {
    format!("event: {event}\ndata: {payload}\n\n")
}

fn responses_lifecycle(terminal: &str, incomplete: bool) -> String {
    let item = json!({"id":"msg-1","type":"message","status":"in_progress","role":"assistant","content":[]});
    let part = json!({"type":"output_text","text":"","annotations":[],"logprobs":[]});
    let done_item = json!({"id":"msg-1","type":"message","status":if incomplete { "incomplete" } else { "completed" },"role":"assistant","content":[{"type":"output_text","text":"hello","annotations":[],"logprobs":[]}]});
    let events = vec![
        ("response.created", snapshot_payload("response.created", 0)),
        (
            "response.in_progress",
            snapshot_payload("response.in_progress", 1),
        ),
        (
            "response.output_item.added",
            json!({"type":"response.output_item.added","output_index":0,"item":item,"sequence_number":2}),
        ),
        (
            "response.content_part.added",
            json!({"type":"response.content_part.added","item_id":"msg-1","output_index":0,"content_index":0,"part":part,"sequence_number":3}),
        ),
        (
            "response.output_text.delta",
            json!({"type":"response.output_text.delta","item_id":"msg-1","output_index":0,"content_index":0,"delta":"hello","logprobs":[],"sequence_number":4}),
        ),
        (
            "response.output_text.done",
            json!({"type":"response.output_text.done","item_id":"msg-1","output_index":0,"content_index":0,"text":"hello","logprobs":[],"sequence_number":5}),
        ),
        (
            "response.content_part.done",
            json!({"type":"response.content_part.done","item_id":"msg-1","output_index":0,"content_index":0,"part":{"type":"output_text","text":"hello","annotations":[],"logprobs":[]},"sequence_number":6}),
        ),
        (
            "response.output_item.done",
            json!({"type":"response.output_item.done","output_index":0,"item":done_item,"sequence_number":7}),
        ),
        (
            terminal,
            json!({"type":terminal,"response":response_object(if incomplete { "incomplete" } else { "completed" }),"timings":timings(),"sequence_number":8}),
        ),
    ];
    events
        .into_iter()
        .map(|(event, payload)| response_frame(event, &payload))
        .collect()
}

fn delayed_raw_sse_client(frame: String, delay: Duration) -> Client {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request = [0; 4096];
        let _ = stream.read(&mut request).unwrap();
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            frame.len()
        )
        .unwrap();
        stream.flush().unwrap();
        thread::sleep(delay);
        stream.write_all(frame.as_bytes()).unwrap();
        stream.flush().unwrap();
    });
    Client::new(format!("http://{address}")).unwrap()
}
