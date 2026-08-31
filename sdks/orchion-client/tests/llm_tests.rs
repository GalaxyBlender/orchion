#![cfg(feature = "llm")]

use orchion_client::llm::{
    ChatCompletionEvent, ChatCompletionRequest, ChatMessage, FinishReason, MessageRole,
    ResponseInputMessage, ResponseStatus, ResponsesEvent, ResponsesInput, ResponsesRequest,
};
use orchion_client::{Client, ClientConfig, ClientError};
use serde_json::{Value, json};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;
use std::time::Duration;
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

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
