#![cfg(feature = "activity")]

use orchion_client::activity::{
    ActivityEvent, ActivityOperation, ActivityOutcome, ActivityQuery, ActivityState,
};
use orchion_client::{Client, ClientConfig, ClientError};
use serde_json::{Value, json};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;
use std::time::Duration;
use wiremock::matchers::{header, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn list_encodes_every_activity_filter_and_auth() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/activity"))
        .and(header("Authorization", "Bearer secret"))
        .and(query_param("limit", "25"))
        .and(query_param("before", "42"))
        .and(query_param("operation", "responses"))
        .and(query_param("outcome", "server_error"))
        .and(query_param("model", "acme/model + one"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "enabled": true,
            "cursor": "43",
            "active": [],
            "history": [],
            "summary": summary(),
            "next_before": null
        })))
        .expect(1)
        .mount(&server)
        .await;
    let client = Client::from_config(
        ClientConfig::new(server.uri())
            .unwrap()
            .with_api_key("secret"),
    )
    .unwrap();
    let query = ActivityQuery::new()
        .with_limit(25)
        .unwrap()
        .with_before(42)
        .with_operation(ActivityOperation::Responses)
        .with_outcome(ActivityOutcome::ServerError)
        .with_model("acme/model + one");

    let page = client.activity().list(query).await.unwrap();

    assert_eq!(page.cursor, "43");
}

#[test]
fn activity_limit_is_validated_locally() {
    for limit in [0, 201] {
        assert!(matches!(
            ActivityQuery::new().with_limit(limit),
            Err(ClientError::BuildRequest { .. })
        ));
    }
    assert!(ActivityQuery::new().with_limit(1).is_ok());
    assert!(ActivityQuery::new().with_limit(200).is_ok());
}

#[tokio::test]
async fn stream_decodes_snapshot_entry_reset_skips_keepalive_and_ends_naturally() {
    let snapshot = payload("1", None, Some(Vec::new()));
    let started = payload("2", Some(entry("2", "in_flight")), None);
    let reset = payload("3", None, None);
    let body = format!(
        ": keep-alive\n\nevent: snapshot\nid: 1\ndata: {snapshot}\n\n\
         event: started\nid: 2\ndata: {started}\n\n\
         : another keep-alive\n\nevent: reset\nid: 3\ndata: {reset}\n\n"
    );
    let (client, server) = sse_mock(&body, "text/event-stream; charset=utf-8").await;

    let mut stream = client.activity().subscribe().await.unwrap();
    assert!(matches!(
        stream.next_event().await.unwrap(),
        Some(ActivityEvent::Snapshot { cursor, active, .. })
            if cursor == "1" && active.is_empty()
    ));
    assert!(matches!(
        stream.next_event().await.unwrap(),
        Some(ActivityEvent::Started { cursor, entry, .. })
            if cursor == "2" && entry.state == ActivityState::InFlight
    ));
    assert!(matches!(
        stream.next_event().await.unwrap(),
        Some(ActivityEvent::Reset { cursor, .. }) if cursor == "3"
    ));
    assert!(stream.next_event().await.unwrap().is_none());
    drop(server);
}

#[tokio::test]
async fn stream_parses_events_split_across_tcp_reads() {
    let body = format!(
        "event: snapshot\nid: 7\ndata: {}\n\n",
        payload("7", None, Some(Vec::new()))
    );
    let chunks = body.bytes().map(|byte| vec![byte]).collect::<Vec<_>>();
    let client = raw_sse_client(chunks, Duration::from_millis(1), body.len(), None);

    let mut stream = client.activity().subscribe().await.unwrap();

    assert!(matches!(
        stream.next_event().await.unwrap(),
        Some(ActivityEvent::Snapshot { cursor, .. }) if cursor == "7"
    ));
    assert!(stream.next_event().await.unwrap().is_none());
}

#[tokio::test]
async fn stream_has_no_total_request_timeout() {
    let body = format!(
        "event: reset\nid: 9\ndata: {}\n\n",
        payload("9", None, None)
    );
    let split = body.len() / 4;
    let chunks = body
        .as_bytes()
        .chunks(split)
        .map(<[u8]>::to_vec)
        .collect::<Vec<_>>();
    let client = raw_sse_client(
        chunks,
        Duration::from_millis(35),
        body.len(),
        Some(Duration::from_millis(100)),
    );

    let mut stream = client.activity().subscribe().await.unwrap();
    let event = stream.next_event().await.unwrap();

    assert!(matches!(event, Some(ActivityEvent::Reset { cursor, .. }) if cursor == "9"));
}

#[tokio::test]
async fn stream_rejects_content_type_unknown_events_and_invalid_payload_shapes() {
    let (client, server) = sse_mock("event: reset\nid: 1\ndata: {}\n\n", "application/json").await;
    let Err(error) = client.activity().subscribe().await else {
        panic!("non-SSE response unexpectedly accepted");
    };
    assert!(matches!(error, ClientError::Decode { .. }));
    drop(server);

    for body in [
        "event: reset\nid: 1\ndata: not-json\n\n".to_string(),
        format!(
            "event: invented\nid: 1\ndata: {}\n\n",
            payload("1", None, None)
        ),
        format!(
            "event: snapshot\nid: 1\ndata: {}\n\n",
            payload("1", Some(entry("1", "in_flight")), Some(Vec::new()))
        ),
        format!(
            "event: completed\nid: 1\ndata: {}\n\n",
            payload("1", Some(entry("1", "in_flight")), None)
        ),
        format!(
            "event: reset\nid: wrong\ndata: {}\n\n",
            payload("1", None, None)
        ),
    ] {
        let (client, server) = sse_mock(&body, "text/event-stream").await;
        let mut stream = client.activity().subscribe().await.unwrap();
        assert!(matches!(
            stream.next_event().await,
            Err(ClientError::Decode { .. })
        ));
        drop(server);
    }

    let invalid_utf8 = raw_sse_client(vec![vec![0xff]], Duration::ZERO, 1, None);
    let mut stream = invalid_utf8.activity().subscribe().await.unwrap();
    assert!(matches!(
        stream.next_event().await,
        Err(ClientError::Decode { .. })
    ));
}

#[tokio::test]
async fn stream_body_timeout_and_disconnect_remain_transport_errors() {
    let stalled = raw_sse_client(
        Vec::new(),
        Duration::from_millis(150),
        100,
        Some(Duration::from_millis(25)),
    );
    let mut stream = stalled.activity().subscribe().await.unwrap();
    let error = stream.next_event().await.unwrap_err();
    match error {
        ClientError::Transport { source } => assert!(source.is_timeout()),
        unexpected => panic!("unexpected timeout error: {unexpected:?}"),
    }

    let body = format!(
        "event: reset\nid: 1\ndata: {}\n\n",
        payload("1", None, None)
    );
    let disconnected = raw_sse_client(
        vec![body.as_bytes().to_vec()],
        Duration::ZERO,
        body.len() + 100,
        None,
    );
    let mut stream = disconnected.activity().subscribe().await.unwrap();
    assert!(stream.next_event().await.unwrap().is_some());
    assert!(matches!(
        stream.next_event().await,
        Err(ClientError::Transport { .. })
    ));
}

async fn sse_mock(body: &str, content_type: &str) -> (Client, MockServer) {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/activity/events"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", content_type)
                .set_body_raw(body, content_type),
        )
        .expect(1)
        .mount(&server)
        .await;
    (Client::new(server.uri()).unwrap(), server)
}

fn raw_sse_client(
    chunks: Vec<Vec<u8>>,
    delay: Duration,
    content_length: usize,
    timeout: Option<Duration>,
) -> Client {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request = [0; 4096];
        let _ = stream.read(&mut request).unwrap();
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {content_length}\r\nConnection: close\r\n\r\n"
        )
        .unwrap();
        stream.flush().unwrap();
        if chunks.is_empty() {
            thread::sleep(delay);
        } else {
            for chunk in chunks {
                stream.write_all(&chunk).unwrap();
                stream.flush().unwrap();
                thread::sleep(delay);
            }
        }
    });
    let config = ClientConfig::new(format!("http://{address}")).unwrap();
    Client::from_config(match timeout {
        Some(timeout) => config.with_timeout(timeout).unwrap(),
        None => config,
    })
    .unwrap()
}

fn payload(cursor: &str, entry: Option<Value>, active: Option<Vec<Value>>) -> Value {
    let mut value = json!({
        "cursor": cursor,
        "summary": summary()
    });
    if let Some(entry) = entry {
        value["entry"] = entry;
    }
    if let Some(active) = active {
        value["active"] = Value::Array(active);
    }
    value
}

fn summary() -> Value {
    json!({
        "active": 0,
        "retained": 0,
        "success_rate": null,
        "p95_duration_ms": null
    })
}

fn entry(id: &str, state: &str) -> Value {
    json!({
        "id": id,
        "state": state,
        "transport": "http",
        "operation": "responses",
        "method": "POST",
        "route": "/v1/responses",
        "started_at_ms": 100
    })
}
