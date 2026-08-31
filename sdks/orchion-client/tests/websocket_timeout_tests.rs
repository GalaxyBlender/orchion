#![cfg(feature = "asr")]

use futures_util::{SinkExt, StreamExt};
use orchion_client::asr::{StreamingEvent, StreamingInputAudioFormat, StreamingStartRequest};
use orchion_client::{Client, ClientConfig, ClientError};
use std::time::Duration;
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::Message;

const CLIENT_TIMEOUT: Duration = Duration::from_millis(100);
const TEST_WATCHDOG: Duration = Duration::from_secs(2);
const BACKPRESSURE_AUDIO_BYTES: usize = 32 * 1024 * 1024;

#[tokio::test]
async fn streaming_connect_times_out_when_the_server_stalls_during_handshake() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (_stream, _) = listener.accept().await.unwrap();
        std::future::pending::<()>().await;
    });
    let config = ClientConfig::new(format!("http://{address}"))
        .unwrap()
        .with_timeout(CLIENT_TIMEOUT)
        .unwrap();
    let client = Client::from_config(config).unwrap();
    let request =
        StreamingStartRequest::new("alibaba/qwen3-asr-flash", StreamingInputAudioFormat::Mp3);

    let result = tokio::time::timeout(TEST_WATCHDOG, client.asr().start_streaming(request))
        .await
        .expect("client did not enforce its configured timeout");

    assert!(matches!(
        result,
        Err(ClientError::Timeout {
            operation: "websocket connect/handshake",
            timeout: CLIENT_TIMEOUT,
        })
    ));
    server.abort();
}

#[tokio::test]
async fn streaming_reads_each_receive_the_configured_timeout() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut websocket = tokio_tungstenite::accept_async(stream).await.unwrap();
        websocket.next().await.unwrap().unwrap();
        websocket
            .send(Message::Text(r#"{"type":"ready"}"#.into()))
            .await
            .unwrap();
        std::future::pending::<()>().await;
    });
    let config = ClientConfig::new(format!("http://{address}"))
        .unwrap()
        .with_timeout(CLIENT_TIMEOUT)
        .unwrap();
    let client = Client::from_config(config).unwrap();
    let request =
        StreamingStartRequest::new("alibaba/qwen3-asr-flash", StreamingInputAudioFormat::Mp3);
    let mut session = tokio::time::timeout(TEST_WATCHDOG, client.asr().start_streaming(request))
        .await
        .expect("streaming handshake exceeded the test watchdog")
        .unwrap();

    let next_event = tokio::time::timeout(TEST_WATCHDOG, session.next_event())
        .await
        .expect("next_event did not enforce the operation timeout");
    assert_timeout(&next_event.unwrap_err(), "websocket next_event");

    let second_read = tokio::time::timeout(TEST_WATCHDOG, session.next_event())
        .await
        .expect("a prior timeout incorrectly exhausted the session timeout");
    assert_timeout(&second_read.unwrap_err(), "websocket next_event");
    server.abort();
}

#[tokio::test]
async fn send_audio_timeout_makes_the_public_session_terminal() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut websocket = tokio_tungstenite::accept_async(stream).await.unwrap();
        websocket.next().await.unwrap().unwrap();
        websocket
            .send(Message::Text(r#"{"type":"ready"}"#.into()))
            .await
            .unwrap();
        std::future::pending::<()>().await;
    });
    let config = ClientConfig::new(format!("http://{address}"))
        .unwrap()
        .with_timeout(CLIENT_TIMEOUT)
        .unwrap();
    let client = Client::from_config(config).unwrap();
    let request =
        StreamingStartRequest::new("alibaba/qwen3-asr-flash", StreamingInputAudioFormat::Mp3);
    let mut session = tokio::time::timeout(TEST_WATCHDOG, client.asr().start_streaming(request))
        .await
        .expect("streaming handshake exceeded the test watchdog")
        .unwrap();

    let error = tokio::time::timeout(
        TEST_WATCHDOG,
        session.send_audio(vec![0; BACKPRESSURE_AUDIO_BYTES]),
    )
    .await
    .expect("send_audio did not enforce the operation timeout")
    .unwrap_err();
    assert_timeout(&error, "websocket send_audio");

    assert_terminal(
        &session.send_audio(vec![1]).await.unwrap_err(),
        "send_audio",
    );
    assert_terminal(&session.finish().await.unwrap_err(), "send_audio");
    assert_terminal(&session.next_event().await.unwrap_err(), "send_audio");
    server.abort();
}

#[tokio::test]
async fn start_streaming_waits_for_ready_before_returning_a_session() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut websocket = tokio_tungstenite::accept_async(stream).await.unwrap();
        websocket.next().await.unwrap().unwrap();
        tokio::time::sleep(Duration::from_millis(40)).await;
        websocket
            .send(Message::Text(r#"{"type":"ready"}"#.into()))
            .await
            .unwrap();
        websocket.next().await.unwrap().unwrap()
    });
    let client = streaming_client(address, Duration::from_secs(1));
    let started_at = tokio::time::Instant::now();

    let mut session = client
        .asr()
        .start_streaming(streaming_request())
        .await
        .unwrap();
    assert!(started_at.elapsed() >= Duration::from_millis(30));
    session.send_audio(vec![1, 2, 3]).await.unwrap();

    assert!(matches!(server.await.unwrap(), Message::Binary(bytes) if bytes.as_ref() == [1, 2, 3]));
}

#[tokio::test]
async fn start_streaming_times_out_waiting_for_ready() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut websocket = tokio_tungstenite::accept_async(stream).await.unwrap();
        websocket.next().await.unwrap().unwrap();
        std::future::pending::<()>().await;
    });
    let client = streaming_client(address, CLIENT_TIMEOUT);

    let result = tokio::time::timeout(
        TEST_WATCHDOG,
        client.asr().start_streaming(streaming_request()),
    )
    .await
    .expect("ready wait did not enforce the operation timeout");
    let Err(error) = result else {
        panic!("streaming unexpectedly started without a ready event");
    };

    assert_timeout(&error, "websocket ready event");
    server.abort();
}

#[tokio::test]
async fn start_streaming_fails_when_the_server_closes_before_ready() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut websocket = tokio_tungstenite::accept_async(stream).await.unwrap();
        websocket.next().await.unwrap().unwrap();
        websocket.close(None).await.unwrap();
    });
    let client = streaming_client(address, CLIENT_TIMEOUT);

    let result = client.asr().start_streaming(streaming_request()).await;
    let Err(error) = result else {
        panic!("streaming unexpectedly started after EOF");
    };

    assert!(
        matches!(error, ClientError::WebSocket { message } if message.contains("before the ready event"))
    );
    server.await.unwrap();
}

#[tokio::test]
async fn start_streaming_rejects_an_unexpected_first_event() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut websocket = tokio_tungstenite::accept_async(stream).await.unwrap();
        websocket.next().await.unwrap().unwrap();
        websocket
            .send(Message::Text(
                r#"{"type":"partial","text":"too soon"}"#.into(),
            ))
            .await
            .unwrap();
    });
    let client = streaming_client(address, CLIENT_TIMEOUT);

    let result = client.asr().start_streaming(streaming_request()).await;
    let Err(error) = result else {
        panic!("streaming unexpectedly accepted a non-ready first event");
    };

    assert!(matches!(error, ClientError::Decode { message } if message.contains("expected ready")));
    server.await.unwrap();
}

#[tokio::test]
async fn streaming_error_event_is_returned_then_completes_the_session() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut websocket = tokio_tungstenite::accept_async(stream).await.unwrap();
        websocket.next().await.unwrap().unwrap();
        websocket
            .send(Message::Text(r#"{"type":"ready"}"#.into()))
            .await
            .unwrap();
        websocket
            .send(Message::Text(
                r#"{"type":"error","error":{"message":"model failed","type":"server_error","param":"model","code":"model_failed"}}"#.into(),
            ))
            .await
            .unwrap();
        std::future::pending::<()>().await;
    });
    let client = streaming_client(address, CLIENT_TIMEOUT);
    let mut session = client
        .asr()
        .start_streaming(streaming_request())
        .await
        .unwrap();

    let event = session.next_event().await.unwrap().unwrap();
    assert!(matches!(
        event,
        StreamingEvent::Error { error }
            if error.message == "model failed" && error.code.as_deref() == Some("model_failed")
    ));
    assert_eq!(session.next_event().await.unwrap(), None);
    assert_terminal(&session.send_audio(vec![1]).await.unwrap_err(), "finish");
    assert_terminal(&session.finish().await.unwrap_err(), "finish");
    server.abort();
}

#[tokio::test]
async fn streaming_disconnect_after_ready_is_a_terminal_unexpected_eof() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut websocket = tokio_tungstenite::accept_async(stream).await.unwrap();
        websocket.next().await.unwrap().unwrap();
        websocket
            .send(Message::Text(r#"{"type":"ready"}"#.into()))
            .await
            .unwrap();
        websocket.close(None).await.unwrap();
    });
    let client = streaming_client(address, CLIENT_TIMEOUT);
    let mut session = client
        .asr()
        .start_streaming(streaming_request())
        .await
        .unwrap();

    assert!(matches!(
        session.next_event().await.unwrap_err(),
        ClientError::UnexpectedEof {
            stream: "asr_streaming"
        }
    ));
    assert_terminal(&session.next_event().await.unwrap_err(), "next_event");
    assert_terminal(
        &session.send_audio(vec![1]).await.unwrap_err(),
        "next_event",
    );
    assert_terminal(&session.finish().await.unwrap_err(), "next_event");
    server.await.unwrap();
}

#[tokio::test]
async fn streaming_final_event_completes_the_session() {
    assert_streaming_terminal(
        r#"{"type":"final","text":"complete transcript"}"#,
        StreamingEvent::Final {
            text: "complete transcript".to_string(),
        },
    )
    .await;
}

#[tokio::test]
async fn streaming_completed_event_completes_the_session() {
    assert_streaming_terminal(r#"{"type":"completed"}"#, StreamingEvent::Completed).await;
}

#[tokio::test]
async fn start_streaming_preserves_a_structured_server_error_before_ready() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut websocket = tokio_tungstenite::accept_async(stream).await.unwrap();
        websocket.next().await.unwrap().unwrap();
        websocket
            .send(Message::Text(
                r#"{"type":"error","error":{"message":"bad model","type":"invalid_request_error","param":"model","code":"invalid_model"}}"#.into(),
            ))
            .await
            .unwrap();
    });
    let client = streaming_client(address, CLIENT_TIMEOUT);

    let result = client.asr().start_streaming(streaming_request()).await;
    let Err(error) = result else {
        panic!("streaming unexpectedly started after a server error");
    };

    assert!(matches!(
        error,
        ClientError::StreamingServer { error }
            if error.message == "bad model"
                && error.param.as_deref() == Some("model")
                && error.code.as_deref() == Some("invalid_model")
    ));
    server.await.unwrap();
}

fn streaming_client(address: std::net::SocketAddr, timeout: Duration) -> Client {
    let config = ClientConfig::new(format!("http://{address}"))
        .unwrap()
        .with_timeout(timeout)
        .unwrap();
    Client::from_config(config).unwrap()
}

fn streaming_request() -> StreamingStartRequest {
    StreamingStartRequest::new("alibaba/qwen3-asr-flash", StreamingInputAudioFormat::Mp3)
}

async fn assert_streaming_terminal(event_json: &'static str, expected: StreamingEvent) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut websocket = tokio_tungstenite::accept_async(stream).await.unwrap();
        websocket.next().await.unwrap().unwrap();
        websocket
            .send(Message::Text(r#"{"type":"ready"}"#.into()))
            .await
            .unwrap();
        websocket
            .send(Message::Text(event_json.into()))
            .await
            .unwrap();
        std::future::pending::<()>().await;
    });
    let client = streaming_client(address, CLIENT_TIMEOUT);
    let mut session = client
        .asr()
        .start_streaming(streaming_request())
        .await
        .unwrap();

    assert_eq!(session.next_event().await.unwrap(), Some(expected));
    assert_eq!(session.next_event().await.unwrap(), None);
    assert_terminal(&session.send_audio(vec![1]).await.unwrap_err(), "finish");
    assert_terminal(&session.finish().await.unwrap_err(), "finish");
    server.abort();
}

fn assert_timeout(error: &ClientError, operation: &'static str) {
    assert!(matches!(
        error,
        ClientError::Timeout {
            operation: actual_operation,
            timeout: CLIENT_TIMEOUT,
        } if *actual_operation == operation
    ));
}

fn assert_terminal(error: &ClientError, operation: &'static str) {
    assert!(matches!(
        error,
        ClientError::StreamingSessionTerminated {
            operation: actual_operation,
        } if *actual_operation == operation
    ));
}
