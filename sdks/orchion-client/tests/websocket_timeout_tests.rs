#![cfg(feature = "asr")]

use futures_util::StreamExt;
use orchion_client::asr::{StreamingInputAudioFormat, StreamingStartRequest};
use orchion_client::{Client, ClientConfig, ClientError};
use std::time::Duration;
use tokio::net::TcpListener;

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
