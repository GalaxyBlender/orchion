use axum::body::{Body, Bytes};
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use http_body_util::BodyExt as _;
use std::convert::Infallible;
use std::time::Duration;

pub(crate) const KEEP_ALIVE_FRAME: Bytes = Bytes::from_static(b": keep-alive\n\n");

pub(crate) fn response(body: Body) -> Response {
    let mut response = (
        StatusCode::OK,
        [("content-type", "text/event-stream")],
        body,
    )
        .into_response();
    set_headers(&mut response);
    response
}

pub(crate) fn set_headers(response: &mut Response) {
    response.headers_mut().insert(
        "cache-control",
        HeaderValue::from_static("no-cache, no-transform"),
    );
    response
        .headers_mut()
        .insert("x-accel-buffering", HeaderValue::from_static("no"));
}

pub(crate) fn numbered_with_keepalive(body: Body, interval: Duration) -> Body {
    let stream = async_stream::stream! {
        let mut body = body;
        let mut next_id = 1_u64;
        loop {
            let frame = tokio::select! {
                () = tokio::time::sleep(interval) => {
                    yield Ok::<Bytes, Infallible>(KEEP_ALIVE_FRAME);
                    continue;
                }
                frame = body.frame() => frame,
            };
            let Some(frame) = frame else { break };
            let Ok(frame) = frame else { break };
            let Ok(data) = frame.into_data() else { continue };
            if data.starts_with(b":") {
                yield Ok(data);
            } else {
                yield Ok(number_frame(next_id, &data));
                next_id += 1;
            }
        }
    };
    Body::from_stream(stream)
}

pub(crate) fn keepalive(body: Body, interval: Duration) -> Body {
    let stream = async_stream::stream! {
        let mut body = body;
        loop {
            let frame = tokio::select! {
                () = tokio::time::sleep(interval) => {
                    yield Ok::<Bytes, Infallible>(KEEP_ALIVE_FRAME);
                    continue;
                }
                frame = body.frame() => frame,
            };
            let Some(frame) = frame else { break };
            let Ok(frame) = frame else { break };
            if let Ok(data) = frame.into_data() {
                yield Ok(data);
            }
        }
    };
    Body::from_stream(stream)
}

pub(crate) fn number_frame(id: u64, data: &[u8]) -> Bytes {
    let mut frame = Vec::with_capacity(data.len() + 24);
    frame.extend_from_slice(format!("id: {id}\n").as_bytes());
    frame.extend_from_slice(data);
    Bytes::from(frame)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn idle_stream_emits_exact_unnumbered_keepalive_comment() {
        let source =
            Body::from_stream(futures_util::stream::pending::<Result<Bytes, Infallible>>());
        let mut body = keepalive(source, Duration::from_millis(1));
        let frame = tokio::time::timeout(Duration::from_secs(1), body.frame())
            .await
            .unwrap()
            .unwrap()
            .unwrap()
            .into_data()
            .unwrap();
        assert_eq!(frame, KEEP_ALIVE_FRAME);
        assert!(!frame.starts_with(b"id:"));
    }

    #[tokio::test]
    async fn numbered_stream_assigns_one_based_ids_to_complete_data_frames() {
        let source = Body::from_stream(futures_util::stream::iter([
            Ok::<_, Infallible>(Bytes::from_static(b"data: one\n\n")),
            Ok(Bytes::from_static(b"data: [DONE]\n\n")),
        ]));
        let body = numbered_with_keepalive(source, Duration::from_secs(60));
        let bytes = body.collect().await.unwrap().to_bytes();
        assert_eq!(bytes, "id: 1\ndata: one\n\nid: 2\ndata: [DONE]\n\n");
    }
}
