use super::contract::{ActivityError, ActivityOperation, ActivityTransport};
use super::store::{ActivityContext, ActivityHub, LiveRequestMetadata};
use axum::body::{Body, Bytes};
use axum::extract::connect_info::ConnectInfo;
use axum::extract::{MatchedPath, Request, State};
use axum::http::header::{CONTENT_LENGTH, USER_AGENT};
use axum::middleware::Next;
use axum::response::Response;
use http_body::{Body as HttpBody, Frame, SizeHint};
use std::net::SocketAddr;
use std::pin::Pin;
use std::task::{Context, Poll};

pub(crate) async fn track_activity(
    State(hub): State<ActivityHub>,
    mut request: Request,
    next: Next,
) -> Response {
    let Some((operation, transport, route)) = classify(&request) else {
        return next.run(request).await;
    };
    let input_bytes = request
        .headers()
        .get(CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse().ok());
    let live_metadata = LiveRequestMetadata::new(
        request
            .extensions()
            .get::<ConnectInfo<SocketAddr>>()
            .map(|ConnectInfo(address)| address.ip().to_string()),
        request
            .headers()
            .get(USER_AGENT)
            .and_then(|value| value.to_str().ok())
            .map(ToOwned::to_owned),
    );
    let Some(context) = hub.start_with_live_metadata(
        operation,
        transport,
        request.method().as_str(),
        route,
        input_bytes,
        live_metadata,
    ) else {
        return next.run(request).await;
    };

    request.extensions_mut().insert(context.clone());
    let mut guard = HttpCompletionGuard::new(context.clone());
    let response = next.run(request).await;
    if context.was_handed_off() && response.status().as_u16() == 101 {
        guard.disarm();
        return response;
    }
    let activity_error = response.extensions().get::<ActivityError>().cloned();
    guard.disarm();
    track_response_body(response, context, activity_error)
}

fn track_response_body(
    response: Response,
    context: ActivityContext,
    activity_error: Option<ActivityError>,
) -> Response {
    let status = response.status().as_u16();
    let content_length = response
        .headers()
        .get(CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse().ok());
    if response.body().is_end_stream() || content_length == Some(0) {
        context.complete_http(status, activity_error);
        return response;
    }

    let (parts, body) = response.into_parts();
    let remaining_bytes = body.size_hint().exact().or(content_length);
    let body = ActivityBody {
        inner: body,
        remaining_bytes,
        completion: Some(HttpResponseCompletion {
            context,
            status,
            activity_error,
            completed: false,
        }),
    };
    Response::from_parts(parts, Body::new(body))
}

fn classify(request: &Request<Body>) -> Option<(ActivityOperation, ActivityTransport, &str)> {
    if request.method() == axum::http::Method::OPTIONS {
        return None;
    }
    let route = request.extensions().get::<MatchedPath>()?.as_str();
    let operation = match (request.method().as_str(), route) {
        ("POST", "/v1/audio/transcriptions") => ActivityOperation::Asr,
        ("GET", "/v1/audio/transcriptions/stream") => ActivityOperation::AsrStream,
        ("POST", "/v1/audio/speech") => ActivityOperation::Tts,
        ("POST", "/v1/ocr") => ActivityOperation::Ocr,
        ("POST", "/v1/pdf/images") => ActivityOperation::Pdf,
        ("POST", "/v1/chat/completions") => ActivityOperation::Chat,
        ("POST", "/v1/responses") => ActivityOperation::Responses,
        _ => return None,
    };
    Some((operation, ActivityTransport::Http, route))
}

struct HttpCompletionGuard {
    context: ActivityContext,
    armed: bool,
}

impl HttpCompletionGuard {
    const fn new(context: ActivityContext) -> Self {
        Self {
            context,
            armed: true,
        }
    }

    const fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for HttpCompletionGuard {
    fn drop(&mut self) {
        if self.armed {
            self.context.cancel();
        }
    }
}

struct ActivityBody {
    inner: Body,
    remaining_bytes: Option<u64>,
    completion: Option<HttpResponseCompletion>,
}

impl HttpBody for ActivityBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        match Pin::new(&mut self.inner).poll_frame(cx) {
            Poll::Ready(None) => {
                if let Some(completion) = self.completion.take() {
                    completion.complete();
                }
                Poll::Ready(None)
            }
            Poll::Ready(Some(Ok(frame))) => {
                if let (Some(remaining), Some(data)) = (&mut self.remaining_bytes, frame.data_ref())
                {
                    *remaining =
                        remaining.saturating_sub(u64::try_from(data.len()).unwrap_or(u64::MAX));
                }
                let completed = frame.is_trailers()
                    || self.remaining_bytes == Some(0)
                    || self.inner.is_end_stream();
                if completed && let Some(completion) = self.completion.take() {
                    completion.complete();
                }
                Poll::Ready(Some(Ok(frame)))
            }
            Poll::Ready(Some(Err(error))) => {
                if let Some(completion) = self.completion.take() {
                    completion.disconnect();
                }
                Poll::Ready(Some(Err(error)))
            }
            Poll::Pending => Poll::Pending,
        }
    }

    fn is_end_stream(&self) -> bool {
        self.completion.is_none() && self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

struct HttpResponseCompletion {
    context: ActivityContext,
    status: u16,
    activity_error: Option<ActivityError>,
    completed: bool,
}

impl HttpResponseCompletion {
    fn complete(mut self) {
        self.context
            .complete_http(self.status, self.activity_error.take());
        self.completed = true;
    }

    fn disconnect(mut self) {
        self.context.disconnect();
        self.completed = true;
    }
}

impl Drop for HttpResponseCompletion {
    fn drop(&mut self) {
        if !self.completed {
            self.context.disconnect();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::activity::{ActivityFilter, ActivityOutcome};
    use crate::application::ActivityPolicy;
    use axum::Router;
    use axum::body::Bytes;
    use axum::http::{HeaderMap, HeaderValue, Method, StatusCode};
    use axum::middleware;
    use axum::routing::post;
    use futures_util::stream;
    use http_body_util::{BodyExt, StreamBody};
    use std::convert::Infallible;
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::time::Duration;
    use tower::ServiceExt;

    #[test]
    fn dropping_an_armed_http_guard_records_cancellation() {
        let hub = ActivityHub::new(ActivityPolicy {
            enabled: true,
            history_capacity: 1,
        });
        let context = hub
            .start(
                ActivityOperation::Asr,
                ActivityTransport::Http,
                "POST",
                "/v1/audio/transcriptions",
                None,
            )
            .unwrap();

        drop(HttpCompletionGuard::new(context));

        let page = hub.page(&ActivityFilter {
            limit: 1,
            ..ActivityFilter::default()
        });
        assert_eq!(page.history[0].outcome, Some(ActivityOutcome::Cancelled));
    }

    #[tokio::test]
    async fn response_body_lifetime_is_included_in_duration() {
        let hub = ActivityHub::new(ActivityPolicy {
            enabled: true,
            history_capacity: 1,
        });
        let app = Router::new()
            .route(
                "/v1/audio/speech",
                post(|| async {
                    Body::from_stream(async_stream::stream! {
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        yield Ok::<_, Infallible>(Bytes::from_static(b"audio"));
                    })
                }),
            )
            .layer(middleware::from_fn_with_state(hub.clone(), track_activity));

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/audio/speech")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(hub.page(&ActivityFilter::default()).active.len(), 1);

        response.into_body().collect().await.unwrap();
        let page = hub.page(&ActivityFilter {
            limit: 1,
            ..ActivityFilter::default()
        });
        assert!(page.active.is_empty());
        assert!(
            page.history[0]
                .duration_ms
                .is_some_and(|duration| duration >= 50)
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fixed_length_response_completes_successfully_over_hyper() {
        let hub = ActivityHub::new(ActivityPolicy {
            enabled: true,
            history_capacity: 1,
        });
        let app = Router::new()
            .route("/v1/audio/speech", post(|| async { Body::from("audio") }))
            .layer(middleware::from_fn_with_state(hub.clone(), track_activity));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let mut events = hub.subscribe();
        let server = tokio::spawn(async move {
            axum::serve(
                listener,
                app.into_make_service_with_connect_info::<SocketAddr>(),
            )
            .await
            .unwrap();
        });

        let response = tokio::task::spawn_blocking(move || {
            let mut stream = TcpStream::connect(address).unwrap();
            stream
                .write_all(
                    b"POST /v1/audio/speech HTTP/1.1\r\nHost: localhost\r\nUser-Agent: orchion-test-agent/1.0\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                )
                .unwrap();
            let mut response = String::new();
            stream.read_to_string(&mut response).unwrap();
            response
        })
        .await
        .unwrap();
        assert!(response.starts_with("HTTP/1.1 200 OK"));
        let started = events.recv().await.unwrap().payload.entry.unwrap();
        assert_eq!(started.address.as_deref(), Some("127.0.0.1"));
        assert_eq!(
            started.user_agent.as_deref(),
            Some("orchion-test-agent/1.0")
        );

        let page = tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                let page = hub.page(&ActivityFilter {
                    limit: 1,
                    ..ActivityFilter::default()
                });
                if !page.history.is_empty() {
                    break page;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(page.history[0].http_status, Some(200));
        assert_eq!(page.history[0].outcome, Some(ActivityOutcome::Success));

        server.abort();
        let _ = server.await;
    }

    #[tokio::test]
    async fn trailers_complete_http_activity() {
        let hub = ActivityHub::new(ActivityPolicy {
            enabled: true,
            history_capacity: 1,
        });
        let context = hub
            .start(
                ActivityOperation::Tts,
                ActivityTransport::Http,
                "POST",
                "/v1/audio/speech",
                None,
            )
            .unwrap();
        let mut trailers = HeaderMap::new();
        trailers.insert("x-stream-finished", HeaderValue::from_static("true"));
        let inner = Body::new(StreamBody::new(stream::iter([Ok::<_, Infallible>(
            Frame::trailers(trailers),
        )])));
        let mut body = ActivityBody {
            remaining_bytes: inner.size_hint().exact(),
            inner,
            completion: Some(HttpResponseCompletion {
                context,
                status: 200,
                activity_error: None,
                completed: false,
            }),
        };

        let frame = body.frame().await.unwrap().unwrap();
        assert!(frame.is_trailers());
        drop(body);

        let page = hub.page(&ActivityFilter {
            limit: 1,
            ..ActivityFilter::default()
        });
        assert_eq!(page.history[0].http_status, Some(200));
        assert_eq!(page.history[0].outcome, Some(ActivityOutcome::Success));
    }

    #[tokio::test]
    async fn dropping_response_before_eof_records_disconnection() {
        let hub = ActivityHub::new(ActivityPolicy {
            enabled: true,
            history_capacity: 1,
        });
        let app = Router::new()
            .route(
                "/v1/audio/speech",
                post(|| async {
                    Body::from_stream(async_stream::stream! {
                        tokio::time::sleep(Duration::from_secs(1)).await;
                        yield Ok::<_, Infallible>(Bytes::from_static(b"audio"));
                    })
                }),
            )
            .layer(middleware::from_fn_with_state(hub.clone(), track_activity));

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/audio/speech")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        drop(response);

        let page = hub.page(&ActivityFilter {
            limit: 1,
            ..ActivityFilter::default()
        });
        assert_eq!(page.history[0].outcome, Some(ActivityOutcome::Disconnected));
        assert_eq!(page.history[0].http_status, None);
    }
}
