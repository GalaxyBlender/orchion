use crate::ClientError;
use crate::client::ensure_success;
use bytes::Bytes;
use eventsource_stream::{Event, EventStream, EventStreamError, Eventsource};
use futures_util::{Stream, StreamExt};
use reqwest::Response;
use reqwest::header::CONTENT_TYPE;
use std::pin::Pin;

type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>;

pub(crate) struct SseStream {
    events: EventStream<ByteStream>,
}

impl SseStream {
    pub(crate) async fn from_response(response: Response) -> Result<Self, ClientError> {
        let response = ensure_success(response).await?;
        validate_content_type(&response)?;
        let bytes: ByteStream = Box::pin(response.bytes_stream());
        Ok(Self {
            events: bytes.eventsource(),
        })
    }

    pub(crate) async fn next_event(&mut self) -> Result<Option<Event>, ClientError> {
        match self.events.next().await {
            Some(Ok(event)) => Ok(Some(event)),
            Some(Err(EventStreamError::Transport(source))) => {
                Err(ClientError::Transport { source })
            }
            Some(Err(error)) => Err(ClientError::decode(format!(
                "invalid server-sent event stream: {error}"
            ))),
            None => Ok(None),
        }
    }
}

fn validate_content_type(response: &Response) -> Result<(), ClientError> {
    let value = response
        .headers()
        .get(CONTENT_TYPE)
        .ok_or_else(|| ClientError::decode("missing SSE Content-Type header"))?
        .to_str()
        .map_err(|error| ClientError::decode(format!("invalid Content-Type header: {error}")))?;
    let media_type = value.split(';').next().unwrap_or_default().trim();
    if media_type.eq_ignore_ascii_case("text/event-stream") {
        Ok(())
    } else {
        Err(ClientError::decode(format!(
            "expected SSE Content-Type `text/event-stream`, received `{value}`"
        )))
    }
}
