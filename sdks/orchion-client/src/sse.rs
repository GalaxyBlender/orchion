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
    last_event_id: Option<u64>,
    expected_event_id: Option<u64>,
}

impl SseStream {
    pub(crate) async fn from_response(response: Response) -> Result<Self, ClientError> {
        let response = ensure_success(response).await?;
        validate_content_type(&response)?;
        let bytes: ByteStream = Box::pin(response.bytes_stream());
        Ok(Self {
            events: bytes.eventsource(),
            last_event_id: None,
            expected_event_id: None,
        })
    }

    pub(crate) async fn from_resumable_response(
        response: Response,
        last_event_id: Option<u64>,
    ) -> Result<Self, ClientError> {
        let mut stream = Self::from_response(response).await?;
        stream.last_event_id = last_event_id;
        stream.expected_event_id = Some(
            last_event_id
                .unwrap_or(0)
                .checked_add(1)
                .ok_or_else(|| ClientError::decode("resumable SSE event ID overflow"))?,
        );
        Ok(stream)
    }

    pub(crate) async fn next_event(&mut self) -> Result<Option<Event>, ClientError> {
        match self.events.next().await {
            Some(Ok(event)) => {
                if let Some(expected) = self.expected_event_id {
                    let received = event.id.parse::<u64>().map_err(|_| {
                        ClientError::decode("resumable SSE event is missing a decimal event ID")
                    })?;
                    if received == 0 || received != expected {
                        return Err(ClientError::decode(format!(
                            "expected resumable SSE event ID {expected}, received {received}"
                        )));
                    }
                    self.last_event_id = Some(received);
                    self.expected_event_id = received.checked_add(1);
                    if self.expected_event_id.is_none() {
                        return Err(ClientError::decode("resumable SSE event ID overflow"));
                    }
                } else if !event.id.is_empty()
                    && let Ok(received) = event.id.parse::<u64>()
                {
                    self.last_event_id = Some(received);
                }
                Ok(Some(event))
            }
            Some(Err(EventStreamError::Transport(source))) => {
                Err(ClientError::Transport { source })
            }
            Some(Err(error)) => Err(ClientError::decode(format!(
                "invalid server-sent event stream: {error}"
            ))),
            None => Ok(None),
        }
    }

    pub(crate) const fn last_event_id(&self) -> Option<u64> {
        self.last_event_id
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
