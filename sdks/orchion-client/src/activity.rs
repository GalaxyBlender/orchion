use crate::client::decode_json;
use crate::sse::SseStream;
use crate::{Client, ClientError};
use serde::Serialize;

pub use orchion_protocol::{
    ActivityEntry, ActivityEventPayload, ActivityOperation, ActivityOutcome, ActivityPage,
    ActivityState, ActivitySummary, ActivityTransport,
};

/// Filters applied when listing retained activity.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
#[non_exhaustive]
pub struct ActivityQuery {
    #[serde(skip_serializing_if = "Option::is_none")]
    limit: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    before: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    operation: Option<ActivityOperation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    outcome: Option<ActivityOutcome>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
}

impl ActivityQuery {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            limit: None,
            before: None,
            operation: None,
            outcome: None,
            model: None,
        }
    }

    /// Sets the page size.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::BuildRequest`] unless `limit` is between 1 and 200 inclusive.
    pub fn with_limit(mut self, limit: usize) -> Result<Self, ClientError> {
        if !(1..=200).contains(&limit) {
            return Err(ClientError::build_request(
                "activity limit must be between 1 and 200",
            ));
        }
        self.limit = Some(limit);
        Ok(self)
    }

    #[must_use]
    pub const fn with_before(mut self, before: u64) -> Self {
        self.before = Some(before);
        self
    }

    #[must_use]
    pub const fn with_operation(mut self, operation: ActivityOperation) -> Self {
        self.operation = Some(operation);
        self
    }

    #[must_use]
    pub const fn with_outcome(mut self, outcome: ActivityOutcome) -> Self {
        self.outcome = Some(outcome);
        self
    }

    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }
}

/// Client for activity history and live events.
pub struct ActivityClient<'a> {
    client: &'a Client,
}

impl<'a> ActivityClient<'a> {
    #[must_use]
    pub(crate) const fn new(client: &'a Client) -> Self {
        Self { client }
    }

    /// Lists activity matching the supplied query.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when the request cannot be sent or the response cannot be decoded.
    pub async fn list(&self, query: ActivityQuery) -> Result<ActivityPage, ClientError> {
        let response = self
            .client
            .get("/api/activity")?
            .query(&query)
            .send()
            .await?;
        decode_json(response).await
    }

    /// Subscribes to live activity events without automatic reconnection.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError`] when the request fails or the response is not an SSE stream.
    pub async fn subscribe(&self) -> Result<ActivityStream, ClientError> {
        let response = self
            .client
            .stream_get("/api/activity/events")?
            .send()
            .await?;
        Ok(ActivityStream {
            stream: SseStream::from_response(response).await?,
        })
    }
}

/// A live stream of semantically validated activity events.
pub struct ActivityStream {
    stream: SseStream,
}

impl ActivityStream {
    /// Returns the next event, or `None` when the server naturally closes the stream.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::Transport`] for body I/O failures and [`ClientError::Decode`] for
    /// malformed SSE, JSON, unknown event names, or invalid activity payload shapes.
    pub async fn next_event(&mut self) -> Result<Option<ActivityEvent>, ClientError> {
        let Some(event) = self.stream.next_event().await? else {
            return Ok(None);
        };
        decode_event(&event)
    }
}

/// A semantically validated activity stream event.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ActivityEvent {
    Snapshot {
        cursor: String,
        active: Vec<ActivityEntry>,
        summary: ActivitySummary,
    },
    Started {
        cursor: String,
        entry: ActivityEntry,
        summary: ActivitySummary,
    },
    Updated {
        cursor: String,
        entry: ActivityEntry,
        summary: ActivitySummary,
    },
    Completed {
        cursor: String,
        entry: ActivityEntry,
        summary: ActivitySummary,
    },
    Reset {
        cursor: String,
        summary: ActivitySummary,
    },
}

fn decode_event(event: &eventsource_stream::Event) -> Result<Option<ActivityEvent>, ClientError> {
    let payload = serde_json::from_str::<ActivityEventPayload>(&event.data)
        .map_err(|error| ClientError::decode(format!("invalid activity event JSON: {error}")))?;
    if event.id.is_empty() || event.id != payload.cursor {
        return Err(ClientError::decode(
            "activity event id must be non-empty and match the payload cursor",
        ));
    }
    validate_absent_fields(&event.event, &payload)?;
    let cursor = payload.cursor.clone();

    let decoded = match event.event.as_str() {
        "snapshot" => ActivityEvent::Snapshot {
            cursor,
            active: require_active(&payload)?,
            summary: require_summary(&payload)?,
        },
        "started" => {
            let entry = require_entry(&payload)?;
            if entry.state != ActivityState::InFlight {
                return Err(ClientError::decode(
                    "started activity event entry must be in flight",
                ));
            }
            ActivityEvent::Started {
                cursor,
                entry,
                summary: require_summary(&payload)?,
            }
        }
        "updated" => ActivityEvent::Updated {
            cursor,
            entry: require_in_flight_entry(&payload, "updated")?,
            summary: require_summary(&payload)?,
        },
        "completed" => {
            let entry = require_entry(&payload)?;
            if entry.state != ActivityState::Completed {
                return Err(ClientError::decode(
                    "completed activity event entry must be completed",
                ));
            }
            ActivityEvent::Completed {
                cursor,
                entry,
                summary: require_summary(&payload)?,
            }
        }
        "reset" => ActivityEvent::Reset {
            cursor,
            summary: require_summary(&payload)?,
        },
        name => {
            return Err(ClientError::decode(format!(
                "unknown activity event name `{name}`"
            )));
        }
    };
    Ok(Some(decoded))
}

fn require_entry(payload: &ActivityEventPayload) -> Result<ActivityEntry, ClientError> {
    payload
        .entry
        .clone()
        .ok_or_else(|| ClientError::decode("activity entry event is missing `entry`"))
}

fn require_active(payload: &ActivityEventPayload) -> Result<Vec<ActivityEntry>, ClientError> {
    let active = payload
        .active
        .clone()
        .ok_or_else(|| ClientError::decode("activity snapshot event is missing `active`"))?;
    if active
        .iter()
        .any(|entry| entry.state != ActivityState::InFlight)
    {
        return Err(ClientError::decode(
            "activity snapshot entries must be in flight",
        ));
    }
    Ok(active)
}

fn require_in_flight_entry(
    payload: &ActivityEventPayload,
    event: &str,
) -> Result<ActivityEntry, ClientError> {
    let entry = require_entry(payload)?;
    if entry.state != ActivityState::InFlight {
        return Err(ClientError::decode(format!(
            "{event} activity event entry must be in flight"
        )));
    }
    Ok(entry)
}

fn require_summary(payload: &ActivityEventPayload) -> Result<ActivitySummary, ClientError> {
    payload
        .summary
        .clone()
        .ok_or_else(|| ClientError::decode("activity event is missing `summary`"))
}

fn validate_absent_fields(name: &str, payload: &ActivityEventPayload) -> Result<(), ClientError> {
    let valid = match name {
        "snapshot" => payload.entry.is_none(),
        "started" | "updated" | "completed" => payload.active.is_none(),
        "reset" => payload.entry.is_none() && payload.active.is_none(),
        _ => true,
    };
    if valid {
        Ok(())
    } else {
        Err(ClientError::decode(format!(
            "activity `{name}` event contains fields forbidden by its payload shape"
        )))
    }
}
