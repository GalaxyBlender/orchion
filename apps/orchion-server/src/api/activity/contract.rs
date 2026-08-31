pub use orchion_protocol::{
    ActivityEntry, ActivityEventPayload, ActivityOperation, ActivityOutcome, ActivityPage,
    ActivityState, ActivitySummary, ActivityTransport,
};

#[derive(Debug, Clone)]
pub(crate) struct ActivityError {
    pub(crate) code: Option<String>,
    pub(crate) message: Option<String>,
}
