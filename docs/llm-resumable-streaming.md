# Resumable LLM Streaming

Resumability is opt-in. Send `X-Orchion-Resumable: true` with a streaming `POST` to `/v1/chat/completions`, `/v1/completions`, or `/v1/responses`. The response includes `X-Orchion-Stream-ID` and `X-Orchion-Stream-TTL-Seconds`. Other header values, or use on a non-streaming request, return `invalid_resumable_stream`.

Every data frame has a decimal, 1-based SSE `id`. Chat and completion `[DONE]` sentinels also have IDs. Keepalive comments are `: keep-alive`, have no ID, and are not retained. Responses payload `sequence_number` remains independent from the transport ID.

Resume with `GET /v1/stream?stream_id=...&follow=true` and an optional decimal `Last-Event-ID`. Replay is strictly after that ID; `follow` defaults to true. Missing, expired, or another principal's streams are indistinguishable `404` responses. A cursor older than the retained prefix returns `409 replay_lost`.

For Responses streams, omitting `Last-Event-ID` or sending `0` is a full replay: payload sequence
validation starts at zero and requires the `response.created` then `response.in_progress` prefix.
With a positive cursor, the SDK cannot validate phases before the cursor; it infers a phase from the
first replayed event and validates all later ordering, including duplicate lifecycle and terminal
rules. Orchion currently emits `response.completed`, `response.incomplete`, or `error`; the SDK also
accepts terminal `response.failed` and `response.cancelled` events for future and remote compatibility.

`POST /v1/streams/lookup` accepts `{ "stream_ids": [...] }` and returns only requested streams visible to the authenticated principal. There is no list-all operation. `DELETE /v1/stream?stream_id=...` is idempotent and returns `204` for unknown or invisible IDs while cancelling an owned active generation.

Dropping a resumable follower does not cancel generation. Dropping a normal streaming response retains the existing cancellation behavior. Sessions and replay data are process-local, bounded, and unavailable after server restart.

`streaming.max_events_per_session` must be at least `1`, and both byte limits must be at least `512`; `max_total_bytes` must also be greater than or equal to `max_bytes_per_session`. These minima reserve enough retained budget for one bounded terminal protocol error frame in chat, completions, and Responses streams, so a retention failure is observable instead of appearing as a silent EOF.
