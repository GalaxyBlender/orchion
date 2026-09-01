# SDK Contract Matrix

This matrix tracks Orchion's supported in-process and server interfaces. It intentionally excludes
the WebUI, Swagger UI, and OpenAPI document routes. A dash means that the capability belongs only
to the other SDK; it is not an implied future interface.

| Capability | In-process `orchion` | Server route | `orchion-client` | Example | Contract verification |
| --- | --- | --- | --- | --- | --- |
| Health | - | `GET /healthz` | `health().check()` | `client-discovery` | server/client router contract |
| Readiness | - | `GET /readyz` | `health().ready()` | - | typed ready/not-ready response + server lifecycle tests |
| Metrics | - | authenticated `GET /metrics` OpenMetrics 1.0 | - | - | auth, exposition, bounded-label, monotonicity, and lifecycle tests |
| Model catalog | registered descriptors | `GET /v1/models` | `models().list()` | `client-discovery` | server/client router contract |
| Model retrieval | registered descriptors | `GET /v1/models/{model}` | `models().retrieve()` | - | public catalog lookup + uniform not-found route tests |
| Model residency | runtime ownership by loaded handles | `GET /api/models/status` | `models().list_statuses()` | `client-discovery`, `client-operations` | shared protocol DTO + router contract |
| Model load/unload | typed `load*` and handle drop/shutdown | `POST /api/models/load`, `POST /api/models/unload` | `models().load()`, `models().unload()` | `client-operations` | shared protocol DTO + server route tests |
| ASR file | `Asr::transcribe*` | `POST /v1/audio/transcriptions` | `asr().transcribe()` | `asr-file`, `client-asr-file` | client HTTP fixtures + server route tests |
| ASR streaming | `AsrStream` | `GET /v1/audio/transcriptions/stream` WebSocket | `asr().start_streaming()` | `asr-streaming`, `client-asr-streaming` | Ready, terminal, server error, EOF, and timeout tests |
| TTS | `Tts::synthesize*` with preset, clone, and design voices | `POST /v1/audio/speech` | `tts().create_speech()`, `tts().create_voice_clone()` | `tts-preset`, `tts-voice-modes`, `client-tts` | client HTTP fixtures + server route tests |
| OCR/OCR-VL | `Ocr::recognize*`; typed `OcrDeployment` provisioning | `POST /v1/ocr` | `ocr().recognize()` | `ocr-basic`, `client-ocr` | backend task/format matrix + route tests |
| Chat completions | `LlmEngine::stream_advanced()` semantic choices | `POST /v1/chat/completions` JSON/SSE | typed multiple choices, tools, reasoning, formats, logprobs, and bias | `llm-complete`, `llm-streaming-cancel`, `client-llm` | indexed/interleaved semantic fixtures, aggregate terminal, `[DONE]`, errors, cancellation |
| LLM vision | worker-local safe mtmd projector and owned PNG/JPEG parts | Chat `image_url`; Responses `input_image`; strict data URLs only | typed Chat and Responses image parts | - | parser/limits/order/capability tests plus ignored real-model canary |
| Legacy completions | advanced raw-prompt choices | `POST /v1/completions` JSON/SSE | typed `n`, logprobs, and logit bias | - | indexed raw choices, aggregate usage, and `[DONE]` tests |
| Responses | semantic choices adapted to dynamic stateless items | `POST /v1/responses` JSON/lifecycle SSE | typed tools/formats/reasoning plus forward-compatible dynamic and terminal events | `client-llm` | text/reasoning/function-call items, monotonic sequence, terminal/error/EOF fixtures |
| Responses input tokens | generation model tokenizer and Responses template | `POST /v1/responses/input_tokens` | `llm().count_response_input_tokens()` | - | template-path worker command + route/client fixtures |
| Embeddings | `LlmEngine::embed()` with text or token inputs | `POST /v1/embeddings` | `llm().create_embeddings()` | - | normalization, dimensions, encoding, validation, capability, and lifecycle tests |
| Resumable LLM streams | - | `X-Orchion-Resumable`, `/v1/stream`, `/v1/streams/lookup` | explicit resumable start/resume/lookup/delete methods | `llm_resumable_streaming` | replay, follower, cursor, capacity, auth, and fixture tests |
| Chat reasoning control | native reasoning-budget sampler | `/v1/chat/completions/control` | typed request builder, control method/result, completion ID on stream handle | `llm_reasoning_control` | native force/duplicate/unavailable, HTTP, SDK, OpenAPI, and metrics fixtures |
| PDF rendering | - | `POST /v1/pdf/images` | `pdf().render_images()` | `client-pdf` | response bytes and metadata-header tests |
| Activity snapshot/events | - | `GET /api/activity`, `GET /api/activity/events` SSE | `activity().list()`, `activity().subscribe()` | `client-operations` | snapshot/event/reset/EOF fixtures + server tests |

## Contract Ownership

- `orchion-core` owns local model identities, capabilities, options, results, and common errors.
- `orchion-protocol` owns stable Orchion-specific wire DTOs shared by server and client, currently ASR
  WebSocket, model lifecycle, Activity, and error objects.
- `orchion-client` owns caller ergonomics, transport behavior, stream state machines, and the
  still-evolving OpenAI-compatible LLM wire adapters.
- `orchion-server` owns routing, deployment residency, request policy, and inference dispatch; the
  client never depends on this crate.

## Lifecycle Rules

- ASR sessions are writable only after `Ready`; `Final`, `Completed`, and server `Error` are terminal.
  EOF before a terminal event is an error.
- Chat SSE completes only at `[DONE]`. Orchion's Responses server emits `response.completed` or
  `response.incomplete` as successful terminals and `error` on failure. The client also treats
  typed `response.failed` and `response.cancelled` events from future or compatible remote servers
  as terminal. Premature EOF before any terminal is an error.
- Activity SSE has no protocol terminal event. Clean EOF ends the subscription normally; callers
  choose whether to reconnect and should treat `Reset` as a signal to refresh the snapshot.
- Inference POST requests are not retried automatically.

`orchion` is currently a workspace-only facade (`publish = false`) because its native backend
adapters are not all registry-publishable. `orchion-client` is the publishable server SDK. Features
remain domain-specific and every client feature is checked independently in CI.
