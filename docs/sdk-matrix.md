# SDK Contract Matrix

This matrix tracks Orchion's supported in-process and server interfaces. It intentionally excludes
the WebUI, Swagger UI, and OpenAPI document routes. A dash means that the capability belongs only
to the other SDK; it is not an implied future interface.

| Capability | In-process `orchion` | Server route | `orchion-client` | Example | Contract verification |
| --- | --- | --- | --- | --- | --- |
| Health | - | `GET /healthz` | `health().check()` | `client-discovery` | server/client router contract |
| Model catalog | registered descriptors | `GET /v1/models` | `models().list()` | `client-discovery` | server/client router contract |
| Model residency | runtime ownership by loaded handles | `GET /api/models/status` | `models().list_statuses()` | `client-discovery`, `client-operations` | shared protocol DTO + router contract |
| Model load/unload | typed `load*` and handle drop/shutdown | `POST /api/models/load`, `POST /api/models/unload` | `models().load()`, `models().unload()` | `client-operations` | shared protocol DTO + server route tests |
| ASR file | `Asr::transcribe*` | `POST /v1/audio/transcriptions` | `asr().transcribe()` | `asr-file`, `client-asr-file` | client HTTP fixtures + server route tests |
| ASR streaming | `AsrStream` | `GET /v1/audio/transcriptions/stream` WebSocket | `asr().start_streaming()` | `asr-streaming`, `client-asr-streaming` | Ready, terminal, server error, EOF, and timeout tests |
| TTS | `Tts::synthesize*` with preset, clone, and design voices | `POST /v1/audio/speech` | `tts().create_speech()`, `tts().create_voice_clone()` | `tts-preset`, `tts-voice-modes`, `client-tts` | client HTTP fixtures + server route tests |
| OCR/OCR-VL | `Ocr::recognize*`; typed `OcrDeployment` provisioning | `POST /v1/ocr` | `ocr().recognize()` | `ocr-basic`, `client-ocr` | backend task/format matrix + route tests |
| Chat completions | `LlmEngine::complete()` and token streaming | `POST /v1/chat/completions` JSON/SSE | `llm().create_chat_completion()`, `llm().stream_chat_completion()` | `llm-complete`, `llm-streaming-cancel`, `client-llm` | JSON, `[DONE]`, in-band error, EOF, and timeout fixtures |
| Responses | same text-generation engine; no separate native wire model | `POST /v1/responses` JSON/lifecycle SSE | `llm().create_response()`, `llm().stream_response()` | `client-llm` | lifecycle order, terminal, in-band error, EOF, and timeout fixtures |
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
- Chat SSE completes only at `[DONE]`. Responses SSE completes only at `response.completed` or
  `response.incomplete`. Premature EOF is an error.
- Activity SSE has no protocol terminal event. Clean EOF ends the subscription normally; callers
  choose whether to reconnect and should treat `Reset` as a signal to refresh the snapshot.
- Inference POST requests are not retried automatically.

`orchion` is currently a workspace-only facade (`publish = false`) because its native backend
adapters are not all registry-publishable. `orchion-client` is the publishable server SDK. Features
remain domain-specific and every client feature is checked independently in CI.
