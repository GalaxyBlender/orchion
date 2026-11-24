# OCR API

[简体中文](ocr.zh-CN.md)

Orchion exposes traditional OCR and OCR-VL through `POST /v1/ocr` with `multipart/form-data`.

Startup provisions every `[[services.ocr.models]]` and `[[services.ocr-vl.models]]` deployment in enabled services, including their configured dependencies, regardless of `default_model`. Each deployment may configure its own `layout_model` artifact. Requests select only the primary deployment with `model`; the server loads and uses that deployment's layout asset automatically.

## Traditional OCR

```sh
curl -X POST http://127.0.0.1:9090/v1/ocr \
  -F file=@document.png \
  -F model=paddlepaddle/pp-ocrv6-tiny \
  -F response_format=json
```

Traditional OCR always supports JSON and plain text. A deployment with a configured layout asset also advertises and accepts Markdown; HTML remains an OCR-VL capability. A traditional deployment without layout rejects Markdown.

### Response formats

| `response_format` | HTTP content type | Body |
| --- | --- | --- |
| `json` | `application/json` | Recognition metadata, plain `text` and `regions[].text`; structured layout/table results also include `markdown` and sanitized `html`. |
| `text` | `text/plain; charset=utf-8` | Readable text only. Table rows use newlines, cells use tabs, and empty cells are retained. |
| `markdown` | `text/markdown; charset=utf-8` | Markdown document with pipe tables, including flattened merged cells. Does not fall back to HTML tables. |
| `html` | `text/html; charset=utf-8` | Sanitized HTML with table structure; supported OCR-VL tasks only. |

Text output emits each merged cell's content once. Markdown expands merged cells
into a rectangular grid: content stays in the top-left cell, covered positions
are empty, and short rows are padded. Use HTML or JSON's `html` field when
row/column spans must be preserved. Cell-internal line breaks are folded
to spaces in text tables, so one table row remains one output line. The plain-text
fields preserve literal recognized text outside tables, including `<` and `>`.

### Table screenshots

Traditional OCR with layout needs a table structure model when the page contains
a table. The shipped configuration explicitly selects SLANet+ and its dictionary
for all traditional deployments alongside `layout_model` in each
`[[services.ocr.models]]` entry:

```toml
table_structure = { model = "//greatv/oar-ocr/slanet_plus.onnx", dictionary = "//greatv/oar-ocr/table_structure_dict_ch.txt", table_type = "wireless" }
```

Model selection is deployment configuration; the runtime does not infer a table
model from `layout_model`. These neutral `//` locators follow `models.source`
instead of pinning the artifacts to a provider.

At INFO level, table deployment preparation reports cache checks/reuse, source
selection, repository download start/completion (including files and elapsed
time), and the final cache-ready state. Changing artifact locators or provider
policy changes the deployment cache identity and can require preparation again;
an unchanged, valid publication is reused without provider requests.

Both artifacts are prepared at startup and shared through the model cache.
`table_type` selects the structure adapter, not the screenshot's detected type;
this SLANet+ profile can also process bordered spreadsheet screenshots. JSON
`text` and block-level `regions` contain readable table text in reading order;
table markup is kept in `markdown` and `html`. Markdown always uses pipe tables;
HTML preserves `rowspan` and `colspan`. Without layout, traditional OCR
still extracts text lines but does not reconstruct the grid.

## OCR-VL

```sh
curl -X POST http://127.0.0.1:9090/v1/ocr \
  -F file=@document.png \
  -F model=paddlepaddle/paddleocr-vl-1.6 \
  -F response_format=markdown
```

OCR-VL supports document-image tasks such as `ocr`, `table`, `formula`, `chart`, `spotting`, and `seal` when supported by the selected model.

Use `task=table` to parse a table crop directly. It accepts JSON, text, Markdown,
and HTML without requiring a layout model; full-page structured `task=ocr`
still requires layout. Other tasks accept JSON and text only. HTML table output
is sanitized; generated scripts, event handlers, and unrelated attributes are
not passed through.

Useful fields:

- `file`: image or document image file.
- `model`: required model ID in `{vendor}/{name}` format.
- `response_format`: `json`, `text`, `markdown`, or `html`, when advertised by the selected deployment's `/v1/models` capabilities.
- `task`: optional OCR-VL task.
- `max_tokens`: optional OCR-VL generation limit.

`layout_model` is deployment configuration, not a request parameter. Requests that send a multipart `layout_model` field are rejected with `unsupported_ocr_parameter`.

## CUDA memory policy

CUDA ONNX sessions use a shared runtime policy: memory patterns are disabled and
idle arena regions are eligible for release after each inference. This covers
traditional OCR, document structure models, and OCR-VL layout detection. CPU and
WebGPU policies are unchanged; the Candle OCR-VL model manages its own caches.

See [real-model memory regression tests](ocr-memory-testing.md) for repeated
requests, task/format switching, and mixed image sizes.

## JSON regions

JSON responses include `regions` with recognized text and polygon coordinates. Traditional OCR without layout returns text-line regions. The traditional OCR and OCR-VL layout pipelines return block-level regions for layout elements with non-blank text, preserving element order and converting generated table markup into readable text. Their region `confidence` is `null` because layout detection scores are not text recognition confidence scores.

`layout_blocks` still includes all detected layout elements, including those without text, with their labels, detection confidence scores, and polygon coordinates. OCR-VL without the layout pipeline currently returns an empty `regions` array. The `text`, `markdown`, and `html` response formats return only the document body, not JSON region metadata.

## Real-model table regression tests

`libs/orchion-ocr/tests/runtime_tables.rs` uses rectangular and merged table
screenshots and a text-only document (`tests/fixtures/table.png`,
`merged-table.png`, and `document.png`, generated from the adjacent SVGs).
Every image covers JSON, text, Markdown, and HTML: traditional
OCR must explicitly reject HTML, while OCR-VL accepts it. The tests check
recognition content, regions, Markdown columns without HTML fallback, and HTML
table presence. OCR-VL's `table` task is also exercised on the table image.
Run it with local models in an optimized build:

```sh
export ORCHION_OCR_TEST_MODELS="$PWD/data/models"
export ORCHION_OCR_TEST_TABLE_MODEL=/absolute/path/slanet_plus.onnx
export ORCHION_OCR_TEST_TABLE_DICTIONARY=/absolute/path/table_structure_dict_ch.txt
cargo test --release -p orchion-ocr --features ocr,ocr-vl \
  --test runtime_tables -- --ignored --nocapture --test-threads=1
```

For CUDA, add the `cuda` feature and set `ORCHION_OCR_TEST_DEVICE=cuda`;
[stage the ONNX providers beside the test binary](ocr-memory-testing.md) when
using a shared Cargo build directory.

Set `ORCHION_OCR_TEST_IMAGE` to an additional table screenshot to exercise the
traditional OCR JSON, text, and Markdown paths alongside the checked-in fixture.

### Live HTTP format matrix

`sdks/orchion-client/tests/runtime_ocr_formats.rs` discovers every configured OCR
model from `/v1/models` and checks all four formats with all three fixtures. OCR-VL
also covers `task=table` on table images. It asserts status codes, exact
Content-Type, recognition content, JSON fields, Markdown syntax, and HTML
structure. Unsupported formats must return a structured 400 error; HTTP 200
alone does not count as success. The merged-table fixture catches HTML fallback
in Markdown even without extra input. Set `ORCHION_OCR_TEST_IMAGE` to include
an additional real-world table screenshot.

```sh
ORCHION_OCR_HTTP_BASE_URL=http://127.0.0.1:9090 \
ORCHION_OCR_TEST_IMAGE=/absolute/path/merged-table.jpg \
cargo test -p orchion-client --no-default-features --features ocr \
  --test runtime_ocr_formats -- --ignored --nocapture
```

This test makes inference requests to an already running server. It does not
start, stop, reconfigure, or deploy the service. The server's HTTP serialization
boundary also has an opt-in real-model test, using the same model environment:

```sh
cargo test -p orchion-server --lib response_tests -- --include-ignored --nocapture
```
