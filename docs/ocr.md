# OCR API

[简体中文](ocr.zh-CN.md)

Orchion exposes traditional OCR and OCR-VL through `POST /v1/ocr` with `multipart/form-data`.

Server `default_model` selects one `[[services.ocr.models]]` or `[[services.ocr-vl.models]]` deployment for startup provisioning. Each deployment may configure its own `layout_model` artifact. Requests select only the primary deployment with `model`; the server loads and uses that deployment's layout asset automatically.

## Traditional OCR

```sh
curl -X POST http://127.0.0.1:9090/v1/ocr \
  -F file=@document.png \
  -F model=PaddlePaddle/PP-OCRv6_tiny \
  -F response_format=json
```

Traditional OCR always supports JSON and plain text. A deployment with a configured layout asset also advertises and accepts Markdown; HTML remains an OCR-VL capability. A traditional deployment without layout rejects Markdown.

## OCR-VL

```sh
curl -X POST http://127.0.0.1:9090/v1/ocr \
  -F file=@document.png \
  -F model=PaddlePaddle/PaddleOCR-VL-1.6 \
  -F response_format=markdown
```

OCR-VL supports document-image tasks such as `ocr`, `table`, `formula`, `chart`, `spotting`, and `seal` when supported by the selected model.

Useful fields:

- `file`: image or document image file.
- `model`: required model ID in `{vendor}/{name}` format.
- `response_format`: `json`, `text`, `markdown`, or `html`, when advertised by the selected deployment's `/v1/models` capabilities.
- `task`: optional OCR-VL task.
- `max_tokens`: optional OCR-VL generation limit.

`layout_model` is deployment configuration, not a request parameter. Requests that send a multipart `layout_model` field are rejected with `unsupported_ocr_parameter`.
