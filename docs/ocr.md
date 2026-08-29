# OCR API

[简体中文](ocr.zh-CN.md)

Orchion exposes traditional OCR and OCR-VL through `POST /v1/ocr` with `multipart/form-data`.

Server `default_model` selects one `[[services.ocr.models]]` or `[[services.ocr-vl.models]]` deployment for startup provisioning. Each deployment configures its optional `layout_model` artifact. In this phase, requests must still provide `model`, and structured responses must still provide the existing `layout_model` runtime ID.

## Traditional OCR

```sh
curl -X POST http://127.0.0.1:9090/v1/ocr \
  -F file=@document.png \
  -F model=PaddlePaddle/PP-OCRv6_tiny \
  -F response_format=json
```

Traditional OCR returns structured text regions and plain text.

## OCR-VL

```sh
curl -X POST http://127.0.0.1:9090/v1/ocr \
  -F file=@document.png \
  -F model=PaddlePaddle/PaddleOCR-VL-1.6 \
  -F layout_model=PaddlePaddle/PP-DocLayoutV3 \
  -F response_format=markdown
```

OCR-VL supports document-image tasks such as `ocr`, `table`, `formula`, `chart`, `spotting`, and `seal` when supported by the selected model.

Useful fields:

- `file`: image or document image file.
- `model`: required model ID in `{vendor}/{name}` format.
- `response_format`: `json`, `text`, `markdown`, or `html`.
- `task`: optional OCR-VL task.
- `layout_model`: layout model; required for `markdown` and `html` responses.
- `max_tokens`: optional OCR-VL generation limit.
