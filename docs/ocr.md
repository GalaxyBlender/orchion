# OCR API

[简体中文](ocr.zh-CN.md)

Orchion exposes traditional OCR and OCR-VL through `POST /v1/ocr` with `multipart/form-data`.

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
  -F response_format=markdown
```

OCR-VL supports document-image tasks such as `ocr`, `table`, `formula`, `chart`, `spotting`, and `seal` when supported by the selected model.

Useful fields:

- `file`: image or document image file.
- `model`: optional model ID in `{vendor}/{name}` format.
- `response_format`: `json`, `text`, `markdown`, or `html`.
- `task`: optional OCR-VL task.
- `layout_model`: optional OCR-VL layout model.
- `max_tokens`: optional OCR-VL generation limit.
