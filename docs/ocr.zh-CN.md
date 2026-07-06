# OCR API

[English](ocr.md)

Orchion 通过 `POST /v1/ocr` 和 `multipart/form-data` 提供传统 OCR 与 OCR-VL 能力。

## 传统 OCR

```sh
curl -X POST http://127.0.0.1:9090/v1/ocr \
  -F file=@document.png \
  -F model=PaddlePaddle/PP-OCRv6_tiny \
  -F response_format=json
```

传统 OCR 返回结构化文本区域和纯文本。

## OCR-VL

```sh
curl -X POST http://127.0.0.1:9090/v1/ocr \
  -F file=@document.png \
  -F model=PaddlePaddle/PaddleOCR-VL-1.6 \
  -F response_format=markdown
```

在所选模型支持时，OCR-VL 支持 `ocr`、`table`、`formula`、`chart`、`spotting` 和 `seal` 等文档图像任务。

常用字段：

- `file`：图片或文档图片文件。
- `model`：可选模型 ID，格式为 `{vendor}/{name}`。
- `response_format`：`json`、`text`、`markdown` 或 `html`。
- `task`：可选 OCR-VL 任务。
- `layout_model`：可选 OCR-VL 版面模型。
- `max_tokens`：可选 OCR-VL 生成长度上限。
