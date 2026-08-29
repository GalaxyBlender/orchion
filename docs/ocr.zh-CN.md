# OCR API

[English](ocr.md)

Orchion 通过 `POST /v1/ocr` 和 `multipart/form-data` 提供传统 OCR 与 OCR-VL 能力。

服务端的 `default_model` 从 `[[services.ocr.models]]` 或 `[[services.ocr-vl.models]]` deployment 中选择启动预置项；每个 deployment 可配置自己的 `layout_model` artifact。本阶段请求仍须显式传入 `model`，结构化响应仍须传入现有的 `layout_model` runtime ID。

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
  -F layout_model=PaddlePaddle/PP-DocLayoutV3 \
  -F response_format=markdown
```

在所选模型支持时，OCR-VL 支持 `ocr`、`table`、`formula`、`chart`、`spotting` 和 `seal` 等文档图像任务。

常用字段：

- `file`：图片或文档图片文件。
- `model`：必填模型 ID，格式为 `{vendor}/{name}`。
- `response_format`：`json`、`text`、`markdown` 或 `html`。
- `task`：可选 OCR-VL 任务。
- `layout_model`：版面模型；`markdown` 和 `html` 响应必须显式传入。
- `max_tokens`：可选 OCR-VL 生成长度上限。
