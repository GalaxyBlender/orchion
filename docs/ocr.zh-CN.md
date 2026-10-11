# OCR API

[English](ocr.md)

Orchion 通过 `POST /v1/ocr` 和 `multipart/form-data` 提供传统 OCR 与 OCR-VL 能力。

服务端的 `default_model` 从 `[[services.ocr.models]]` 或 `[[services.ocr-vl.models]]` deployment 中选择启动预置项；每个 deployment 可配置自己的 `layout_model` artifact。请求只通过 `model` 选择 primary deployment，服务端会自动加载并使用该 deployment 的 layout asset。

## 传统 OCR

```sh
curl -X POST http://127.0.0.1:9090/v1/ocr \
  -F file=@document.png \
  -F model=paddlepaddle/pp-ocrv6-tiny \
  -F response_format=json
```

传统 OCR 始终支持 JSON 和纯文本。配置了 layout asset 的 deployment 还会发布并接受 Markdown；HTML 仍仅属于 OCR-VL 能力。传统 OCR 未配置 layout 时会拒绝 Markdown。

## OCR-VL

```sh
curl -X POST http://127.0.0.1:9090/v1/ocr \
  -F file=@document.png \
  -F model=paddlepaddle/paddleocr-vl-1.6 \
  -F response_format=markdown
```

在所选模型支持时，OCR-VL 支持 `ocr`、`table`、`formula`、`chart`、`spotting` 和 `seal` 等文档图像任务。

常用字段：

- `file`：图片或文档图片文件。
- `model`：必填模型 ID，格式为 `{vendor}/{name}`。
- `response_format`：`json`、`text`、`markdown` 或 `html`，具体以所选 deployment 在 `/v1/models` 中发布的能力为准。
- `task`：可选 OCR-VL 任务。
- `max_tokens`：可选 OCR-VL 生成长度上限。

`layout_model` 属于 deployment 配置，不是请求参数。multipart 请求若仍发送 `layout_model` 字段，服务端会以 `unsupported_ocr_parameter` 拒绝。

## JSON 文字区域

JSON 响应包含 `regions`，提供识别文字和多边形坐标。未使用 layout 的传统 OCR 返回行级文字区域；传统 OCR 和 OCR-VL 的 layout 流水线则为有非空白文字的版面元素返回块级区域，保留原始文字和元素顺序。这些区域的 `confidence` 为 `null`，因为版面检测得分不是文字识别置信度。

`layout_blocks` 仍包含所有检测到的版面元素，包括没有文字的元素，并提供类别、检测置信度和多边形坐标。未使用 layout 流水线的 OCR-VL 当前仍返回空的 `regions` 数组。`text`、`markdown` 和 `html` 响应格式仅返回文档正文，不包含 JSON 区域元数据。
