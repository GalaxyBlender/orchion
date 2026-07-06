# PDF API

[English](pdf.md)

Orchion 通过 `POST /v1/pdf/images` 提供 PDF 页面渲染能力。

该接口接收 PDF 上传，并返回包含渲染后页面图片的 ZIP 文件。

```sh
curl -X POST http://127.0.0.1:9090/v1/pdf/images \
  -F file=@document.pdf \
  -F response_format=png \
  -F pages=1,3-5 \
  -F scale=2 \
  --output pdf-images.zip
```

字段：

- `file`：必填 PDF 文件。
- `response_format`：可选图片格式，支持 `png`、`jpeg` 或 `webp`。
- `pages`：可选页面范围，例如 `1`、`1,3-5` 或 `all`。
- `scale`：可选渲染缩放，范围为 `0.1` 到 `4.0`。

响应头包含 `x-pdf-page-count` 和 `x-pdf-image-count`。
