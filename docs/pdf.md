# PDF API

[简体中文](pdf.zh-CN.md)

Orchion exposes PDF page rendering through `POST /v1/pdf/images`.

Server operators can bound rendering with `[server].max_pdf_pages`, `max_pdf_pixels`, and `max_pdf_output_size`. Requests exceeding a configured limit return `pdf_limit_exceeded`.

The endpoint accepts a PDF upload and returns a ZIP archive containing rendered page images.

```sh
curl -X POST http://127.0.0.1:9090/v1/pdf/images \
  -F file=@document.pdf \
  -F response_format=png \
  -F pages=1,3-5 \
  -F scale=2 \
  --output pdf-images.zip
```

Fields:

- `file`: required PDF file.
- `response_format`: optional image format, one of `png`, `jpeg`, or `webp`.
- `pages`: optional page selection such as `1`, `1,3-5`, or `all`.
- `scale`: optional render scale from `0.1` through `4.0`.

Response headers include `x-pdf-page-count` and `x-pdf-image-count`.
