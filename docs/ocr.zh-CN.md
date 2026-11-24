# OCR API

[English](ocr.md)

Orchion 通过 `POST /v1/ocr` 和 `multipart/form-data` 提供传统 OCR 与 OCR-VL 能力。

启动时会准备所有已启用服务的 `[[services.ocr.models]]` 和 `[[services.ocr-vl.models]]` deployment 及其已配置的依赖，不受 `default_model` 限制；每个 deployment 可配置自己的 `layout_model` artifact。请求只通过 `model` 选择 primary deployment，服务端会自动加载并使用该 deployment 的 layout asset。

## 传统 OCR

```sh
curl -X POST http://127.0.0.1:9090/v1/ocr \
  -F file=@document.png \
  -F model=paddlepaddle/pp-ocrv6-tiny \
  -F response_format=json
```

传统 OCR 始终支持 JSON 和纯文本。配置了 layout asset 的 deployment 还会发布并接受 Markdown；HTML 仍仅属于 OCR-VL 能力。传统 OCR 未配置 layout 时会拒绝 Markdown。

### 响应格式

| `response_format` | HTTP Content-Type | 内容 |
| --- | --- | --- |
| `json` | `application/json` | 识别元数据，`text`、`regions[].text` 为纯文本；版面/表格结构化结果还包含 `markdown` 和经过安全清理的 `html`。 |
| `text` | `text/plain; charset=utf-8` | 仅返回可读文本：表格行以换行分隔，单元格以 Tab 分隔，保留空单元格。 |
| `markdown` | `text/markdown; charset=utf-8` | Markdown 文档，表格始终使用管道语法；合并单元格展平，不回退为 HTML 表格。 |
| `html` | `text/html; charset=utf-8` | 保留表格结构的安全 HTML，仅支持具备该能力的 OCR-VL 任务。 |

纯文本中合并单元格的内容只输出一次，单元格内部换行折叠为空格，确保一行表格对应一行文本。Markdown 将合并单元格展开为规则网格：内容仅放在左上角，被覆盖的位置留空，短行补齐空单元格。需要保留跨行、跨列结构时，使用 HTML 或 JSON 的 `html` 字段。纯文本字段中，表格之外识别出的字面文本（包括 `<`、`>`）不会被当作标签删除。

### 表格截图

传统 OCR 配合版面检测时，识别包含表格的页面还需要表格结构模型。随包 `config.toml` 已在各 `[[services.ocr.models]]` 的 `layout_model` 旁显式配置 SLANet+ 及其字典：

```toml
table_structure = { model = "//greatv/oar-ocr/slanet_plus.onnx", dictionary = "//greatv/oar-ocr/table_structure_dict_ch.txt", table_type = "wireless" }
```

这两个资源会在启动时准备，并通过模型缓存共享。`table_type` 选择结构识别适配器，并非截图检测到的表格类型；该 SLANet+ 配置也可识别带边框的电子表格截图。JSON 的 `text` 和块级 `regions` 按阅读顺序包含可读的表格文本，表格标记放在 `markdown`、`html` 字段。Markdown 始终输出管道表格，HTML 则保留 `rowspan`、`colspan`。未配置 layout 的传统 OCR 仍可提取文字行，但不会重建表格网格。

模型选择由 deployment 配置决定，运行时不会根据 `layout_model` 隐式选择表格模型。默认的 `//` 地址遵循 `models.source`，不固定下载来源。

INFO 日志会显示表格 deployment 的缓存检查/复用、来源选择、仓库下载开始/完成（包含文件列表和耗时），以及缓存就绪状态。修改资源地址或来源策略会改变 deployment 缓存标识，可能需要重新准备；配置不变且缓存有效时，会直接复用，不请求下载来源。

## OCR-VL

```sh
curl -X POST http://127.0.0.1:9090/v1/ocr \
  -F file=@document.png \
  -F model=paddlepaddle/paddleocr-vl-1.6 \
  -F response_format=markdown
```

在所选模型支持时，OCR-VL 支持 `ocr`、`table`、`formula`、`chart`、`spotting` 和 `seal` 等文档图像任务。

对单个表格截图可使用 `task=table`，支持 JSON、纯文本、Markdown 和 HTML，不需要版面模型。整页结构化 `task=ocr` 仍需要 layout；其他任务仅支持 JSON 和纯文本。HTML 表格输出会过滤模型生成的脚本、事件处理属性等不安全内容。

常用字段：

- `file`：图片或文档图片文件。
- `model`：必填模型 ID，格式为 `{vendor}/{name}`。
- `response_format`：`json`、`text`、`markdown` 或 `html`，具体以所选 deployment 在 `/v1/models` 中发布的能力为准。
- `task`：可选 OCR-VL 任务。
- `max_tokens`：可选 OCR-VL 生成长度上限。

`layout_model` 属于 deployment 配置，不是请求参数。multipart 请求若仍发送 `layout_model` 字段，服务端会以 `unsupported_ocr_parameter` 拒绝。

## CUDA 显存策略

所有 CUDA ONNX 会话采用统一的运行时策略：关闭 memory pattern，并在每次推理后收缩可释放的闲置内存池区域。该策略覆盖传统 OCR、文档结构模型及 OCR-VL 的版面检测。CPU 和 WebGPU 策略不变；Candle OCR-VL 主模型独立管理自身缓存。

重复请求、任务/格式切换和混合图片尺寸的验证方法见[真实模型显存回归测试](ocr-memory-testing.md)。

## JSON 文字区域

JSON 响应包含 `regions`，提供识别文字和多边形坐标。未使用 layout 的传统 OCR 返回行级文字区域；传统 OCR 和 OCR-VL 的 layout 流水线则为有非空白文字的版面元素返回块级区域，保留元素顺序，并将生成的表格标记转换为可读文本。这些区域的 `confidence` 为 `null`，因为版面检测得分不是文字识别置信度。

`layout_blocks` 仍包含所有检测到的版面元素，包括没有文字的元素，并提供类别、检测置信度和多边形坐标。未使用 layout 流水线的 OCR-VL 当前仍返回空的 `regions` 数组。`text`、`markdown` 和 `html` 响应格式仅返回文档正文，不包含 JSON 区域元数据。

## 真实模型表格回归测试

`libs/orchion-ocr/tests/runtime_tables.rs` 使用规则表格截图 `tests/fixtures/table.png`、合并单元格截图 `merged-table.png` 和纯文字图片 `document.png`（源文件为相邻 SVG）。每张图片都覆盖 JSON、纯文本、Markdown、HTML：传统 OCR 必须明确拒绝 HTML，OCR-VL 则接受 HTML。测试检查识别文字、块级区域、无 HTML 回退的 Markdown 列，以及 HTML 中是否存在表格；OCR-VL 还对表格图验证 `task=table`。准备本地模型后，使用优化构建运行：

```sh
export ORCHION_OCR_TEST_MODELS="$PWD/data/models"
export ORCHION_OCR_TEST_TABLE_MODEL=/absolute/path/slanet_plus.onnx
export ORCHION_OCR_TEST_TABLE_DICTIONARY=/absolute/path/table_structure_dict_ch.txt
cargo test --release -p orchion-ocr --features ocr,ocr-vl \
  --test runtime_tables -- --ignored --nocapture --test-threads=1
```

CUDA 测试需增加 `cuda` feature，并设置 `ORCHION_OCR_TEST_DEVICE=cuda`。使用共享 Cargo 构建目录时，参考[显存测试文档](ocr-memory-testing.md)将 ONNX provider 库放到测试程序旁。

设置 `ORCHION_OCR_TEST_IMAGE` 可额外传入一张表格截图，与仓库测试图片一起验证传统 OCR 的 JSON、纯文本和 Markdown 路径。

### 真实 HTTP 格式矩阵

`sdks/orchion-client/tests/runtime_ocr_formats.rs` 从 `/v1/models` 获取所有已配置的 OCR 模型，逐个验证上述三张图片的全部四种响应格式。OCR-VL 还会对表格图验证 `task=table`。检查项目包括状态码、准确的 Content-Type、识别文字、JSON 字段、Markdown 语法和 HTML 结构。不支持的格式必须返回结构化的 400 错误，不能仅因 HTTP 200 就判定成功。合并单元格图片默认参与测试，不额外传图也能捕获 Markdown 回退成 HTML 的问题。设置 `ORCHION_OCR_TEST_IMAGE` 可加入其他实际表格截图。

```sh
ORCHION_OCR_HTTP_BASE_URL=http://127.0.0.1:9090 \
ORCHION_OCR_TEST_IMAGE=/absolute/path/merged-table.jpg \
cargo test -p orchion-client --no-default-features --features ocr \
  --test runtime_ocr_formats -- --ignored --nocapture
```

该测试只向已运行的服务发送推理请求，不启动、停止、修改配置或部署服务。服务端的 HTTP 序列化边界也有真实模型测试，沿用前面的模型环境变量：

```sh
cargo test -p orchion-server --lib response_tests -- --include-ignored --nocapture
```
