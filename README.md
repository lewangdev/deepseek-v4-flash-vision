# deepseek-v4-flash-vision

给 OpenCode Go 订阅里的 **DeepSeek V4 Flash** 加上视觉（图片识别）能力的本地网关。

DeepSeek V4 Flash 是 OpenCode Go 订阅里最便宜的文本模型，但它不认图片。本网关在本地开一个多格式 API，把文本流量交给 DeepSeek V4 Flash，一旦检测到图片就自动把整条请求路由到视觉模型（默认 mimo-v2.5）。工具（Claude Code、Cursor、各种 agent SDK）只要把 BASE_URL 指到本网关即可即插即用，无需改代码。

```
┌──────────────┐   /v1/chat/completions (OpenAI Chat)
│  下游工具     │   /v1/responses        (OpenAI Responses)
│ (任意 client) │   /v1/messages         (Anthropic)      ┌──────────────┐
└──────┬───────┘─────────────────────────────────────────►│              │
       │  请求  ─►  格式转换 → 规范 IR → 路由器(检测图片)    │  本网关 (Go)  │
       │  ◄────────────────────────────────────────────   │              │
   三种响应格式 ◄─ 流式/非流式 转换 ←←（文本增量、工具调用汇总） │              │
                                                            └──────┬───────┘
                                      ┌─────────────────────────────┼──────────────────┐
                                      │ 有图片 ──► mimo-v2.5  ──► /chat/completions       │
              opencode.ai/zen/go/v1   │ 无图片 ──► deepseek-v4-flash ──► /chat/completions│
              (OpenCode Go 订阅)      │ overrides ──► grok-4.5 / gpt-5.6-luna ──► /responses│
                                      └───────────────────────────────────────────────────┘
```

核心是**规范 IR**（中间表示）：三种下游格式与三种上游端点各写一对「convert」，路由器只认 IR，不需要 N×M 的全矩阵转换。

## 功能

- **三个对外接口**：OpenAI `Chat Completions`、OpenAI `Responses`、Anthropic `Messages`。
- **自动路由**：请求里出现图片 → 视觉模型（默认 `mimo-v2.5`）；否则 → 文本模型 `deepseek-v4-flash`。`router.auto_vision: false` 可关闭。
- **多模型**：客户端显式指定的已知模型（`gpt-5.6-luna`、`grok-4.5` 等）按其对应的上游端点转发，未知模型名自动回落。
- **单文件配置**：`config.yaml`，无 Web 页面。
- **流式（SSE）**：跨接口转换时文本增量实时转发；工具调用的参数在流结束时作为完整块一次性下发（上游与下游同接口族的流式工具增量完全保留）。
- **辅助端点**：`/v1/models` 与 `/healthz`。

## 安装 / 构建

```sh
git clone https://github.com/lewangdev/deepseek-v4-flash-vision.git
cd deepseek-v4-flash-vision
make build          # 等价于 CGO_ENABLED=0 go build -o deepseek-v4-flash-vision .
```

> **构建与测试需用 `CGO_ENABLED=0`**（Makefile 已内置）。纯 Go 二进制跨平台、零依赖、可静态分发；并且在部分 macOS 沙箱环境下，cgo 链接 `net/http` 的可执行文件会因动态库问题无法 exec（报 `missing LC_UUID load command` —— 误导性报错，实为 cgo/动态链接依赖问题），关闭 cgo 即可绕开。这也让网关可以在极小的 Docker 镜像里跑（`scratch` + 一个二进制）。

## 快速开始

```sh
cp config.example.yaml config.yaml
# 编辑 config.yaml，填入 opencode.api_key（在 https://opencode.ai/auth 复制）
make run            # ./deepseek-v4-flash-vision -config config.yaml
```

配置（完整示例见 `config.example.yaml`）：

```yaml
server:
  address: "127.0.0.1:8787"   # 监听地址，默认仅本机
  api_key: ""                 # 非空则下游必须带 key；空 = 不校验
  default_max_tokens: 2048    # 上游是 /messages 且客户端没带 max_tokens 时的兜底

opencode:
  base_url: "https://opencode.ai/zen/go/v1"
  api_key: ""                 # OpenCode Go 订阅 key

router:
  primary: "deepseek-v4-flash"   # 文本默认
  vision: "mimo-v2.5"           # 检测到图片时使用（实测支持）
  auto_vision: true
  overrides:                     # 显式客户端模型名 -> 上游模型 + 端点
    "deepseek-v4-flash": { id: "deepseek-v4-flash", endpoint: "chat/completions" }
    "mimo-v2.5":         { id: "mimo-v2.5",         endpoint: "chat/completions" }
    "qwen3.7-max":       { id: "qwen3.7-max",       endpoint: "messages" }
    "gpt-5.6-luna":      { id: "gpt-5.6-luna",      endpoint: "responses" }
    "grok-4.5":          { id: "grok-4.5",          endpoint: "responses" }
```

注意 OpenCode Go 每个模型的接口格式**固定**：`chat/completions`（GLM / Kimi / DeepSeek / MiMo / Hy3）、`messages`（MiniMax / Qwen）、`responses`（Grok 4.5 / GPT 5.6 Luna）。`endpoint` 只能取三者之一，网关负责跨格式转换。

## 接入你的工具

**Anthropic 风格**（Claude Code 等）：设 `ANTHROPIC_BASE_URL` 指向网关根地址：

```sh
export ANTHROPIC_BASE_URL="http://127.0.0.1:8787"
export ANTHROPIC_API_KEY="anything"     # 网关不校验时随手填占位
```

**OpenAI 风格**（OpenAI SDK、Cursor、agent SDK 等）：指到 `/v1`：

```sh
export OPENAI_BASE_URL="http://127.0.0.1:8787/v1"
export OPENAI_API_KEY="anything"
```

`/v1/models` 会返回路由器已知的模型列表（OpenAI `list models` 格式），大多数 SDK 启动时会拿它做模型枚举。

## 路由规则

| 请求特征 | 模型 | 上游端点 |
|---|---|---|
| 含图片 + `auto_vision` | `router.vision`（默认 `mimo-v2.5`） | 该模型的 `overrides` 端点（默认 `chat/completions`） |
| 纯文本 | `router.primary`（默认 `deepseek-v4-flash`） | `chat/completions` |
| 显式指定已知非主/视觉模型（如 `gpt-5.6-luna`） | 该模型 | `overrides` 里配的端点 |
| 未知模型名 | 自动回落（同上两行） | — |

## 视觉模型的选择与边界

默认视觉模型是 **`mimo-v2.5`**。它的能力**实测验证过**：能接收 base64 图片并正确描述内容（"这张图片是纯红色的"）。它还与默认文本模型 DeepSeek V4 Flash **同属 `chat/completions` 接口**——默认路径全程同族，SSE 天然透传，最稳最省。

订阅里实测过的候选（各有取舍，均可改 `router.vision` + 加对应 override 切换）：

| 模型 | 接口 | 图片输入 | 实测 |
|---|---|---|---|
| **mimo-v2.5** ✅ 默认 | chat/completions | ✅ | 能正确描述图片 |
| qwen3.8-max | messages | ✅ | 能正确描述图片 |
| minimax-m3 | messages | ✅ | 能正确描述图片 |
| gpt-5.6-luna | responses | ✅ | 接受图片输入 |
| glm-5.2 | chat/completions | ⚠️ | 接受但不解析内容 |
| qwen3.7-max | messages | ❌ | 返回 400（纯文本） |
| grok-4.5 / mimo-v2-pro | — | ❌ | 拒绝图片 |
| mimo-v2-omni | — | — | 已废弃，官方提示迁移 mimo-v2.5 |

选 `messages` / `responses` 接口的视觉模型时，客户端是 chat 风格的话会走跨族转换（工具参数块在流结束时一次性下发）；默认的 mimo-v2.5 没有这个开销。DeepSeek V4 Flash 本身只走 `chat/completions`，不接图片。

## 验证

```sh
make test        # CGO_ENABLED=0 go test -count=1 ./...  （convert / router / streamconv / server 全量 e2e）
make vet         # go vet ./...
curl http://127.0.0.1:8787/healthz
curl http://127.0.0.1:8787/v1/models
```

`internal/server/e2e_test.go` 用假上游覆盖了完整 HTTP 路径：文本透传、图片跨族转换与流式、Anthropic 客户端 → DeepSeek 文本、工具调用双向映射等。

## 诚实的边界

- `/models` 元数据端点没有 `capability_*` 标记，本 README 中每款模型的图片能力都来自**真实订阅实测**（上文表格）；订阅模型列表变化时以 `curl /models` 为准。
- 跨族流式只保证**文本增量实时**；工具参数块在流结束时一次性下发（同族流式工具增量完全保留）。
- v1 不做本地缓存 / 持久化 / 多用户鉴权；上游真实鉴权头以 OpenCode 实际要求为准，`opencode.headers` 可覆盖。

## License

MIT。模块路径 `github.com/lewangdev/deepseek-v4-flash-vision`（fork 时可全局替换为自己的路径）。
