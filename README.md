# dsh-my-free-adapter

为 [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) 提供免费模型的 LLM Provider 插件。

- **Kilo（默认启用）**：`https://api.kilo.ai/api/gateway`，无需 API Key，自动只暴露免费且支持工具调用的文本模型。
- **OpenCode Zen（默认关闭）**：`https://opencode.ai/zen/v1`。Zen 的免费层现在只允许官方 OpenCode 客户端访问，第三方客户端会收到 `403 FreeTierError`，因此改为显式 opt-in。

## 功能

- 从网关 `/models` 动态发现模型，并缓存 7 天；刷新失败时回退到内置快照，不影响已注册的 Provider。
- 只列出「免费 + 支持 tools + 文本输出」的模型。
- 正确的 DSH 0.2.x 适配器契约：
  - 异步 `listModels` / `resolveModel`；
  - 把 DSH 的结构化消息（system / developer / user / assistant / tool）转换为 pi-ai 上下文，工具调用与工具结果在后续轮次中保持完整；
  - 工具调用流式分片携带真实 `id` / `name`，`arguments` 保持原始 JSON 字符串；
  - usage、缓存 token、reasoning token、finish reason 与错误码完整映射；
  - 把 provider 的上下文超限错误归类为 `CONTEXT_WINDOW_EXCEEDED`，让 DSH 走自动压缩；
  - 请求前的 prompt 估算 + 输出预算三层钳制，避免 prompt 与输出叠加超窗。
- 图片输入：当模型声明支持图片且宿主提供 attachment 服务时，按请求尺寸读取并内联；否则给出明确的 `UNSUPPORTED_CONTENT`。
- 文件输入：通过 `ctx.llm.fileRequestText()` 投影为宿主一致的只读句柄文本。
- 所有 HTTP 请求都有超时；流式响应有 120 秒空闲看门狗。

## 安装

在本仓库构建后，用 DSH 的插件管理器安装本地链接：

```powershell
pnpm install
pnpm run build
dsh plugin install link:D:\my-free-adapter
```

也可以使用 `plugin_manager` / Web GUI 的「设置 → 插件」安装同一个 `link:` 路径。

安装后：

- 插件条目：`include:my-free`（`dsh-my-free-adapter`）。
- Provider 路由：`my-kilo`（以及启用后的 `my-zen`）。
- 目录状态：`~/.my-free-adapter/kilo-status.json`、`~/.my-free-adapter/zen-status.json`。

## 配置

在 profile 的 `cordis.patch.yml` 中按需覆盖：

```yaml
- id: my-free
  name: dsh-my-free-adapter
  config:
    kiloEnabled: true
    kiloProviderId: my-kilo
    kiloBaseUrl: https://api.kilo.ai/api/gateway
    kiloApiKey: ''
    zenEnabled: false
    zenProviderId: my-zen
    zenBaseUrl: https://opencode.ai/zen/v1
    zenApiKey: public
    zenCapabilitiesUrl: https://models.opencode.ai/api.json
    maxOutputTokens: 32768
    dataDir: ''
```

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `kiloEnabled` | `true` | 是否注册 Kilo 路由 |
| `kiloProviderId` | `my-kilo` | Provider 路由名 |
| `kiloBaseUrl` | Kilo Gateway | 网关根地址（不要带 `/models`） |
| `kiloApiKey` | 空 | 可选。为空时使用网关的 `anonymous` 访客令牌 |
| `zenEnabled` | `false` | 是否注册 Zen 路由（免费层目前会 403） |
| `zenApiKey` | `public` | Zen 使用的 Bearer token |
| `maxOutputTokens` | `32768` | 单次请求输出硬上限（同时参与声明预算的 25% 窗口钳制） |
| `dataDir` | `~/.my-free-adapter` | 模型缓存与状态文件目录 |

## 已知限制

- **OpenCode Zen**：`403 FreeTierError — OpenCode's free tier can only be used from within OpenCode`。这不是本插件能通过补 header 修复的问题；Zen 免费层已改为客户端契约限制。需要 Zen 时请提供可用的 `zenApiKey` 并自行确认授权方式。
- **Kilo 免费模型列表会变化**：以 `~/.my-free-adapter/kilo-status.json` 的 `exposed` / `lastError` 为准。
- **隐私**：Kilo 的部分免费模型带有 `mayTrainOnYourPrompts` 标记，免费线路也可能记录提示词；不要向免费模型发送密钥、隐私数据或未公开代码。
- 免费模型可能很慢、限流或返回纯 reasoning；插件会把它们映射成标准的 `RATE_LIMIT` / `TIMEOUT` / `EMPTY_RESPONSE` 错误码，交给 DSH 的重试层处理。
- 插件遵循 DSH 的 attribution 约定发送 `user-agent`；Zen 路由因网关要求改用 OpenCode UA。

## 开发

```powershell
pnpm install
pnpm run typecheck
pnpm run build
node tests/smoke.mjs   # 用假 Context 加载构建产物，验证注册/列模型/流式
node tests/live.mjs    # 访问真实 Kilo 网关的端到端测试（需要网络）
```

主要文件：

- `src/index.ts`：Cordis 插件入口、配置、Provider 注册。
- `src/catalog.ts`：模型目录、免费判定、缓存、状态、超时。
- `src/adapter.ts`：pi-ai Provider、DSH 消息/流协议转换。
- `src/budget.ts`：prompt 估算与输出预算。
