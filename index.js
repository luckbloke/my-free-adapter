import Schema from '@deepseek-ai/schemastery'

export const name = 'my-free-adapter'
export const inject = ['llm']

export const Config = Schema.object({
  providerId: Schema.string().default('my-free'),
  // Kilo Gateway 免注册端点
  baseUrl: Schema.string().default('https://api.kilo.ai/api/gateway'),
})

// Kilo 免注册通道的核心模型：kilo-auto/free
// 它会自动路由到当前最佳的免费模型，无需手动追踪
const CATALOG = {
  providerId: 'my-free',
  displayName: 'Kilo Free Gateway',
  models: [
    {
      id: 'kilo-auto/free',
      name: 'Auto Free (Kilo 自动路由)',
      contextWindow: 200000,  // 保守估计，实际由上游模型决定
      maxOutputTokens: 8192,
    },
  ],
}

export function apply(ctx, config) {
  const dispose = ctx.llm.registerAdapter({
    id: config.providerId,
    name: CATALOG.displayName,

    async listModels() {
      return CATALOG.models
    },

    async *streamChat({ model, messages, signal }) {
      // 关键：Kilo 免注册通道不需要 Authorization 头
      const response = await fetch(`${config.baseUrl}/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          // 故意不加 Authorization，这是免注册通道的核心
        },
        body: JSON.stringify({
          model: model.id,  // 直接用 'kilo-auto/free'，不要做替换
          messages,
          stream: true,
        }),
        signal,
      })

      if (!response.ok) {
        throw new Error(`Kilo gateway ${response.status}`)
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const data = line.slice(6).trim()
          if (data === '[DONE]') return

          try {
            const parsed = JSON.parse(data)
            const delta = parsed.choices?.[0]?.delta
            if (delta?.content) {
              yield { type: 'text-delta', text: delta.content }
            }
          } catch {
            // 忽略无法解析的行
          }
        }
      }
    },
  })

  ctx.effect(() => () => dispose())
}