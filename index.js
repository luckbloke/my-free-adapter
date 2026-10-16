import Schema from '@deepseek-ai/schemastery'
import { LlmAdapter } from '@deepseek-ai/dsh-llm'

export const name = 'my-free-adapter'
export const inject = ['llm']

export const Config = Schema.object({
  providerId: Schema.string().default('my-free'),
  baseUrl: Schema.string().default('https://api.kilo.ai/api/gateway'),
})

class KiloFreeAdapter extends LlmAdapter {
  constructor(config) {
    super()
    this.config = config
  }

  async *stream(options) {
    const response = await fetch(`${this.config.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: options.model,
        messages: options.messages,
        stream: true,
      }),
      signal: options.signal,
    })

    if (!response.ok) {
      throw new Error(`Kilo gateway ${response.status}`)
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    // 遵守 DSH StreamChunk 协议：先 block-start
    yield { type: 'block-start', index: 0, blockType: 'text' }

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        const data = line.slice(6).trim()
        if (data === '[DONE]') {
          // 正常结束
          yield { type: 'block-end', index: 0, block: { type: 'text', text: '' } }
          yield { type: 'finish', reason: { kind: 'stop' } }
          return
        }

        try {
          const parsed = JSON.parse(data)
          const delta = parsed.choices?.[0]?.delta
          if (delta?.content) {
            yield { type: 'text-delta', index: 0, text: delta.content }
          }
        } catch {
          // 忽略
        }
      }
    }

    // 流意外结束时的兜底
    yield { type: 'block-end', index: 0, block: { type: 'text', text: '' } }
    yield { type: 'finish', reason: { kind: 'stop' } }
  }
}

export function apply(ctx, config) {
  // 关键修复：第一个参数是 provider 路由数组，不是对象
  const adapter = new KiloFreeAdapter(config)
  const dispose = ctx.llm.registerAdapter([config.providerId], adapter)

  ctx.effect(() => () => dispose())
}