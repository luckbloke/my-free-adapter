import { ModelCatalog, modelInfo, type KiloModel } from './catalog.ts'
import {
  estimateRequestTokens,
  resolveWireBudget,
  blendCalibration,
  overflowMessage,
  MIN_ANSWER_TOKENS,
} from './budget.ts'
import { CONTEXT_WINDOW_EXCEEDED, type HarnessChunk, type FinishReason } from './events.ts'

export interface HarnessMessage {
  role: 'system' | 'user' | 'assistant'
  content: unknown
}

export interface HarnessGenerateOptions {
  provider: string
  model: string
  messages: HarnessMessage[]
  system?: string
  tools?: Array<{ name: string; description: string; parameters: unknown }>
  maxTokens?: number
  temperature?: number
  signal?: AbortSignal
}

export interface LlmModelReasoningInfo {
  efforts: ReadonlyArray<{ id: string; name: string }>
}

export class KiloFreeAdapter {
  #catalog: ModelCatalog
  #providerId: string
  #baseUrl: string
  #calibration = new Map<string, number>()

  constructor(catalog: ModelCatalog, providerId: string, baseUrl: string) {
    this.#catalog = catalog
    this.#providerId = providerId
    this.#baseUrl = baseUrl.replace(/\/+$/, '')
  }

  providerInfo(provider: string): { id: string; name: string } {
    return { id: provider, name: 'Kilo Free Gateway' }
  }

  listModels(provider: string): Array<{ provider: string; id: string; name: string; inputModalities: string[] }> {
    return this.#catalog.list().map((id) => {
      const detail = this.#catalog.get(id)
      const info = detail ? modelInfo(detail) : { name: id, inputModalities: ['text'] }
      return { provider, id, name: info.name ?? id, inputModalities: info.inputModalities ?? ['text'] }
    })
  }

  resolveModel(provider: string, model: string): {
    provider: string
    id: string
    name: string
    inputModalities: string[]
    context: { contextWindow: number }
    defaultMaxTokens: number
    reasoning?: LlmModelReasoningInfo
  } {
    const detail = this.#catalog.get(model)
    const info = detail
      ? modelInfo(detail)
      : { id: model, name: model, contextWindow: 262144, maxTokens: 32768, inputModalities: ['text'], reasoning: false, supportsTools: true }
    const reasoning = info.reasoning
      ? { efforts: ['minimal', 'low', 'medium', 'high', 'xhigh', 'max'].map((id) => ({ id, name: id })) }
      : undefined
    return {
      provider,
      id: model,
      name: info.name,
      inputModalities: info.inputModalities,
      context: { contextWindow: info.contextWindow },
      defaultMaxTokens: info.maxTokens,
      ...(reasoning ? { reasoning } : {}),
    }
  }

  async *stream(options: HarnessGenerateOptions): AsyncGenerator<HarnessChunk> {
    const modelId = options.model.trim()
    const detail = this.#catalog.get(modelId)
    if (!detail) {
      yield { type: 'finish', reason: { kind: 'error', failure: { message: `model ${modelId} not found`, code: 'MODEL_NOT_FOUND' } } }
      return
    }
    const info = modelInfo(detail)
    const rawInput = estimateRequestTokens(options.messages, options.system)
    const budget = resolveWireBudget({
      requested: options.maxTokens,
      modelMaxTokens: info.maxTokens,
      contextWindow: info.contextWindow,
      estimatedInputTokens: rawInput,
      calibration: this.#calibration.get(modelId),
    })
    if (budget.overflow) {
      yield { type: 'usage', usage: { inputTokens: budget.effectiveInputTokens, outputTokens: 0 } }
      yield { type: 'finish', reason: { kind: 'error', failure: { message: overflowMessage(budget.effectiveInputTokens, info.contextWindow), code: CONTEXT_WINDOW_EXCEEDED } } }
      return
    }
    const wireMax = budget.maxTokens ?? MIN_ANSWER_TOKENS

    const res = await fetch(`${this.#baseUrl}/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: modelId,
        messages: options.messages.map((m) => ({ role: m.role, content: m.content })),
        max_tokens: wireMax,
        temperature: options.temperature,
        stream: true,
        stream_options: { include_usage: true },
      }),
      signal: options.signal,
    })

    if (!res.ok) {
      yield { type: 'finish', reason: { kind: 'error', failure: { message: `Kilo gateway ${res.status}`, code: 'UPSTREAM' } } }
      return
    }

    const reader = res.body!.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    let reportedInput = 0
    let reportedOutput = 0
    let started = false
    let finished = false

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        const data = line.slice(6).trim()
        if (data === '[DONE]') { finished = true; break }
        try {
          const parsed = JSON.parse(data)
          const delta = parsed.choices?.[0]?.delta
          if (!started && delta?.content) {
            started = true
            yield { type: 'block-start', index: 0, blockType: 'text' }
          }
          if (delta?.content) yield { type: 'text-delta', index: 0, text: delta.content }
          if (parsed.usage) {
            reportedInput = parsed.usage.prompt_tokens ?? 0
            reportedOutput = parsed.usage.completion_tokens ?? 0
          }
        } catch { /* ignore */ }
      }
      if (finished) break
    }

    if (started) yield { type: 'block-end', index: 0, block: { type: 'text', text: '' } }
    yield { type: 'usage', usage: { inputTokens: reportedInput, outputTokens: reportedOutput } }
    yield { type: 'finish', reason: { kind: 'stop' } }

    if (reportedInput > 0 && rawInput > 0) {
      this.#calibration.set(modelId, blendCalibration(this.#calibration.get(modelId), reportedInput / rawInput))
    }
  }
}