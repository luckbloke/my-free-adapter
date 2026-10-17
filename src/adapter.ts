import { createHash } from 'node:crypto'
import {
  CONTEXT_WINDOW_EXCEEDED_CODE,
  EMPTY_RESPONSE_CODE,
  LlmAdapter,
  LlmError,
  attributionHeaders,
  isContextWindowExceededError,
  isQuotaExceededError,
} from '@deepseek-ai/dsh-llm'
import type {
  ContentBlock,
  FinishReason,
  GenerateOptions,
  ImageBlock,
  LlmModelInfo,
  LlmResolvedModelInfo,
  ModelModality,
  ReasoningEffortId,
  StreamChunk,
  TokenUsage,
  ToolCallId,
} from '@deepseek-ai/dsh-llm'
import { createProvider } from '@earendil-works/pi-ai'
import type {
  Api,
  AssistantMessage as PiAssistantMessage,
  AssistantMessageEvent,
  Context as PiContext,
  ImageContent,
  Message as PiMessage,
  Model,
  SimpleStreamOptions,
  TextContent,
  ThinkingLevel,
  Usage as PiUsage,
} from '@earendil-works/pi-ai'
import * as openaiCompletions from '@earendil-works/pi-ai/api/openai-completions'
import * as openaiResponses from '@earendil-works/pi-ai/api/openai-responses'

import { ModelCatalog, modelInfo, type ModelInfo, type ModelRecord } from './catalog.ts'
import {
  MIN_ANSWER_TOKENS,
  blendCalibration,
  estimateRequestTokens,
  overflowMessage,
  resolveWireBudget,
} from './budget.ts'

const STREAM_IDLE_TIMEOUT_MS = 120_000
const MAX_IMAGE_BYTES = 5 * 1024 * 1024
const CATALOG_READY_TIMEOUT_MS = 2_000
const DEFAULT_CONTEXT_WINDOW = 262144
const DEFAULT_MAX_TOKENS = 32768

const EFFORT_LABELS: Record<string, string> = {
  minimal: 'Minimal',
  low: 'Low',
  medium: 'Medium',
  high: 'High',
  xhigh: 'XHigh',
  max: 'Max',
}

/** Durable image reference shape the attachment service accepts. */
export interface ImageRefLike {
  attachmentId: string
  mediaType: string
  bytes: number
  width: number
  height: number
  name?: string
}

/** Durable file reference shape projected to model-visible handle text. */
export interface FileRefLike {
  attachmentId: string
  name: string
  bytes: number
}

export interface AttachmentStoreLike {
  readImageRequest(
    ref: ImageRefLike,
    target: { width: number; height: number; maxBytes: number },
    signal?: AbortSignal,
  ): Promise<{ data: Uint8Array; mediaType: string; bytes: number }>
}

export interface FreeAdapterOptions {
  catalog: ModelCatalog
  providerId: string
  baseUrl: string
  providerName: string
  /** Credential sent as a bearer token. */
  apiKey?: string
  /** Credential used when `apiKey` is blank (keyless gateways need a sentinel). */
  defaultKey: string
  /** Hard per-request output ceiling for this route. */
  maxOutputTokens?: number
  /** Resolve the optional durable attachment service at request time. */
  resolveAttachments?: () => AttachmentStoreLike | undefined
  /** Project one durable file reference into the host's deterministic handle text. */
  resolveFileText?: (ref: FileRefLike) => string
}

function numeric(value: unknown, fallback: number): number {
  if (typeof value === 'number' && Number.isFinite(value) && value > 0) return value
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value)
    if (Number.isFinite(parsed) && parsed > 0) return parsed
  }
  return fallback
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function toModalities(values: readonly string[]): ModelModality[] {
  const modalities: ModelModality[] = []
  for (const value of values) {
    if ((value === 'text' || value === 'image') && !modalities.includes(value)) modalities.push(value)
  }
  return modalities.length > 0 ? modalities : ['text']
}

function flattenText(blocks: readonly ContentBlock[] | string | undefined): string {
  if (typeof blocks === 'string') return blocks
  if (!Array.isArray(blocks)) return ''
  return blocks.filter((block) => block.type === 'text').map((block) => block.text).join('')
}

function contentHasImage(blocks: readonly ContentBlock[] | string | undefined): boolean {
  if (!Array.isArray(blocks)) return false
  return blocks.some((block) => block.type === 'image' && block.offloaded !== true)
}

function parseArguments(raw: string): Record<string, unknown> {
  try {
    const parsed: unknown = JSON.parse(raw || '{}')
    return parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed)
      ? parsed as Record<string, unknown>
      : {}
  } catch {
    return {}
  }
}

function zeroUsage(): PiUsage {
  return {
    input: 0,
    output: 0,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: 0,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  }
}

function mapUsage(usage: PiUsage): TokenUsage {
  return {
    inputTokens: Number.isFinite(usage.input) ? usage.input : 0,
    outputTokens: Number.isFinite(usage.output) ? usage.output : 0,
    ...Number.isFinite(usage.totalTokens) ? { totalTokens: usage.totalTokens } : {},
    ...usage.cacheRead > 0 ? { cacheReadTokens: usage.cacheRead } : {},
    ...usage.cacheWrite > 0 ? { cacheWriteTokens: usage.cacheWrite } : {},
    ...typeof usage.reasoning === 'number' && usage.reasoning > 0 ? { reasoningTokens: usage.reasoning } : {},
  }
}

/** Classify provider text into Harness failure codes the retry layer understands. */
function classifyError(message: string): string {
  if (/FreeTierError|free tier can only be used from within OpenCode/i.test(message)) return 'FREE_TIER_GATED'
  if (isQuotaExceededError(message)) return 'QUOTA'
  if (/\b(?:401|403)\b|unauthori[sz]ed|forbidden/i.test(message)) return 'AUTH'
  if (/\b429\b|rate.?limit/i.test(message)) return 'RATE_LIMIT'
  if (/\b413\b|failed to buffer the request body|payload too large|request body too large/i.test(message)) {
    return 'INVALID_REQUEST'
  }
  if (/\b400\b|invalid.?request/i.test(message)) return 'INVALID_REQUEST'
  if (/\b5\d\d\b/.test(message)) return 'SERVER'
  if (/\btime(?:d)?\s*out\b|timeout/i.test(message)) return 'TIMEOUT'
  if (/stream ended (?:before|without)\b/i.test(message)) return 'TRANSPORT'
  if (/\b(?:network|connection|socket|fetch)\b|\bECONN[A-Z]+\b|other side closed|terminated|premature close/i.test(message)) {
    return 'TRANSPORT'
  }
  return 'UPSTREAM'
}

function mapStopReason(message: PiAssistantMessage, contextWindow: number): FinishReason {
  const detail = message.errorMessage ?? ''
  if (message.stopReason === 'error' && isContextWindowExceededError(detail)) {
    return {
      kind: 'error',
      failure: { message: detail, code: CONTEXT_WINDOW_EXCEEDED_CODE },
    }
  }
  switch (message.stopReason) {
    case 'stop':
      if (message.content.length === 0) {
        return {
          kind: 'error',
          failure: {
            message: `model "${message.model}" completed without any content`,
            code: EMPTY_RESPONSE_CODE,
          },
        }
      }
      return { kind: 'stop' }
    case 'length': {
      const used = (message.usage?.input ?? 0) + (message.usage?.output ?? 0)
      if (contextWindow > 0 && used >= contextWindow) {
        return {
          kind: 'error',
          failure: {
            message: `output reached the model context window (${used}/${contextWindow} tokens)`,
            code: CONTEXT_WINDOW_EXCEEDED_CODE,
          },
        }
      }
      return { kind: 'max-tokens' }
    }
    case 'toolUse':
      return { kind: 'tool-calls' }
    case 'aborted':
      return { kind: 'aborted', failure: { message: detail || 'pi-ai stream aborted', code: 'ABORTED' } }
    case 'error':
      return { kind: 'error', failure: { message: detail || 'pi-ai stream error', code: classifyError(detail) } }
    default:
      return { kind: 'error', failure: { message: detail || 'pi-ai stream error', code: 'UPSTREAM' } }
  }
}

/** `AbortSignal.any` without requiring it in the TypeScript lib. */
function anySignal(signals: AbortSignal[]): AbortSignal {
  const any = (AbortSignal as unknown as { any?: (values: AbortSignal[]) => AbortSignal }).any
  return any ? any.call(AbortSignal, signals) : signals[0] ?? new AbortController().signal
}

/** Stable, non-reversible session identity; raw prompt text never reaches a header. */
function sessionSeed(options: GenerateOptions): string {
  if (options.sessionId) return String(options.sessionId)
  for (const message of options.messages) {
    if (message.role !== 'user') continue
    const text = flattenText(message.content)
    if (text) return createHash('sha256').update(text).digest('hex')
  }
  return 'my-free-adapter-default-session'
}

function injectStop(payload: unknown, stop: readonly string[]): unknown {
  if (payload !== null && typeof payload === 'object') {
    (payload as Record<string, unknown>).stop = [...stop]
  }
  return payload
}

abstract class BaseFreeAdapter extends LlmAdapter {
  protected catalog: ModelCatalog
  protected providerId: string
  protected baseUrl: string
  protected apiKey: string
  protected providerName: string
  protected outputCap: number
  protected provider: ReturnType<typeof createProvider<Api>>
  protected resolveAttachments?: () => AttachmentStoreLike | undefined
  protected resolveFileText?: (ref: FileRefLike) => string

  #calibration = new Map<string, number>()

  constructor(options: FreeAdapterOptions) {
    super()
    this.catalog = options.catalog
    this.providerId = options.providerId
    this.baseUrl = options.baseUrl.replace(/\/+$/, '')
    this.providerName = options.providerName
    this.apiKey = options.apiKey?.trim() || options.defaultKey
    this.outputCap = numeric(options.maxOutputTokens, DEFAULT_MAX_TOKENS)
    this.resolveAttachments = options.resolveAttachments
    this.resolveFileText = options.resolveFileText

    this.provider = createProvider<Api>({
      id: options.providerId,
      name: options.providerName,
      baseUrl: this.baseUrl,
      auth: {
        apiKey: {
          name: `${options.providerName} credential`,
          resolve: async () => ({ auth: { apiKey: this.apiKey } }),
        },
      },
      models: [],
      api: {
        'openai-completions': openaiCompletions,
        'openai-responses': openaiResponses,
      },
    })
  }

  /** Wire protocol for one catalog record (Zen routes some models to Responses). */
  protected resolveApiId(_detail: ModelRecord | undefined): Api {
    return 'openai-completions'
  }

  /** pi-ai compatibility overrides for this gateway dialect. */
  protected compatFor(_detail: ModelRecord | undefined, _api: Api): Model<Api>['compat'] {
    return undefined
  }

  /** Headers required by the gateway (and Harness attribution). */
  protected abstract buildHeaders(options: GenerateOptions): Record<string, string>

  override providerInfo(provider: string) {
    return { id: provider, name: this.providerName }
  }

  override async listModels(provider: string): Promise<readonly LlmModelInfo[]> {
    await this.catalog.ready(CATALOG_READY_TIMEOUT_MS)
    return this.catalog.list().map((id) => {
      const detail = this.catalog.get(id)
      const info = detail ? modelInfo(detail, this.outputCap) : this.fallbackInfo(id)
      return {
        provider,
        id,
        name: info.name,
        inputModalities: toModalities(info.inputModalities),
      }
    })
  }

  override async resolveModel(provider: string, model: string, _signal?: AbortSignal): Promise<LlmResolvedModelInfo> {
    await this.catalog.ready(CATALOG_READY_TIMEOUT_MS)
    const detail = this.catalog.get(model)
    const info = detail ? modelInfo(detail, this.outputCap) : this.fallbackInfo(model)
    const efforts = info.reasoning && info.reasoningEfforts?.length
      ? info.reasoningEfforts.map((id) => ({
          id: id as unknown as ReasoningEffortId,
          name: EFFORT_LABELS[id] ?? id,
        }))
      : undefined
    return {
      provider,
      id: model,
      name: info.name,
      inputModalities: toModalities(info.inputModalities),
      context: { contextWindow: info.contextWindow },
      defaultMaxTokens: info.maxTokens,
      ...efforts ? { reasoning: { efforts } } : {},
    }
  }

  private fallbackInfo(id: string): ModelInfo {
    return {
      id,
      name: id,
      contextWindow: DEFAULT_CONTEXT_WINDOW,
      maxTokens: Math.min(this.outputCap, DEFAULT_MAX_TOKENS),
      inputModalities: ['text'],
      reasoning: false,
      supportsTools: true,
    }
  }

  protected toPiModel(detail: ModelRecord, headers: Record<string, string>): Model<Api> {
    const info = modelInfo(detail, this.outputCap)
    const pricing = detail.pricing ?? {}
    const api = this.resolveApiId(detail)
    const compat = this.compatFor(detail, api)
    return {
      id: detail.id,
      name: info.name,
      api,
      provider: this.providerId,
      baseUrl: this.baseUrl,
      reasoning: info.reasoning,
      ...info.reasoning ? { thinkingLevelMap: { off: null, low: 'low', medium: 'medium', high: 'high' } } : {},
      input: toModalities(info.inputModalities),
      cost: {
        input: numeric(pricing.prompt, 0),
        output: numeric(pricing.completion, 0),
        cacheRead: 0,
        cacheWrite: 0,
      },
      contextWindow: info.contextWindow,
      maxTokens: info.maxTokens,
      headers,
      ...compat ? { compat } : {},
    } as Model<Api>
  }

  private fallbackModel(id: string, headers: Record<string, string>): Model<Api> {
    const api: Api = 'openai-completions'
    const compat = this.compatFor(undefined, api)
    return {
      id,
      name: id,
      api,
      provider: this.providerId,
      baseUrl: this.baseUrl,
      reasoning: false,
      input: ['text'],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: DEFAULT_CONTEXT_WINDOW,
      maxTokens: Math.min(this.outputCap, DEFAULT_MAX_TOKENS),
      headers,
      ...compat ? { compat } : {},
    } as Model<Api>
  }

  override async *stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    const modelId = options.model.trim()
    await this.catalog.ready(CATALOG_READY_TIMEOUT_MS)
    const detail = this.catalog.get(modelId)
    const info = detail ? modelInfo(detail, this.outputCap) : this.fallbackInfo(modelId)
    const headers = this.buildHeaders(options)

    const rawInput = estimateRequestTokens({
      messages: options.messages as any,
      system: options.system,
      tools: options.tools as any,
    })

    const budget = resolveWireBudget({
      requested: options.maxTokens,
      modelMaxTokens: info.maxTokens,
      contextWindow: info.contextWindow,
      estimatedInputTokens: rawInput,
      calibration: this.#calibration.get(modelId),
    })

    if (budget.overflow) {
      yield { type: 'usage', usage: { inputTokens: budget.effectiveInputTokens, outputTokens: 0 } }
      yield {
        type: 'finish',
        reason: {
          kind: 'error',
          failure: {
            message: overflowMessage(budget.effectiveInputTokens, info.contextWindow),
            code: CONTEXT_WINDOW_EXCEEDED_CODE,
          },
        },
      }
      return
    }

    const wireMax = budget.maxTokens ?? MIN_ANSWER_TOKENS
    const model = detail ? this.toPiModel(detail, headers) : this.fallbackModel(modelId, headers)

    if (options.messages.some((message) => contentHasImage(message.content)) && !model.input.includes('image')) {
      throw new LlmError(
        `model "${modelId}" does not accept image input (text-only modality)`,
        'UNSUPPORTED_CONTENT',
      )
    }

    const context = await this.toPiContext(options, model)

    const effort = String(options.reasoningEffort ?? '')
    const reasoning = ['minimal', 'low', 'medium', 'high', 'xhigh', 'max'].includes(effort)
      ? effort as ThinkingLevel
      : undefined
    const stop = options.stop?.length ? options.stop : undefined

    const consumer = new AbortController()
    const upstream = options.signal ? anySignal([options.signal, consumer.signal]) : consumer.signal
    const piOptions: SimpleStreamOptions = {
      apiKey: this.apiKey,
      maxRetries: 0,
      signal: upstream,
      maxTokens: wireMax,
      ...options.temperature !== undefined ? { temperature: options.temperature } : {},
      ...reasoning ? { reasoning } : {},
      ...options.sessionId ? { sessionId: String(options.sessionId) } : {},
      ...stop ? { onPayload: (payload: unknown) => injectStop(payload, stop) } : {},
    }

    let idleTimer: NodeJS.Timeout | undefined
    const armIdle = (): void => {
      if (idleTimer) clearTimeout(idleTimer)
      idleTimer = setTimeout(() => consumer.abort(new Error('stream idle timeout')), STREAM_IDLE_TIMEOUT_MS)
      idleTimer.unref?.()
    }

    const toolIds = new Map<number, { id: string; name: string }>()
    let calibrated = false
    const calibrate = (usage: PiUsage): void => {
      const billedInput = (usage.input ?? 0) + (usage.cacheRead ?? 0) + (usage.cacheWrite ?? 0)
      if (calibrated || billedInput <= 0 || rawInput <= 0) return
      calibrated = true
      this.#calibration.set(modelId, blendCalibration(this.#calibration.get(modelId), billedInput / rawInput))
    }

    try {
      armIdle()
      const events = this.provider.streamSimple(model, context, piOptions)
      for await (const event of events as AsyncIterable<AssistantMessageEvent>) {
        armIdle()
        switch (event.type) {
          case 'start':
            break
          case 'text_start':
            yield { type: 'block-start', index: event.contentIndex, blockType: 'text' }
            break
          case 'text_delta':
            yield { type: 'text-delta', index: event.contentIndex, text: event.delta }
            break
          case 'text_end':
            yield { type: 'block-end', index: event.contentIndex, block: { type: 'text', text: event.content } }
            break
          case 'thinking_start':
            yield { type: 'block-start', index: event.contentIndex, blockType: 'reasoning' }
            break
          case 'thinking_delta':
            yield { type: 'reasoning-delta', index: event.contentIndex, text: event.delta }
            break
          case 'thinking_end':
            yield {
              type: 'block-end',
              index: event.contentIndex,
              block: { type: 'reasoning', text: event.content },
            }
            break
          case 'toolcall_start': {
            const partial = event.partial.content[event.contentIndex]
            const call = partial?.type === 'toolCall' ? partial : undefined
            toolIds.set(event.contentIndex, { id: call?.id ?? '', name: call?.name ?? '' })
            yield { type: 'block-start', index: event.contentIndex, blockType: 'tool-call' }
            break
          }
          case 'toolcall_delta': {
            const known = toolIds.get(event.contentIndex)
            yield {
              type: 'tool-call-delta',
              index: event.contentIndex,
              id: (known?.id ?? '') as ToolCallId,
              ...known?.name ? { name: known.name } : {},
              argumentsDelta: event.delta,
            }
            break
          }
          case 'toolcall_end':
            yield {
              type: 'block-end',
              index: event.contentIndex,
              block: {
                type: 'tool-call',
                id: event.toolCall.id as ToolCallId,
                name: event.toolCall.name,
                arguments: JSON.stringify(event.toolCall.arguments ?? {}),
              },
            }
            break
          case 'done': {
            calibrate(event.message.usage)
            yield { type: 'usage', usage: mapUsage(event.message.usage) }
            yield { type: 'finish', reason: mapStopReason(event.message, info.contextWindow) }
            return
          }
          case 'error': {
            calibrate(event.error.usage)
            yield { type: 'usage', usage: mapUsage(event.error.usage) }
            const message = options.signal?.aborted
              ? { ...event.error, stopReason: 'aborted' as const }
              : event.error
            yield { type: 'finish', reason: mapStopReason(message, info.contextWindow) }
            return
          }
        }
      }
      throw new LlmError('pi-ai event stream ended without done/error', 'STREAM_CLOSED')
    } catch (error) {
      if (options.signal?.aborted) {
        throw new LlmError('request aborted by caller', 'ABORTED', { cause: error })
      }
      if (consumer.signal.aborted) {
        throw new LlmError(
          `upstream stream was idle for ${STREAM_IDLE_TIMEOUT_MS}ms`,
          'TIMEOUT',
          { cause: error },
        )
      }
      throw error
    } finally {
      if (idleTimer) clearTimeout(idleTimer)
      consumer.abort(new Error('stream consumer finished'))
    }
  }

  private async toPiContext(options: GenerateOptions, model: Model<Api>): Promise<PiContext> {
    const systemParts: string[] = []
    if (typeof options.system === 'string' && options.system.trim()) systemParts.push(options.system.trim())

    const messages: PiMessage[] = []
    const toolNames = new Map<string, string>()
    let index = 0

    // The loop puts its system prompt in a leading system message; fold that
    // into pi-ai's dedicated slot so the provider sees a normal system prompt.
    for (; index < options.messages.length; index += 1) {
      const message = options.messages[index]!
      if (message.role !== 'system') break
      const text = flattenText(message.content)
      if (text) systemParts.push(text)
    }

    for (; index < options.messages.length; index += 1) {
      const message = options.messages[index]!
      if (message.role === 'system' || message.role === 'developer') {
        // This adapter declares no in-history tool/system update protocol, so
        // the complete current tool list is sent each request. Preserve any
        // textual instruction; tool-addition/removal blocks are redundant.
        const text = flattenText(message.content)
        if (text) systemParts.push(text)
        continue
      }
      if (message.role === 'assistant') {
        const assistant = this.toPiAssistant(message, model, toolNames)
        if (assistant.content.length > 0) messages.push(assistant)
        continue
      }
      if (message.role === 'tool') {
        const content = await this.contentToPi(message.content, model)
        messages.push({
          role: 'toolResult',
          toolCallId: message.toolCallId,
          toolName: toolNames.get(message.toolCallId) ?? 'unknown',
          content: typeof content === 'string'
            ? [{ type: 'text', text: content || '(no output)' }]
            : content.length > 0 ? content : [{ type: 'text', text: '(no output)' }],
          isError: message.isError ?? false,
          timestamp: 0,
        })
        continue
      }
      const content = await this.contentToPi(message.content, model)
      messages.push({
        role: 'user',
        content: typeof content === 'string' && content.length === 0 ? '(empty message)' : content,
        timestamp: 0,
      })
    }

    const systemPrompt = systemParts.join('\n\n').trim()
    const tools = options.tools?.map((tool) => ({
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters,
    }))
    return {
      ...systemPrompt ? { systemPrompt } : {},
      messages,
      ...tools && tools.length > 0 ? { tools: tools as PiContext['tools'] } : {},
    }
  }

  private toPiAssistant(
    message: Extract<GenerateOptions['messages'][number], { role: 'assistant' }>,
    model: Model<Api>,
    toolNames: Map<string, string>,
  ): PiAssistantMessage {
    const content: PiAssistantMessage['content'] = []
    for (const block of message.content) {
      switch (block.type) {
        case 'text':
          if (block.text) content.push({ type: 'text', text: block.text })
          break
        case 'reasoning':
          if (block.text) content.push({ type: 'thinking', thinking: block.text })
          break
        case 'tool-call':
          toolNames.set(block.id, block.name)
          content.push({
            type: 'toolCall',
            id: block.id,
            name: block.name,
            arguments: parseArguments(block.arguments),
          })
          break
        default:
          break
      }
    }
    const stopReason = content.some((block) => block.type === 'toolCall') ? 'toolUse' : 'stop'
    return {
      role: 'assistant',
      content,
      api: model.api,
      provider: model.provider,
      model: model.id,
      usage: zeroUsage(),
      stopReason,
      timestamp: 0,
    }
  }

  private async contentToPi(
    blocks: readonly ContentBlock[] | string,
    model: Model<Api>,
  ): Promise<string | (TextContent | ImageContent)[]> {
    if (typeof blocks === 'string') return blocks
    const parts: (TextContent | ImageContent)[] = []
    let hasImage = false
    for (const block of blocks) {
      switch (block.type) {
        case 'text':
          if (block.text.length > 0) parts.push({ type: 'text', text: block.text })
          break
        case 'image': {
          if (block.offloaded === true) {
            parts.push({ type: 'text', text: this.offloadedImageText(block) })
            break
          }
          if (!model.input.includes('image')) {
            throw new LlmError(
              `model "${model.id}" does not accept image input (text-only modality)`,
              'UNSUPPORTED_CONTENT',
            )
          }
          parts.push(await this.loadImage(block))
          hasImage = true
          break
        }
        case 'file':
          parts.push({ type: 'text', text: this.fileHandleText(block.attachment) })
          break
        default:
          // Reasoning blocks are assistant-only and never reach this path.
          break
      }
    }
    if (!hasImage) return parts.map((part) => (part.type === 'text' ? part.text : '')).join('')
    return parts
  }

  private offloadedImageText(block: ImageBlock): string {
    const name = block.attachment.name ?? block.attachment.attachmentId
    return `[image omitted from this request: ${name}]`
  }

  private fileHandleText(ref: FileRefLike): string {
    try {
      const text = this.resolveFileText?.(ref)
      if (text && text.length > 0) return text
    } catch {
      // Fall through to the descriptive placeholder.
    }
    return `[file: ${ref.name}, ${ref.bytes} bytes]`
  }

  private async loadImage(block: ImageBlock): Promise<ImageContent> {
    const store = this.resolveAttachments?.()
    if (!store) {
      throw new LlmError('image input requires the durable attachment service', 'UNSUPPORTED_CONTENT')
    }
    try {
      const request = await store.readImageRequest(block.attachment, {
        width: block.attachment.width,
        height: block.attachment.height,
        maxBytes: MAX_IMAGE_BYTES,
      })
      return {
        type: 'image',
        data: Buffer.from(request.data).toString('base64'),
        mimeType: request.mediaType,
      }
    } catch (error) {
      throw new LlmError(`unable to read image attachment: ${errorText(error)}`, 'UNSUPPORTED_CONTENT', { cause: error })
    }
  }
}

/** Kilo's keyless gateway: any bearer is accepted; "anonymous" is its guest token. */
export class KiloFreeAdapter extends BaseFreeAdapter {
  constructor(options: FreeAdapterOptions) {
    super({ ...options, defaultKey: options.defaultKey || 'anonymous' })
  }

  protected override compatFor(_detail: ModelRecord | undefined, _api: Api): Model<Api>['compat'] {
    return {
      supportsDeveloperRole: false,
      supportsStrictMode: false,
      supportsStore: false,
      maxTokensField: 'max_tokens',
      thinkingFormat: 'openrouter',
      supportsReasoningEffort: true,
      supportsLongCacheRetention: false,
    } as Model<Api>['compat']
  }

  protected override buildHeaders(_options: GenerateOptions): Record<string, string> {
    return attributionHeaders()
  }
}

/** OpenCode Zen: free routes are opt-in because the gateway now gates its free tier. */
export class ZenFreeAdapter extends BaseFreeAdapter {
  private static readonly RESPONSES_MODEL_IDS = new Set(['muse-spark-1.2-contributor-free'])

  constructor(options: FreeAdapterOptions) {
    super({ ...options, defaultKey: options.defaultKey || 'public' })
  }

  private static zenModelApi(modelId: string): 'openai-completions' | 'openai-responses' {
    const id = modelId.trim().toLowerCase()
    if (ZenFreeAdapter.RESPONSES_MODEL_IDS.has(id)) return 'openai-responses'
    return /(?:^|[-_:])responses(?:[-_:]|$)/.test(id) ? 'openai-responses' : 'openai-completions'
  }

  protected override resolveApiId(detail: ModelRecord | undefined): Api {
    return detail ? ZenFreeAdapter.zenModelApi(detail.id) : 'openai-completions'
  }

  protected override compatFor(_detail: ModelRecord | undefined, api: Api): Model<Api>['compat'] {
    if (api === 'openai-responses') {
      return {
        supportsDeveloperRole: false,
        supportsStrictMode: false,
        supportsLongCacheRetention: false,
      } as Model<Api>['compat']
    }
    return {
      supportsDeveloperRole: false,
      supportsStrictMode: false,
      supportsStore: false,
      maxTokensField: 'max_tokens',
      thinkingFormat: 'openai',
      supportsReasoningEffort: true,
      supportsLongCacheRetention: false,
    } as Model<Api>['compat']
  }

  private static canonicalSessionId(seed: string): string {
    const value = seed.trim()
    if (/^ses_[0-9a-f]{12}[0-9A-Za-z]{14}$/.test(value)) return value
    const digest = createHash('sha256').update(value).digest()
    let tail = ''
    const alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    for (const byte of digest.subarray(6, 20)) tail += alphabet[byte % alphabet.length]
    return `ses_${digest.subarray(0, 6).toString('hex')}${tail}`
  }

  protected override buildHeaders(options: GenerateOptions): Record<string, string> {
    const session = ZenFreeAdapter.canonicalSessionId(sessionSeed(options))
    return {
      'user-agent': 'opencode/1.18.35 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.14',
      'x-opencode-client': 'cli',
      'x-opencode-session': session,
      'x-opencode-session-id': session,
      'x-session-affinity': session,
      'X-Session-Id': session,
      'x-opencode-project': 'dsh-my-free-adapter',
      'authorization': `Bearer ${this.apiKey}`,
    }
  }
}
