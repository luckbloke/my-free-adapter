import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname } from 'node:path'
import { outputCeilingForWindow } from './budget.ts'

export const KILO_GATEWAY_BASE_URL = 'https://api.kilo.ai/api/gateway'
export const OPENCODE_ZEN_BASE_URL = 'https://opencode.ai/zen/v1'
export const KILO_GATEWAY_MAX_OUTPUT_TOKENS = 524_288
export const OPENCODE_CAPABILITIES_URL = 'https://models.opencode.ai/api.json'

const CACHE_TTL_MS = 7 * 24 * 60 * 60 * 1000
const CAPABILITY_TTL_MS = 6 * 60 * 60 * 1000
const DEFAULT_FETCH_TIMEOUT_MS = 30_000
const DEFAULT_READY_TIMEOUT_MS = 4_000

export interface ModelRecord {
  id: string
  name?: string
  context_length?: number | string | null
  contextWindow?: number | string | null
  context_window?: number | string | null
  max_context_tokens?: number | string | null
  max_completion_tokens?: number | string | null
  maxTokens?: number | string | null
  max_tokens?: number | string | null
  max_output_tokens?: number | string | null
  top_provider?: Record<string, unknown> | null
  limit?: Record<string, unknown> | null
  limits?: Record<string, unknown> | null
  supported_parameters?: string[] | null
  isFree?: boolean
  is_free?: boolean
  deprecated?: boolean
  pricing?: { prompt?: string | number; completion?: string | number } | null
  architecture?: { input_modalities?: string[] | null; output_modalities?: string[] | null } | null
  [key: string]: unknown
}

export interface ModelInfo {
  id: string
  name: string
  contextWindow: number
  maxTokens: number
  inputModalities: string[]
  reasoning: boolean
  reasoningEfforts?: string[]
  supportsTools: boolean
}

const DEFAULT_CONTEXT_WINDOW = 262144
const DEFAULT_MAX_TOKENS = 32768
const REASONING_EFFORTS = ['low', 'medium', 'high'] as const

function positiveInteger(value: unknown): number | undefined {
  const n = typeof value === 'number' ? value : typeof value === 'string' && value.trim() ? Number(value) : NaN
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : undefined
}

function explicitBoolean(value: unknown): boolean | undefined {
  if (typeof value === 'boolean') return value
  if (typeof value === 'number' && (value === 0 || value === 1)) return value === 1
  if (typeof value === 'string') {
    const s = value.trim().toLowerCase()
    if (s === 'true' || s === '1' || s === 'yes') return true
    if (s === 'false' || s === '0' || s === 'no') return false
  }
  return undefined
}

/** A model whose prompt and completion prices are both exactly zero. */
function hasZeroPricing(model: ModelRecord): boolean {
  const pricing = model.pricing
  if (!pricing || typeof pricing !== 'object') return false
  if (pricing.prompt === undefined || pricing.completion === undefined) return false
  const prompt = Number(pricing.prompt)
  const completion = Number(pricing.completion)
  return Number.isFinite(prompt) && Number.isFinite(completion) && prompt === 0 && completion === 0
}

/**
 * Free-model predicate shared by both gateways. Explicit catalog flags win;
 * zero pricing covers gateways that only expose price, and the id conventions
 * cover catalogs that expose neither (OpenCode Zen).
 */
export function isFreeModel(model: ModelRecord): boolean {
  if (explicitBoolean(model.deprecated) === true) return false
  const explicit = explicitBoolean(model.isFree) ?? explicitBoolean(model.is_free)
  if (explicit !== undefined) return explicit
  if (hasZeroPricing(model)) return true
  const id = model.id.trim().toLowerCase()
  return (
    id === 'kilo-auto/free' ||
    id === 'openrouter/free' ||
    id === 'big-pickle' ||
    id.endsWith(':free') ||
    id.endsWith('-free') ||
    id.endsWith('/free')
  )
}

/** Reject image/audio-only output models: they cannot serve a chat turn. */
export function isTextOutputModel(model: ModelRecord): boolean {
  const output = model.architecture?.output_modalities
  if (!Array.isArray(output) || output.length === 0) return true
  return output.some((m) => String(m).trim().toLowerCase() === 'text')
}

function supportsTools(model: ModelRecord): boolean {
  const params = model.supported_parameters
  return !Array.isArray(params) || params.length === 0 || params.some((x) => String(x).toLowerCase() === 'tools')
}

function supportsReasoning(model: ModelRecord): boolean {
  const params = model.supported_parameters
  if (!Array.isArray(params)) return false
  return params.some((x) => ['reasoning', 'include_reasoning', 'reasoning_effort'].includes(String(x).toLowerCase()))
}

export function modelInfo(model: ModelRecord, outputCap = KILO_GATEWAY_MAX_OUTPUT_TOKENS): ModelInfo {
  const records: Record<string, unknown>[] = [model as unknown as Record<string, unknown>]
  for (const candidate of [model.top_provider, model.limit, model.limits]) {
    if (candidate && typeof candidate === 'object' && !Array.isArray(candidate)) records.push(candidate)
  }

  const positiveLimits = (keys: string[]): number[] => {
    const out: number[] = []
    for (const record of records) for (const key of keys) {
      const value = positiveInteger(record[key])
      if (value !== undefined) out.push(value)
    }
    return out
  }

  const contextLimits = positiveLimits(['context_length', 'contextWindow', 'context_window', 'max_context_tokens'])
  const contextWindow = contextLimits.length ? Math.min(...contextLimits) : DEFAULT_CONTEXT_WINDOW

  const outputLimits = positiveLimits(['max_completion_tokens', 'maxTokens', 'max_tokens', 'max_output_tokens'])
  const advertised = outputLimits.length ? Math.min(...outputLimits) : DEFAULT_MAX_TOKENS
  const maxTokens = Math.max(1, Math.min(
    advertised,
    contextWindow,
    positiveInteger(outputCap) ?? advertised,
    outputCeilingForWindow(contextWindow),
  ))

  const declaredInput = Array.isArray(model.architecture?.input_modalities)
    ? model.architecture.input_modalities.map(String).filter(Boolean) : ['text']
  if (!declaredInput.includes('text')) declaredInput.unshift('text')
  const inputModalities = declaredInput.filter((m) => m === 'text' || m === 'image')
  if (inputModalities.length === 0) inputModalities.push('text')

  const reasoning = supportsReasoning(model)

  return {
    id: model.id,
    name: model.name ?? model.id,
    contextWindow,
    maxTokens,
    inputModalities,
    reasoning,
    ...(reasoning ? { reasoningEfforts: [...REASONING_EFFORTS] } : {}),
    supportsTools: supportsTools(model),
  }
}

export function decodeModels(data: unknown): ModelRecord[] {
  if (!data || typeof data !== 'object' || Array.isArray(data)) return []
  const container = data as { data?: unknown; models?: unknown }
  const rows = Array.isArray(container.data) ? container.data
    : Array.isArray(container.models) ? container.models
    : []
  const seen = new Set<string>()
  const out: ModelRecord[] = []
  for (const row of rows) {
    if (!row || typeof row !== 'object') continue
    const model = row as ModelRecord
    if (typeof model.id !== 'string' || !model.id.trim()) continue
    const id = model.id.trim()
    if (seen.has(id)) continue
    seen.add(id)
    out.push({ ...model, id, name: typeof model.name === 'string' && model.name ? model.name : id })
  }
  return out
}

export interface OpenCodeCapability {
  context?: number
  output?: number
  reasoning: boolean
  toolCall: boolean
  inputModalities: string[]
}

export function parseOpenCodeCapabilities(data: unknown): Map<string, OpenCodeCapability> {
  const result = new Map<string, OpenCodeCapability>()
  if (!data || typeof data !== 'object' || Array.isArray(data)) return result

  for (const provider of Object.values(data as Record<string, unknown>)) {
    if (!provider || typeof provider !== 'object' || Array.isArray(provider)) continue
    const p = provider as Record<string, unknown>
    const api = typeof p.api === 'string' ? p.api.toLowerCase() : ''
    if (!api.includes('opencode.ai/zen')) continue

    const models = p.models
    if (!models || typeof models !== 'object' || Array.isArray(models)) continue

    for (const [key, raw] of Object.entries(models as Record<string, unknown>)) {
      if (!raw || typeof raw !== 'object' || Array.isArray(raw)) continue
      const r = raw as Record<string, unknown>
      const id = typeof r.id === 'string' && r.id.trim() ? r.id.trim() : key.trim()
      if (!id) continue

      const limit = (r.limit && typeof r.limit === 'object' && !Array.isArray(r.limit) ? r.limit : {}) as Record<string, unknown>
      const modalities = (r.modalities && typeof r.modalities === 'object' && !Array.isArray(r.modalities) ? r.modalities : {}) as Record<string, unknown>

      const context = positiveInteger(limit.context)
      const output = positiveInteger(limit.output)
      const inputList = Array.isArray(modalities.input)
        ? modalities.input.filter((x): x is string => typeof x === 'string') : []

      result.set(id, {
        ...(context !== undefined ? { context } : {}),
        ...(output !== undefined ? { output } : {}),
        reasoning: r.reasoning === true,
        toolCall: r.tool_call === true,
        inputModalities: inputList.length ? inputList : ['text'],
      })
    }
  }
  return result
}

export function enrichZenModels(models: ModelRecord[], capabilities: Map<string, OpenCodeCapability>): ModelRecord[] {
  if (capabilities.size === 0) return models
  return models.map((model) => {
    const cap = capabilities.get(model.id)
    if (!cap) return model
    const enriched = { ...model }
    const hasContext = enriched.context_length != null || enriched.contextWindow != null
      || enriched.context_window != null || enriched.max_context_tokens != null
    if (!hasContext && cap.context !== undefined) enriched.context_length = cap.context
    const hasOutput = enriched.max_completion_tokens != null || enriched.maxTokens != null
      || enriched.max_tokens != null || enriched.max_output_tokens != null
    if (!hasOutput && cap.output !== undefined) enriched.max_output_tokens = cap.output
    if (!Array.isArray(enriched.supported_parameters)) {
      const params: string[] = []
      if (cap.toolCall) params.push('tools')
      if (cap.reasoning) params.push('reasoning')
      if (params.length) enriched.supported_parameters = params
    }
    if (!Array.isArray(enriched.architecture?.input_modalities) && cap.inputModalities.length) {
      enriched.architecture = { ...(enriched.architecture ?? {}), input_modalities: cap.inputModalities }
    }
    return enriched
  })
}

export const KILO_STATIC_FALLBACK: ModelRecord[] = [
  {
    id: 'kilo-auto/free', name: 'Auto Free', isFree: true,
    context_length: 256_000, max_completion_tokens: 32_768,
    supported_parameters: ['tools', 'reasoning'],
  },
  {
    id: 'openrouter/free', name: 'OpenRouter Free', isFree: true,
    context_length: 200_000, max_completion_tokens: 32_768,
    supported_parameters: ['tools'],
  },
]

/** Current OpenCode Zen free routes (verified against the live catalog). */
export const ZEN_STATIC_FALLBACK: ModelRecord[] = [
  'big-pickle',
  'beam-free',
  'mimo-v2.6-flash-free',
  'nemotron-3.5-lightning-free',
  'nemotron-3.5-super-vl-free',
  'jev-1.13-free',
  'longcat-2.5-preview-free',
  'deepseek-v4.1-flash-free',
  'glm-5.3-flash-free',
  'space-bunny-free',
].map((id) => ({
  id,
  name: id,
  isFree: true,
  context_length: 262_144,
  max_completion_tokens: 32_768,
  supported_parameters: ['tools'],
  ...(id.includes('-vl-') ? { architecture: { input_modalities: ['text', 'image'], output_modalities: ['text'] } } : {}),
}))

export interface CatalogConfig {
  url: string
  cachePath?: string
  staticFallback?: ModelRecord[]
  freePredicate?: (model: ModelRecord) => boolean
  headers?: () => Record<string, string>
  /** Hard ceiling for a single request's output budget. */
  gatewayMaxOutputTokens?: number | null
  capabilitiesUrl?: string
  capabilityUserAgent?: string
  startupRetryMs?: number
  statusPath?: string
  label?: string
  fetchTimeoutMs?: number
  /** Observe a failed refresh for diagnostics; serving keeps the last good catalog. */
  onError?: (message: string) => void
}

type CatalogSource = 'empty' | 'cache' | 'network' | 'fallback'

interface CachePayload {
  updatedAt?: number
  models?: unknown
}

function errorText(error: unknown): string {
  if (error instanceof Error) return error.message
  return String(error)
}

/** Race a promise against a timer without rejecting; used to bound catalog readiness. */
async function withTimeout(promise: Promise<unknown>, timeoutMs: number): Promise<void> {
  if (timeoutMs <= 0) return
  let timer: NodeJS.Timeout | undefined
  try {
    await Promise.race([
      promise.then(() => undefined, () => undefined),
      new Promise<void>((resolve) => {
        timer = setTimeout(resolve, timeoutMs)
        timer.unref?.()
      }),
    ])
  } finally {
    if (timer) clearTimeout(timer)
  }
}

function timeoutSignal(timeoutMs: number, abort: AbortSignal): AbortSignal {
  const any = (AbortSignal as unknown as { any?: (signals: AbortSignal[]) => AbortSignal }).any
  const timeout = AbortSignal.timeout(Math.max(1, timeoutMs))
  return any ? any.call(AbortSignal, [abort, timeout]) : timeout
}

/**
 * One gateway's model catalog.
 *
 * The constructor starts loading the disk cache immediately; `start()` performs
 * the first network refresh and then polls. `ready()` gives request paths a
 * bounded wait for whichever of those finishes first, so model listing and
 * context resolution neither block forever nor return an empty list merely
 * because the cache read is still in flight.
 */
export class ModelCatalog {
  #models = new Map<string, ModelRecord>()
  #config: CatalogConfig
  #timer: NodeJS.Timeout | null = null
  #stopped = false
  #onRefresh?: (count: number) => void
  #lastError = ''
  #updatedAt = 0
  #capabilities: Map<string, OpenCodeCapability> | null = null
  #capabilitiesAt = 0
  #cacheLoaded: Promise<void>
  #firstRefresh: Promise<void> | null = null
  #startPromise: Promise<void> | null = null
  #refreshing: Promise<void> | null = null
  #abort = new AbortController()
  #source: CatalogSource = 'empty'

  constructor(config: CatalogConfig, onRefresh?: (count: number) => void) {
    this.#config = config
    this.#onRefresh = onRefresh
    this.#cacheLoaded = this.#loadCache()
  }

  async #loadCache(): Promise<void> {
    const path = this.#config.cachePath
    if (!path) return
    try {
      const raw = JSON.parse(await readFile(path, 'utf8')) as CachePayload
      // A network refresh that finished first is authoritative.
      if (this.#updatedAt > 0 || this.#stopped) return
      const updatedAt = typeof raw.updatedAt === 'number' && Number.isFinite(raw.updatedAt) ? raw.updatedAt : 0
      if (updatedAt <= 0 || Date.now() - updatedAt > CACHE_TTL_MS) return
      const models = decodeModels({ data: raw.models })
      if (models.length === 0) return
      this.#models = new Map(models.map((m) => [m.id, m]))
      this.#updatedAt = updatedAt
      this.#source = 'cache'
    } catch {
      // Missing or malformed cache is not an error; the network refresh follows.
    }
  }

  async #saveCache(models: ModelRecord[]): Promise<void> {
    const path = this.#config.cachePath
    if (!path) return
    try {
      await mkdir(dirname(path), { recursive: true })
      await writeFile(path, JSON.stringify({ updatedAt: Date.now(), models }), 'utf8')
    } catch {
      // A cache write failure must not break model serving.
    }
  }

  async #writeStatus(): Promise<void> {
    const path = this.#config.statusPath
    if (!path) return
    try {
      await mkdir(dirname(path), { recursive: true })
      const status = {
        label: this.#config.label ?? 'catalog',
        status: this.#updatedAt === 0 ? (this.#lastError ? 'error' : 'pending') : 'ready',
        source: this.#source,
        total: this.#models.size,
        exposed: this.list().length,
        lastRefresh: this.#updatedAt ? new Date(this.#updatedAt).toISOString() : undefined,
        lastError: this.#lastError || undefined,
        writtenAt: new Date().toISOString(),
      }
      await writeFile(path, JSON.stringify(status, null, 2), 'utf8')
    } catch {
      // Status is diagnostic only.
    }
  }

  async #capabilitiesWithin(timeoutMs: number): Promise<Map<string, OpenCodeCapability> | null> {
    let timer: NodeJS.Timeout | undefined
    try {
      return await Promise.race([
        this.#fetchCapabilities(),
        new Promise<null>((resolve) => {
          timer = setTimeout(() => resolve(null), timeoutMs)
          timer.unref?.()
        }),
      ])
    } finally {
      if (timer) clearTimeout(timer)
    }
  }

  async #fetchCapabilities(): Promise<Map<string, OpenCodeCapability> | null> {
    const url = this.#config.capabilitiesUrl
    if (!url) return null
    const now = Date.now()
    if (this.#capabilities && now - this.#capabilitiesAt < CAPABILITY_TTL_MS) return this.#capabilities
    try {
      const timeoutMs = Math.min(this.#config.fetchTimeoutMs ?? DEFAULT_FETCH_TIMEOUT_MS, 8_000)
      const signal = timeoutSignal(timeoutMs, this.#abort.signal)
      const res = await fetch(url, {
        headers: {
          accept: 'application/json',
          ...(this.#config.capabilityUserAgent ? { 'user-agent': this.#config.capabilityUserAgent } : {}),
        },
        signal,
      })
      if (!res.ok) return this.#capabilities
      const capabilities = parseOpenCodeCapabilities(await res.json())
      this.#capabilities = capabilities
      this.#capabilitiesAt = now
      return capabilities
    } catch {
      return this.#capabilities
    }
  }

  /** Load the cache, refresh once, retry while empty, then poll every five minutes. */
  async start(): Promise<void> {
    if (this.#startPromise) return this.#startPromise
    this.#startPromise = (async () => {
      await this.#cacheLoaded
      if (this.#stopped) return
      const first = this.refresh()
      this.#firstRefresh = first
      await first
      const retryMs = this.#config.startupRetryMs ?? 15_000
      let attempts = 0
      // Retry while the catalog comes from the cache or the static fallback:
      // the point of the startup loop is to reach a live network catalog.
      while (this.#source !== 'network' && attempts < 4 && !this.#stopped && retryMs > 0) {
        attempts += 1
        await new Promise((resolve) => {
          const timer = setTimeout(resolve, retryMs)
          timer.unref?.()
        })
        if (this.#stopped) return
        await this.refresh()
      }
      if (this.#stopped) return
      this.#timer = setInterval(() => void this.refresh(), 300_000)
      this.#timer.unref?.()
    })()
    return this.#startPromise
  }

  /** Bounded wait for the cache and the first network attempt. */
  async ready(timeoutMs = DEFAULT_READY_TIMEOUT_MS): Promise<void> {
    await withTimeout(this.#cacheLoaded, timeoutMs)
    if (this.#firstRefresh) await withTimeout(this.#firstRefresh, timeoutMs)
  }

  stop(): void {
    this.#stopped = true
    if (this.#timer) {
      clearInterval(this.#timer)
      this.#timer = null
    }
    this.#abort.abort(new Error('catalog stopped'))
  }

  async refresh(): Promise<void> {
    if (this.#stopped) return
    if (this.#refreshing) return this.#refreshing
    this.#refreshing = this.#doRefresh()
    try {
      await this.#refreshing
    } finally {
      this.#refreshing = null
    }
  }

  async #doRefresh(): Promise<void> {
    try {
      const headers = { accept: 'application/json', ...(this.#config.headers?.() ?? {}) }
      const signal = timeoutSignal(this.#config.fetchTimeoutMs ?? DEFAULT_FETCH_TIMEOUT_MS, this.#abort.signal)
      const res = await fetch(this.#config.url, { headers, signal })
      if (!res.ok) throw new Error(`HTTP ${res.status}${res.statusText ? ` ${res.statusText}` : ''}`)
      let models = decodeModels(await res.json())
      if (models.length === 0) throw new Error('empty model list')

      if (this.#config.capabilitiesUrl) {
        // Capability enrichment is a bonus: a slow metadata endpoint must not
        // hold the model list hostage. The fetch keeps running and caches its
        // result for the next refresh.
        const capabilities = await this.#capabilitiesWithin(2_500)
        if (capabilities) models = enrichZenModels(models, capabilities)
      }
      if (this.#stopped) return

      this.#models = new Map(models.map((m) => [m.id, m]))
      this.#updatedAt = Date.now()
      this.#lastError = ''
      this.#source = 'network'
      void this.#saveCache(models)
      void this.#writeStatus()
      this.#onRefresh?.(models.length)
    } catch (error) {
      if (this.#stopped) return
      this.#lastError = errorText(error)
      if (this.#models.size === 0 && this.#config.staticFallback) {
        this.#models = new Map(this.#config.staticFallback.map((m) => [m.id, m]))
        this.#source = 'fallback'
      }
      this.#config.onError?.(this.#lastError)
      void this.#writeStatus()
    }
  }

  list(): string[] {
    const predicate = this.#config.freePredicate ?? isFreeModel
    return [...this.#models.values()]
      .filter((model) => predicate(model) && supportsTools(model) && isTextOutputModel(model))
      .map((model) => model.id)
      .sort((a, b) => {
        // Put the gateway's own auto-router first: it is the safest default for
        // a new session, and a stable selector order matters for the GUI.
        const rank = (id: string): number => (id === 'kilo-auto/free' || id === 'openrouter/free' ? 0 : 1)
        return rank(a) - rank(b) || a.localeCompare(b)
      })
  }

  get(id: string): ModelRecord | undefined {
    return this.#models.get(id)
  }

  get lastError(): string {
    return this.#lastError
  }

  get source(): CatalogSource {
    return this.#source
  }

  get updatedAt(): number {
    return this.#updatedAt
  }
}
