export const KILO_GATEWAY_BASE_URL = 'https://api.kilo.ai/api/gateway'
export const KILO_GATEWAY_MAX_OUTPUT_TOKENS = 524_288

export interface KiloModel {
  id: string
  name?: string
  context_length?: number | string | null
  max_completion_tokens?: number | string | null
  top_provider?: { context_length?: number | string | null; max_completion_tokens?: number | string | null } | null
  limit?: Record<string, unknown> | null
  supported_parameters?: string[] | null
  isFree?: boolean
  is_free?: boolean
}

export interface KiloModelInfo {
  id: string
  name: string
  contextWindow: number
  maxTokens: number
  inputModalities: string[]
  reasoning: boolean
  supportsTools: boolean
}

const DEFAULT_CONTEXT_WINDOW = 262144
const DEFAULT_MAX_TOKENS = 32768

function positiveInteger(value: unknown): number | undefined {
  const n = typeof value === 'number' ? value : typeof value === 'string' ? Number(value) : NaN
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : undefined
}

function isFreeModel(model: KiloModel): boolean {
  if (model.isFree === false) return false
  if (model.is_free === false) return false
  if (model.isFree === true || model.is_free === true) return true
  const id = model.id.toLowerCase()
  return id === 'kilo-auto/free' || id === 'openrouter/free' || id.endsWith(':free') || id.endsWith('-free')
}

function supportsTools(model: KiloModel): boolean {
  const p = model.supported_parameters
  return !Array.isArray(p) || p.length === 0 || p.some((x) => String(x).toLowerCase() === 'tools')
}

function supportsReasoning(model: KiloModel): boolean {
  const p = model.supported_parameters
  if (!Array.isArray(p)) return false
  return p.some((x) => ['reasoning', 'include_reasoning', 'reasoning_effort'].includes(String(x).toLowerCase()))
}

export function modelInfo(model: KiloModel, gatewayMax: number | null = KILO_GATEWAY_MAX_OUTPUT_TOKENS): KiloModelInfo {
  const records: Record<string, unknown>[] = [model as unknown as Record<string, unknown>]
  for (const c of [model.top_provider, model.limit]) {
    if (c && typeof c === 'object' && !Array.isArray(c)) records.push(c as Record<string, unknown>)
  }
  const positiveLimits = (keys: string[]): number[] => {
    const out: number[] = []
    for (const r of records) for (const k of keys) {
      const v = positiveInteger(r[k])
      if (v !== undefined) out.push(v)
    }
    return out
  }
  const ctx = positiveLimits(['context_length', 'contextWindow', 'context_window', 'max_context_tokens'])
  const contextWindow = ctx.length ? Math.min(...ctx) : DEFAULT_CONTEXT_WINDOW
  const out = positiveLimits(['max_completion_tokens', 'maxTokens', 'max_tokens', 'max_output_tokens'])
  const advertised = out.length ? Math.min(...out) : DEFAULT_MAX_TOKENS
  const gateway = gatewayMax ?? Number.MAX_SAFE_INTEGER
  const maxTokens = Math.min(advertised, contextWindow, gateway, Math.max(1, Math.floor(contextWindow * 0.25)))
  return {
    id: model.id,
    name: model.name ?? model.id,
    contextWindow,
    maxTokens,
    inputModalities: ['text'],
    reasoning: supportsReasoning(model),
    supportsTools: supportsTools(model),
  }
}

export function decodeKiloModels(data: unknown): KiloModel[] {
  if (!data || typeof data !== 'object' || Array.isArray(data)) return []
  const rows = (data as { data?: unknown }).data
  if (!Array.isArray(rows)) return []
  const seen = new Set<string>()
  const out: KiloModel[] = []
  for (const row of rows) {
    if (!row || typeof row !== 'object') continue
    const m = row as KiloModel
    if (typeof m.id !== 'string' || !m.id.trim()) continue
    if (seen.has(m.id)) continue
    seen.add(m.id)
    out.push({ ...m, id: m.id.trim(), name: typeof m.name === 'string' ? m.name : m.id })
  }
  return out
}

export class ModelCatalog {
  #models = new Map<string, KiloModel>()
  #url: string
  #timer: NodeJS.Timeout | null = null
  #stopped = false

  constructor(url = `${KILO_GATEWAY_BASE_URL}/models`) {
    this.#url = url
  }

  async start(): Promise<void> {
    await this.refresh()
    this.#timer = setInterval(() => void this.refresh(), 300_000)
    this.#timer.unref?.()
  }

  stop(): void {
    this.#stopped = true
    if (this.#timer) clearInterval(this.#timer)
  }

  async refresh(): Promise<void> {
    try {
      const res = await fetch(this.#url, { headers: { accept: 'application/json' } })
      if (!res.ok) return
      const models = decodeKiloModels(await res.json())
      if (models.length === 0) return
      this.#models = new Map(models.map((m) => [m.id, m]))
    } catch {
      // keep previous snapshot
    }
  }

  list(): string[] {
    return [...this.#models.values()].filter((m) => isFreeModel(m) && supportsTools(m)).map((m) => m.id).sort()
  }

  get(id: string): KiloModel | undefined {
    return this.#models.get(id)
  }
}