export const MAX_OUTPUT_WINDOW_SHARE = 0.25
export const WIRE_SAFETY_SHARE = 0.02
export const MIN_WIRE_SAFETY_TOKENS = 1024
export const MIN_ANSWER_TOKENS = 2048
export const MIN_CALIBRATION = 0.5
export const MAX_CALIBRATION = 4
export const CALIBRATION_ALPHA = 0.3

export function outputCeilingForWindow(ctx: number): number {
  if (!Number.isFinite(ctx) || ctx <= 0) return 1
  return Math.max(1, Math.floor(ctx * MAX_OUTPUT_WINDOW_SHARE))
}

export function wireSafetyTokens(ctx: number): number {
  if (!Number.isFinite(ctx) || ctx <= 0) return MIN_WIRE_SAFETY_TOKENS
  return Math.max(MIN_WIRE_SAFETY_TOKENS, Math.ceil(ctx * WIRE_SAFETY_SHARE))
}

export function blendCalibration(prev: number | undefined, observed: number): number {
  if (!Number.isFinite(observed) || observed <= 0) return prev ?? 1
  const bounded = Math.min(MAX_CALIBRATION, Math.max(MIN_CALIBRATION, observed))
  if (prev === undefined || !Number.isFinite(prev)) return bounded
  const blended = prev + (bounded - prev) * CALIBRATION_ALPHA
  return Math.min(MAX_CALIBRATION, Math.max(MIN_CALIBRATION, blended))
}

function positiveInteger(v: unknown): number | undefined {
  const n = typeof v === 'number' ? v : typeof v === 'string' && v.trim() ? Number(v) : undefined
  if (n === undefined || !Number.isFinite(n) || n <= 0) return undefined
  return Math.floor(n)
}

export interface WireBudgetInput {
  requested?: number | string | null
  modelMaxTokens: number
  contextWindow: number
  estimatedInputTokens: number
  calibration?: number
}

export interface WireBudget {
  maxTokens?: number
  effectiveInputTokens: number
  overflow: boolean
}

export function resolveWireBudget(input: WireBudgetInput): WireBudget {
  const window = positiveInteger(input.contextWindow) ?? 1
  const cap = Math.max(1, Math.min(positiveInteger(input.modelMaxTokens) ?? 1, window))
  const requested = positiveInteger(input.requested) ?? cap
  const ceiling = Math.min(requested, cap)
  const calibration = Number.isFinite(input.calibration)
    ? Math.min(MAX_CALIBRATION, Math.max(MIN_CALIBRATION, input.calibration as number))
    : 1
  const raw = Math.max(0, Number.isFinite(input.estimatedInputTokens) ? input.estimatedInputTokens : 0)
  const effectiveInputTokens = Math.ceil(raw * calibration)
  const room = window - effectiveInputTokens - wireSafetyTokens(window)
  if (!Number.isFinite(room) || room < MIN_ANSWER_TOKENS) return { effectiveInputTokens, overflow: true }
  return { maxTokens: Math.min(ceiling, room), effectiveInputTokens, overflow: false }
}

export function overflowMessage(estimated: number, ctx: number): string {
  return `input exceeds the model's maximum context length of ${ctx} tokens (about ${estimated} input tokens, leaving less than ${MIN_ANSWER_TOKENS} for the answer); compact this session, switch to a model with a larger context window, or start a new session`
}

// 简化的 prompt 估算：按字符数除以 3 取上界
export function estimateRequestTokens(messages: Array<{ content: unknown }>, system?: string): number {
  let text = system ?? ''
  for (const m of messages) {
    const c = m.content
    text += typeof c === 'string' ? c : JSON.stringify(c ?? '')
  }
  return Math.ceil(text.length / 3)
}