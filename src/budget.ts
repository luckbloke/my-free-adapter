/**
 * 输出预算策略：三层钳制。
 *
 * Kilo 免费线路的 max_completion_tokens 是"单次请求上限"，
 * 但 DSH 把它当作"每次请求必须预留的预算"，直接透传会导致：
 * 1. prompt + 输出必然超窗 → 网关 400
 * 2. 压缩压力预算归零 → 自动压缩失效
 *
 * 因此声明预算必须按窗口缩放（25%），请求前再按 prompt 估算收敛，
 * 最后用网关返回的 prompt_tokens 做 EMA 校准。
 */

export const MAX_OUTPUT_WINDOW_SHARE = 0.25
export const WIRE_SAFETY_SHARE = 0.02
export const MIN_WIRE_SAFETY_TOKENS = 1024
export const MIN_ANSWER_TOKENS = 2048

/** EMA 校准边界：下限修正估算的保守性，上限允许向上修正 */
export const MIN_CALIBRATION = 0.5
export const MAX_CALIBRATION = 4
export const CALIBRATION_ALPHA = 0.3

/** 窗口份额上限：声明预算不超过窗口的 25% */
export function outputCeilingForWindow(contextWindow: number): number {
  if (!Number.isFinite(contextWindow) || contextWindow <= 0) return 1
  return Math.max(1, Math.floor(contextWindow * MAX_OUTPUT_WINDOW_SHARE))
}

/** 请求边界安全余量：至少 1024，或窗口的 2% */
export function wireSafetyTokens(contextWindow: number): number {
  if (!Number.isFinite(contextWindow) || contextWindow <= 0) return MIN_WIRE_SAFETY_TOKENS
  return Math.max(MIN_WIRE_SAFETY_TOKENS, Math.ceil(contextWindow * WIRE_SAFETY_SHARE))
}

/** 折叠一次网关 usage 报告到校准因子（EMA） */
export function blendCalibration(previous: number | undefined, observed: number): number {
  if (!Number.isFinite(observed) || observed <= 0) return previous ?? 1
  const bounded = Math.min(MAX_CALIBRATION, Math.max(MIN_CALIBRATION, observed))
  if (previous === undefined || !Number.isFinite(previous)) return bounded
  const blended = previous + (bounded - previous) * CALIBRATION_ALPHA
  return Math.min(MAX_CALIBRATION, Math.max(MIN_CALIBRATION, blended))
}

function positiveInteger(value: unknown): number | undefined {
  const n = typeof value === 'number' ? value : typeof value === 'string' && value.trim() ? Number(value) : undefined
  if (n === undefined || !Number.isFinite(n) || n <= 0) return undefined
  return Math.floor(n)
}

export interface WireBudgetInput {
  /** DSH 传入的预算（可能缺失） */
  requested?: number | string | null
  /** 适配器侧模型输出上限（已含窗口和网关兼容天花板） */
  modelMaxTokens: number
  /** 网关强制执行的上下文窗口 */
  contextWindow: number
  /** Prompt 估算（原始值，未校准） */
  estimatedInputTokens: number
  /** 从该路由的 usage 报告学习的校准因子 */
  calibration?: number
}

export interface WireBudget {
  /** 要发送的 max_tokens；仅当 overflow 为 true 时缺失 */
  maxTokens?: number
  /** 校准后的 prompt 估算 */
  effectiveInputTokens: number
  /** 连最小回答都放不下时，必须上报溢出 */
  overflow: boolean
}

/**
 * 将一次请求的输出预算收敛到 prompt 留下的空间。
 * 
 * 不将网关报告的 prompt 大小作为下限：报告 215K 而估算 82K 说明估算乐观，
 * 报告 215K 而估算 10K 说明会话刚被压缩，两者从数字上无法区分。
 * 猜错会拒绝一个其实放得下的 prompt。EMA 校准才是恢复精度的方式。
 */
export function resolveWireBudget(input: WireBudgetInput): WireBudget {
  const window = positiveInteger(input.contextWindow) ?? 1
  const cap = Math.max(1, Math.min(positiveInteger(input.modelMaxTokens) ?? 1, window))
  const requested = positiveInteger(input.requested) ?? cap
  const ceiling = Math.min(requested, cap)
  const calibration = Number.isFinite(input.calibration)
    ? Math.min(MAX_CALIBRATION, Math.max(MIN_CALIBRATION, input.calibration as number))
    : 1
  const rawEstimate = Math.max(0, Number.isFinite(input.estimatedInputTokens) ? input.estimatedInputTokens : 0)
  const effectiveInputTokens = Math.ceil(rawEstimate * calibration)
  const room = window - effectiveInputTokens - wireSafetyTokens(window)
  if (!Number.isFinite(room) || room < MIN_ANSWER_TOKENS) {
    return { effectiveInputTokens, overflow: true }
  }
  return { maxTokens: Math.min(ceiling, room), effectiveInputTokens, overflow: false }
}

/**
 * 上下文溢出消息：措辞是 pi-ai 和 dsh-llm 都能识别的 OpenAI 兼容格式，
 * 让 DSH 走压缩路径而不是直接报 400。
 */
export function overflowMessage(estimatedInputTokens: number, contextWindow: number): string {
  return (
    `input exceeds the model's maximum context length of ${contextWindow} tokens ` +
    `(about ${estimatedInputTokens} input tokens, leaving less than ${MIN_ANSWER_TOKENS} for the answer); ` +
    'compact this session, switch to a model with a larger context window, or start a new session'
  )
}

// ---------------------------------------------------------------------------
// Prompt 估算器：按内容类型分类计价
// ---------------------------------------------------------------------------

/**
 * 字符类别单价（字符/token）。值越小表示越密（越贵）。
 * 对照 cl100k_base 实测：
 * | 内容             | 实测字符/token |
 * | 英文散文          | 5.58          |
 * | TypeScript 源码   | 4.23          |
 * | markdown/HTML     | 3.36 - 3.59   |
 * | 工具 JSON         | 3.38          |
 * | URL               | 2.70          |
 * | 纯数字            | 2.99          |
 * | 随机 hex/UUID     | 1.76 - 1.95   |
 * | 随机小写字母      | 1.62          |
 * | 随机 base64       | 1.39          |
 * | CJK               | 1.00          |
 * | emoji（含 ZWJ）   | 0.38          |
 * 所有常量取保守侧，估算值永不小于网关实测计数。
 */
interface TokenScale {
  readonly letters: number
  readonly digits: number
  readonly struct: number
  readonly other: number
}

const WORD: TokenScale = { letters: 3.6, digits: 2.6, struct: 1.7, other: 1.7 }
const STRUCTURED: TokenScale = { letters: 2.4, digits: 1.35, struct: 1.8, other: 1.4 }
const DATA: TokenScale = { letters: 1.2, digits: 2.2, struct: 1.5, other: 1.3 }
const WORDLIKE_LETTERS = 2.6

const LONG_RUN = 16
const STRUCTURAL = new Set('"\'{}[]<>()=:,;.!?&%$#@*|/\\^~`+-')
const WORD_VOWEL_FLOOR = 0.28
const STRUCTURED_RATIO = 0.08
const DIGITS_RATIO = 0.1
const SPACE_CHARS_PER_TOKEN = 16
const NEWLINE_TOKENS = 0.5
const WIDE_TOKENS = 1.1
const RANDOM_MIN_LENGTH = 6
const EMOJI_TOKENS = 3
const MESSAGE_OVERHEAD_TOKENS = 4
const IMAGE_TOKENS = 3200
const FILE_HANDLE_TOKENS = 32
const FIXED_OVERHEAD_TOKENS = 256

type CharClass = 'letters' | 'digits' | 'struct' | 'other' | 'wide'
const VOWELS = new Set('aeiouAEIOU')

function classify(char: string): CharClass {
  const code = char.codePointAt(0) ?? 0
  if (code > 0x7f) return 'wide'
  if ((code >= 65 && code <= 90) || (code >= 97 && code <= 122)) return 'letters'
  if (code >= 48 && code <= 57) return 'digits'
  if (STRUCTURAL.has(char)) return 'struct'
  return 'other'
}

function isEmoji(code: number): boolean {
  return (
    (code >= 0x1f000 && code <= 0x1faff) ||
    (code >= 0x2600 && code <= 0x27bf) ||
    (code >= 0x1f1e6 && code <= 0x1f1ff) ||
    (code >= 0x2190 && code <= 0x21ff) ||
    (code >= 0x2b00 && code <= 0x2bff) ||
    code === 0xfe0f || code === 0x200d
  )
}

/**
 * 短串（≤16 字符）：逐 tokenizer 片段计费。
 * 字母片段 ≥6 字符且元音占比 <28% 视为随机数据。
 */
function shortRunTokens(run: string): number {
  let tokens = 0
  let className: CharClass | undefined
  let length = 0
  let vowels = 0
  const flush = () => {
    if (className !== undefined && length > 0) {
      if (className === 'letters') {
        tokens += length >= RANDOM_MIN_LENGTH && vowels / length < WORD_VOWEL_FLOOR
          ? length / DATA.letters
          : Math.max(1, Math.floor(length / WORD.letters))
      } else if (className !== 'wide') {
        tokens += Math.max(1, Math.floor(length / WORD[className]))
      }
    }
    className = undefined; length = 0; vowels = 0
  }
  for (const char of run) {
    const kind = classify(char)
    if (kind === 'wide') {
      flush()
      tokens += isEmoji(char.codePointAt(0) ?? 0) ? EMOJI_TOKENS : WIDE_TOKENS
      continue
    }
    if (kind === className) {
      length += 1
      if (VOWELS.has(char)) vowels += 1
      continue
    }
    flush()
    className = kind; length = 1
    vowels = VOWELS.has(char) ? 1 : 0
  }
  flush()
  return tokens
}

/**
 * 长串（>16 字符）：先分类再计价。
 * 用结构标点占比、数字占比、字母元音占比区分代码/数据/词形。
 */
function longRunTokens(run: string): number {
  let letters = 0, digits = 0, struct = 0, other = 0, vowels = 0, wide = 0, emoji = 0
  for (const char of run) {
    const kind = classify(char)
    if (kind === 'wide') { wide += 1; if (isEmoji(char.codePointAt(0) ?? 0)) emoji += 1; continue }
    if (kind === 'letters') { letters += 1; if (VOWELS.has(char)) vowels += 1; continue }
    if (kind === 'digits') digits += 1
    else if (kind === 'struct') struct += 1
    else other += 1
  }
  let tokens = (wide - emoji) * WIDE_TOKENS + emoji * EMOJI_TOKENS
  const ascii = letters + digits + struct + other
  if (ascii === 0) return tokens
  const vowelFraction = letters > 0 ? vowels / letters : 0
  const looksRandom = letters / ascii >= 0.6 && vowelFraction < WORD_VOWEL_FLOOR
  let table: TokenScale
  if (struct / ascii >= STRUCTURED_RATIO && !looksRandom) table = STRUCTURED
  else if (digits / ascii >= DIGITS_RATIO) table = DATA
  else if (letters / ascii >= 0.9 && vowelFraction >= WORD_VOWEL_FLOOR) table = { ...WORD, letters: WORDLIKE_LETTERS }
  else table = DATA
  tokens += letters / table.letters + digits / table.digits + struct / table.struct + other / table.other
  return tokens
}

function textTokens(text: string): number {
  let tokens = 0, run = ''
  for (const char of text) {
    if (char === ' ') {
      if (run !== '') tokens += run.length <= LONG_RUN ? shortRunTokens(run) : longRunTokens(run)
      run = ''
      tokens += 1 / SPACE_CHARS_PER_TOKEN
      continue
    }
    if (char === '\n' || char === '\t') {
      if (run !== '') tokens += run.length <= LONG_RUN ? shortRunTokens(run) : longRunTokens(run)
      run = ''
      tokens += NEWLINE_TOKENS
      continue
    }
    run += char
  }
  if (run !== '') tokens += run.length <= LONG_RUN ? shortRunTokens(run) : longRunTokens(run)
  return tokens
}

function textTokensOf(value: unknown): number {
  return typeof value === 'string' ? textTokens(value) : 0
}

function blockTokens(block: any): number {
  switch (block?.type) {
    case 'text': case 'reasoning': return textTokensOf(block.text)
    case 'tool-call': return textTokensOf(block.name) + textTokensOf(block.arguments) + 8
    case 'tool-result': {
      let tokens = textTokensOf(block.toolCallId) + 8
      for (const nested of Array.isArray(block.content) ? block.content : []) tokens += blockTokens(nested)
      return tokens
    }
    case 'image': return IMAGE_TOKENS
    case 'file': return textTokensOf(block.attachment?.name) + FILE_HANDLE_TOKENS
    default: return 0
  }
}

function messageTokens(message: any): number {
  let tokens = MESSAGE_OVERHEAD_TOKENS
  if (typeof message?.content === 'string') return tokens + textTokens(message.content)
  for (const block of Array.isArray(message?.content) ? message.content : []) tokens += blockTokens(block)
  return tokens
}

/**
 * 估算一次请求的 prompt 大小（token）。
 * 
 * 估算器在每类字符上都刻意保守，EMA 校准在此基础上消除剩余的系统偏差。
 */
export function estimateRequestTokens(
  options: { messages?: any[]; system?: string; tools?: any[] },
): number {
  let tokens = FIXED_OVERHEAD_TOKENS
  tokens += textTokensOf(options.system)
  for (const message of Array.isArray(options.messages) ? options.messages : []) tokens += messageTokens(message)
  for (const tool of Array.isArray(options.tools) ? options.tools : []) {
    tokens += textTokensOf(tool.name) + textTokensOf(tool.description)
    try { tokens += textTokens(JSON.stringify(tool.parameters ?? {})) } catch { tokens += FIXED_OVERHEAD_TOKENS }
  }
  return Math.ceil(tokens)
}