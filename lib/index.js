import Schema from "@deepseek-ai/schemastery";
import { CONTEXT_WINDOW_EXCEEDED_CODE, EMPTY_RESPONSE_CODE, LlmAdapter, LlmError, attributionHeaders, isContextWindowExceededError, isQuotaExceededError } from "@deepseek-ai/dsh-llm";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { createProvider } from "@earendil-works/pi-ai";
import * as openaiCompletions from "@earendil-works/pi-ai/api/openai-completions";
import * as openaiResponses from "@earendil-works/pi-ai/api/openai-responses";

//#region src/budget.ts
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
const MAX_OUTPUT_WINDOW_SHARE = .25;
const WIRE_SAFETY_SHARE = .02;
const MIN_WIRE_SAFETY_TOKENS = 1024;
const MIN_ANSWER_TOKENS = 2048;
/** EMA 校准边界：下限修正估算的保守性，上限允许向上修正 */
const MIN_CALIBRATION = .5;
const MAX_CALIBRATION = 4;
const CALIBRATION_ALPHA = .3;
/** 窗口份额上限：声明预算不超过窗口的 25% */
function outputCeilingForWindow(contextWindow) {
	if (!Number.isFinite(contextWindow) || contextWindow <= 0) return 1;
	return Math.max(1, Math.floor(contextWindow * MAX_OUTPUT_WINDOW_SHARE));
}
/** 请求边界安全余量：至少 1024，或窗口的 2% */
function wireSafetyTokens(contextWindow) {
	if (!Number.isFinite(contextWindow) || contextWindow <= 0) return MIN_WIRE_SAFETY_TOKENS;
	return Math.max(MIN_WIRE_SAFETY_TOKENS, Math.ceil(contextWindow * WIRE_SAFETY_SHARE));
}
/** 折叠一次网关 usage 报告到校准因子（EMA） */
function blendCalibration(previous, observed) {
	if (!Number.isFinite(observed) || observed <= 0) return previous ?? 1;
	const bounded = Math.min(MAX_CALIBRATION, Math.max(MIN_CALIBRATION, observed));
	if (previous === void 0 || !Number.isFinite(previous)) return bounded;
	const blended = previous + (bounded - previous) * CALIBRATION_ALPHA;
	return Math.min(MAX_CALIBRATION, Math.max(MIN_CALIBRATION, blended));
}
function positiveInteger$1(value) {
	const n = typeof value === "number" ? value : typeof value === "string" && value.trim() ? Number(value) : void 0;
	if (n === void 0 || !Number.isFinite(n) || n <= 0) return void 0;
	return Math.floor(n);
}
/**
* 将一次请求的输出预算收敛到 prompt 留下的空间。
* 
* 不将网关报告的 prompt 大小作为下限：报告 215K 而估算 82K 说明估算乐观，
* 报告 215K 而估算 10K 说明会话刚被压缩，两者从数字上无法区分。
* 猜错会拒绝一个其实放得下的 prompt。EMA 校准才是恢复精度的方式。
*/
function resolveWireBudget(input) {
	const window = positiveInteger$1(input.contextWindow) ?? 1;
	const cap = Math.max(1, Math.min(positiveInteger$1(input.modelMaxTokens) ?? 1, window));
	const requested = positiveInteger$1(input.requested) ?? cap;
	const ceiling = Math.min(requested, cap);
	const calibration = Number.isFinite(input.calibration) ? Math.min(MAX_CALIBRATION, Math.max(MIN_CALIBRATION, input.calibration)) : 1;
	const rawEstimate = Math.max(0, Number.isFinite(input.estimatedInputTokens) ? input.estimatedInputTokens : 0);
	const effectiveInputTokens = Math.ceil(rawEstimate * calibration);
	const room = window - effectiveInputTokens - wireSafetyTokens(window);
	if (!Number.isFinite(room) || room < MIN_ANSWER_TOKENS) return {
		effectiveInputTokens,
		overflow: true
	};
	return {
		maxTokens: Math.min(ceiling, room),
		effectiveInputTokens,
		overflow: false
	};
}
/**
* 上下文溢出消息：措辞是 pi-ai 和 dsh-llm 都能识别的 OpenAI 兼容格式，
* 让 DSH 走压缩路径而不是直接报 400。
*/
function overflowMessage(estimatedInputTokens, contextWindow) {
	return `input exceeds the model's maximum context length of ${contextWindow} tokens (about ${estimatedInputTokens} input tokens, leaving less than ${MIN_ANSWER_TOKENS} for the answer); compact this session, switch to a model with a larger context window, or start a new session`;
}
const WORD = {
	letters: 3.6,
	digits: 2.6,
	struct: 1.7,
	other: 1.7
};
const STRUCTURED = {
	letters: 2.4,
	digits: 1.35,
	struct: 1.8,
	other: 1.4
};
const DATA = {
	letters: 1.2,
	digits: 2.2,
	struct: 1.5,
	other: 1.3
};
const WORDLIKE_LETTERS = 2.6;
const LONG_RUN = 16;
const STRUCTURAL = /* @__PURE__ */ new Set("\"'{}[]<>()=:,;.!?&%$#@*|/\\^~`+-");
const WORD_VOWEL_FLOOR = .28;
const STRUCTURED_RATIO = .08;
const DIGITS_RATIO = .1;
const SPACE_CHARS_PER_TOKEN = 16;
const NEWLINE_TOKENS = .5;
const WIDE_TOKENS = 1.1;
const RANDOM_MIN_LENGTH = 6;
const EMOJI_TOKENS = 3;
const MESSAGE_OVERHEAD_TOKENS = 4;
const IMAGE_TOKENS = 3200;
const FILE_HANDLE_TOKENS = 32;
const FIXED_OVERHEAD_TOKENS = 256;
const VOWELS = /* @__PURE__ */ new Set("aeiouAEIOU");
function classify(char) {
	const code = char.codePointAt(0) ?? 0;
	if (code > 127) return "wide";
	if (code >= 65 && code <= 90 || code >= 97 && code <= 122) return "letters";
	if (code >= 48 && code <= 57) return "digits";
	if (STRUCTURAL.has(char)) return "struct";
	return "other";
}
function isEmoji(code) {
	return code >= 126976 && code <= 129791 || code >= 9728 && code <= 10175 || code >= 127462 && code <= 127487 || code >= 8592 && code <= 8703 || code >= 11008 && code <= 11263 || code === 65039 || code === 8205;
}
/**
* 短串（≤16 字符）：逐 tokenizer 片段计费。
* 字母片段 ≥6 字符且元音占比 <28% 视为随机数据。
*/
function shortRunTokens(run) {
	let tokens = 0;
	let className;
	let length = 0;
	let vowels = 0;
	const flush = () => {
		if (className !== void 0 && length > 0) {
			if (className === "letters") tokens += length >= RANDOM_MIN_LENGTH && vowels / length < WORD_VOWEL_FLOOR ? length / DATA.letters : Math.max(1, Math.floor(length / WORD.letters));
			else if (className !== "wide") tokens += Math.max(1, Math.floor(length / WORD[className]));
		}
		className = void 0;
		length = 0;
		vowels = 0;
	};
	for (const char of run) {
		const kind = classify(char);
		if (kind === "wide") {
			flush();
			tokens += isEmoji(char.codePointAt(0) ?? 0) ? EMOJI_TOKENS : WIDE_TOKENS;
			continue;
		}
		if (kind === className) {
			length += 1;
			if (VOWELS.has(char)) vowels += 1;
			continue;
		}
		flush();
		className = kind;
		length = 1;
		vowels = VOWELS.has(char) ? 1 : 0;
	}
	flush();
	return tokens;
}
/**
* 长串（>16 字符）：先分类再计价。
* 用结构标点占比、数字占比、字母元音占比区分代码/数据/词形。
*/
function longRunTokens(run) {
	let letters = 0, digits = 0, struct = 0, other = 0, vowels = 0, wide = 0, emoji = 0;
	for (const char of run) {
		const kind = classify(char);
		if (kind === "wide") {
			wide += 1;
			if (isEmoji(char.codePointAt(0) ?? 0)) emoji += 1;
			continue;
		}
		if (kind === "letters") {
			letters += 1;
			if (VOWELS.has(char)) vowels += 1;
			continue;
		}
		if (kind === "digits") digits += 1;
		else if (kind === "struct") struct += 1;
		else other += 1;
	}
	let tokens = (wide - emoji) * WIDE_TOKENS + emoji * EMOJI_TOKENS;
	const ascii = letters + digits + struct + other;
	if (ascii === 0) return tokens;
	const vowelFraction = letters > 0 ? vowels / letters : 0;
	const looksRandom = letters / ascii >= .6 && vowelFraction < WORD_VOWEL_FLOOR;
	let table;
	if (struct / ascii >= STRUCTURED_RATIO && !looksRandom) table = STRUCTURED;
	else if (digits / ascii >= DIGITS_RATIO) table = DATA;
	else if (letters / ascii >= .9 && vowelFraction >= WORD_VOWEL_FLOOR) table = {
		...WORD,
		letters: WORDLIKE_LETTERS
	};
	else table = DATA;
	tokens += letters / table.letters + digits / table.digits + struct / table.struct + other / table.other;
	return tokens;
}
function textTokens(text) {
	let tokens = 0, run = "";
	for (const char of text) {
		if (char === " ") {
			if (run !== "") tokens += run.length <= LONG_RUN ? shortRunTokens(run) : longRunTokens(run);
			run = "";
			tokens += 1 / SPACE_CHARS_PER_TOKEN;
			continue;
		}
		if (char === "\n" || char === "	") {
			if (run !== "") tokens += run.length <= LONG_RUN ? shortRunTokens(run) : longRunTokens(run);
			run = "";
			tokens += NEWLINE_TOKENS;
			continue;
		}
		run += char;
	}
	if (run !== "") tokens += run.length <= LONG_RUN ? shortRunTokens(run) : longRunTokens(run);
	return tokens;
}
function textTokensOf(value) {
	return typeof value === "string" ? textTokens(value) : 0;
}
function blockTokens(block) {
	switch (block?.type) {
		case "text":
		case "reasoning": return textTokensOf(block.text);
		case "tool-call": return textTokensOf(block.name) + textTokensOf(block.arguments) + 8;
		case "tool-result": {
			let tokens = textTokensOf(block.toolCallId) + 8;
			for (const nested of Array.isArray(block.content) ? block.content : []) tokens += blockTokens(nested);
			return tokens;
		}
		case "image": return IMAGE_TOKENS;
		case "file": return textTokensOf(block.attachment?.name) + FILE_HANDLE_TOKENS;
		default: return 0;
	}
}
function messageTokens(message) {
	let tokens = MESSAGE_OVERHEAD_TOKENS;
	if (typeof message?.content === "string") return tokens + textTokens(message.content);
	for (const block of Array.isArray(message?.content) ? message.content : []) tokens += blockTokens(block);
	return tokens;
}
/**
* 估算一次请求的 prompt 大小（token）。
* 
* 估算器在每类字符上都刻意保守，EMA 校准在此基础上消除剩余的系统偏差。
*/
function estimateRequestTokens(options) {
	let tokens = FIXED_OVERHEAD_TOKENS;
	tokens += textTokensOf(options.system);
	for (const message of Array.isArray(options.messages) ? options.messages : []) tokens += messageTokens(message);
	for (const tool of Array.isArray(options.tools) ? options.tools : []) {
		tokens += textTokensOf(tool.name) + textTokensOf(tool.description);
		try {
			tokens += textTokens(JSON.stringify(tool.parameters ?? {}));
		} catch {
			tokens += FIXED_OVERHEAD_TOKENS;
		}
	}
	return Math.ceil(tokens);
}

//#endregion
//#region src/catalog.ts
const KILO_GATEWAY_BASE_URL = "https://api.kilo.ai/api/gateway";
const OPENCODE_ZEN_BASE_URL = "https://opencode.ai/zen/v1";
const KILO_GATEWAY_MAX_OUTPUT_TOKENS = 524288;
const OPENCODE_CAPABILITIES_URL = "https://models.opencode.ai/api.json";
const CACHE_TTL_MS = 10080 * 60 * 1e3;
const CAPABILITY_TTL_MS = 360 * 60 * 1e3;
const DEFAULT_FETCH_TIMEOUT_MS = 3e4;
const DEFAULT_READY_TIMEOUT_MS = 4e3;
const DEFAULT_CONTEXT_WINDOW$1 = 262144;
const DEFAULT_MAX_TOKENS$1 = 32768;
const REASONING_EFFORTS = [
	"low",
	"medium",
	"high"
];
function positiveInteger(value) {
	const n = typeof value === "number" ? value : typeof value === "string" && value.trim() ? Number(value) : NaN;
	return Number.isFinite(n) && n > 0 ? Math.floor(n) : void 0;
}
function explicitBoolean(value) {
	if (typeof value === "boolean") return value;
	if (typeof value === "number" && (value === 0 || value === 1)) return value === 1;
	if (typeof value === "string") {
		const s = value.trim().toLowerCase();
		if (s === "true" || s === "1" || s === "yes") return true;
		if (s === "false" || s === "0" || s === "no") return false;
	}
}
/** A model whose prompt and completion prices are both exactly zero. */
function hasZeroPricing(model) {
	const pricing = model.pricing;
	if (!pricing || typeof pricing !== "object") return false;
	if (pricing.prompt === void 0 || pricing.completion === void 0) return false;
	const prompt = Number(pricing.prompt);
	const completion = Number(pricing.completion);
	return Number.isFinite(prompt) && Number.isFinite(completion) && prompt === 0 && completion === 0;
}
/**
* Free-model predicate shared by both gateways. Explicit catalog flags win;
* zero pricing covers gateways that only expose price, and the id conventions
* cover catalogs that expose neither (OpenCode Zen).
*/
function isFreeModel(model) {
	if (explicitBoolean(model.deprecated) === true) return false;
	const explicit = explicitBoolean(model.isFree) ?? explicitBoolean(model.is_free);
	if (explicit !== void 0) return explicit;
	if (hasZeroPricing(model)) return true;
	const id = model.id.trim().toLowerCase();
	return id === "kilo-auto/free" || id === "openrouter/free" || id === "big-pickle" || id.endsWith(":free") || id.endsWith("-free") || id.endsWith("/free");
}
/** Reject image/audio-only output models: they cannot serve a chat turn. */
function isTextOutputModel(model) {
	const output = model.architecture?.output_modalities;
	if (!Array.isArray(output) || output.length === 0) return true;
	return output.some((m) => String(m).trim().toLowerCase() === "text");
}
function supportsTools(model) {
	const params = model.supported_parameters;
	return !Array.isArray(params) || params.length === 0 || params.some((x) => String(x).toLowerCase() === "tools");
}
function supportsReasoning(model) {
	const params = model.supported_parameters;
	if (!Array.isArray(params)) return false;
	return params.some((x) => [
		"reasoning",
		"include_reasoning",
		"reasoning_effort"
	].includes(String(x).toLowerCase()));
}
function modelInfo(model, outputCap$1 = KILO_GATEWAY_MAX_OUTPUT_TOKENS) {
	const records = [model];
	for (const candidate of [
		model.top_provider,
		model.limit,
		model.limits
	]) if (candidate && typeof candidate === "object" && !Array.isArray(candidate)) records.push(candidate);
	const positiveLimits = (keys) => {
		const out = [];
		for (const record of records) for (const key of keys) {
			const value = positiveInteger(record[key]);
			if (value !== void 0) out.push(value);
		}
		return out;
	};
	const contextLimits = positiveLimits([
		"context_length",
		"contextWindow",
		"context_window",
		"max_context_tokens"
	]);
	const contextWindow = contextLimits.length ? Math.min(...contextLimits) : DEFAULT_CONTEXT_WINDOW$1;
	const outputLimits = positiveLimits([
		"max_completion_tokens",
		"maxTokens",
		"max_tokens",
		"max_output_tokens"
	]);
	const advertised = outputLimits.length ? Math.min(...outputLimits) : DEFAULT_MAX_TOKENS$1;
	const maxTokens = Math.max(1, Math.min(advertised, contextWindow, positiveInteger(outputCap$1) ?? advertised, outputCeilingForWindow(contextWindow)));
	const declaredInput = Array.isArray(model.architecture?.input_modalities) ? model.architecture.input_modalities.map(String).filter(Boolean) : ["text"];
	if (!declaredInput.includes("text")) declaredInput.unshift("text");
	const inputModalities = declaredInput.filter((m) => m === "text" || m === "image");
	if (inputModalities.length === 0) inputModalities.push("text");
	const reasoning = supportsReasoning(model);
	return {
		id: model.id,
		name: model.name ?? model.id,
		contextWindow,
		maxTokens,
		inputModalities,
		reasoning,
		...reasoning ? { reasoningEfforts: [...REASONING_EFFORTS] } : {},
		supportsTools: supportsTools(model)
	};
}
function decodeModels(data) {
	if (!data || typeof data !== "object" || Array.isArray(data)) return [];
	const container = data;
	const rows = Array.isArray(container.data) ? container.data : Array.isArray(container.models) ? container.models : [];
	const seen = /* @__PURE__ */ new Set();
	const out = [];
	for (const row of rows) {
		if (!row || typeof row !== "object") continue;
		const model = row;
		if (typeof model.id !== "string" || !model.id.trim()) continue;
		const id = model.id.trim();
		if (seen.has(id)) continue;
		seen.add(id);
		out.push({
			...model,
			id,
			name: typeof model.name === "string" && model.name ? model.name : id
		});
	}
	return out;
}
function parseOpenCodeCapabilities(data) {
	const result = /* @__PURE__ */ new Map();
	if (!data || typeof data !== "object" || Array.isArray(data)) return result;
	for (const provider of Object.values(data)) {
		if (!provider || typeof provider !== "object" || Array.isArray(provider)) continue;
		const p = provider;
		if (!(typeof p.api === "string" ? p.api.toLowerCase() : "").includes("opencode.ai/zen")) continue;
		const models = p.models;
		if (!models || typeof models !== "object" || Array.isArray(models)) continue;
		for (const [key, raw] of Object.entries(models)) {
			if (!raw || typeof raw !== "object" || Array.isArray(raw)) continue;
			const r = raw;
			const id = typeof r.id === "string" && r.id.trim() ? r.id.trim() : key.trim();
			if (!id) continue;
			const limit = r.limit && typeof r.limit === "object" && !Array.isArray(r.limit) ? r.limit : {};
			const modalities = r.modalities && typeof r.modalities === "object" && !Array.isArray(r.modalities) ? r.modalities : {};
			const context = positiveInteger(limit.context);
			const output = positiveInteger(limit.output);
			const inputList = Array.isArray(modalities.input) ? modalities.input.filter((x) => typeof x === "string") : [];
			result.set(id, {
				...context !== void 0 ? { context } : {},
				...output !== void 0 ? { output } : {},
				reasoning: r.reasoning === true,
				toolCall: r.tool_call === true,
				inputModalities: inputList.length ? inputList : ["text"]
			});
		}
	}
	return result;
}
function enrichZenModels(models, capabilities) {
	if (capabilities.size === 0) return models;
	return models.map((model) => {
		const cap = capabilities.get(model.id);
		if (!cap) return model;
		const enriched = { ...model };
		if (!(enriched.context_length != null || enriched.contextWindow != null || enriched.context_window != null || enriched.max_context_tokens != null) && cap.context !== void 0) enriched.context_length = cap.context;
		if (!(enriched.max_completion_tokens != null || enriched.maxTokens != null || enriched.max_tokens != null || enriched.max_output_tokens != null) && cap.output !== void 0) enriched.max_output_tokens = cap.output;
		if (!Array.isArray(enriched.supported_parameters)) {
			const params = [];
			if (cap.toolCall) params.push("tools");
			if (cap.reasoning) params.push("reasoning");
			if (params.length) enriched.supported_parameters = params;
		}
		if (!Array.isArray(enriched.architecture?.input_modalities) && cap.inputModalities.length) enriched.architecture = {
			...enriched.architecture ?? {},
			input_modalities: cap.inputModalities
		};
		return enriched;
	});
}
const KILO_STATIC_FALLBACK = [{
	id: "kilo-auto/free",
	name: "Auto Free",
	isFree: true,
	context_length: 256e3,
	max_completion_tokens: 32768,
	supported_parameters: ["tools", "reasoning"]
}, {
	id: "openrouter/free",
	name: "OpenRouter Free",
	isFree: true,
	context_length: 2e5,
	max_completion_tokens: 32768,
	supported_parameters: ["tools"]
}];
/** Current OpenCode Zen free routes (verified against the live catalog). */
const ZEN_STATIC_FALLBACK = [
	"big-pickle",
	"beam-free",
	"mimo-v2.6-flash-free",
	"nemotron-3.5-lightning-free",
	"nemotron-3.5-super-vl-free",
	"jev-1.13-free",
	"longcat-2.5-preview-free",
	"deepseek-v4.1-flash-free",
	"glm-5.3-flash-free",
	"space-bunny-free"
].map((id) => ({
	id,
	name: id,
	isFree: true,
	context_length: 262144,
	max_completion_tokens: 32768,
	supported_parameters: ["tools"],
	...id.includes("-vl-") ? { architecture: {
		input_modalities: ["text", "image"],
		output_modalities: ["text"]
	} } : {}
}));
function errorText$1(error) {
	if (error instanceof Error) return error.message;
	return String(error);
}
/** Race a promise against a timer without rejecting; used to bound catalog readiness. */
async function withTimeout(promise, timeoutMs) {
	if (timeoutMs <= 0) return;
	let timer;
	try {
		await Promise.race([promise.then(() => void 0, () => void 0), new Promise((resolve) => {
			timer = setTimeout(resolve, timeoutMs);
			timer.unref?.();
		})]);
	} finally {
		if (timer) clearTimeout(timer);
	}
}
function timeoutSignal(timeoutMs, abort) {
	const any = AbortSignal.any;
	const timeout = AbortSignal.timeout(Math.max(1, timeoutMs));
	return any ? any.call(AbortSignal, [abort, timeout]) : timeout;
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
var ModelCatalog = class {
	#models = /* @__PURE__ */ new Map();
	#config;
	#timer = null;
	#stopped = false;
	#onRefresh;
	#lastError = "";
	#updatedAt = 0;
	#capabilities = null;
	#capabilitiesAt = 0;
	#cacheLoaded;
	#firstRefresh = null;
	#startPromise = null;
	#refreshing = null;
	#abort = new AbortController();
	#source = "empty";
	constructor(config, onRefresh) {
		this.#config = config;
		this.#onRefresh = onRefresh;
		this.#cacheLoaded = this.#loadCache();
	}
	async #loadCache() {
		const path = this.#config.cachePath;
		if (!path) return;
		try {
			const raw = JSON.parse(await readFile(path, "utf8"));
			if (this.#updatedAt > 0 || this.#stopped) return;
			const updatedAt = typeof raw.updatedAt === "number" && Number.isFinite(raw.updatedAt) ? raw.updatedAt : 0;
			if (updatedAt <= 0 || Date.now() - updatedAt > CACHE_TTL_MS) return;
			const models = decodeModels({ data: raw.models });
			if (models.length === 0) return;
			this.#models = new Map(models.map((m) => [m.id, m]));
			this.#updatedAt = updatedAt;
			this.#source = "cache";
		} catch {}
	}
	async #saveCache(models) {
		const path = this.#config.cachePath;
		if (!path) return;
		try {
			await mkdir(dirname(path), { recursive: true });
			await writeFile(path, JSON.stringify({
				updatedAt: Date.now(),
				models
			}), "utf8");
		} catch {}
	}
	async #writeStatus() {
		const path = this.#config.statusPath;
		if (!path) return;
		try {
			await mkdir(dirname(path), { recursive: true });
			const status = {
				label: this.#config.label ?? "catalog",
				status: this.#updatedAt === 0 ? this.#lastError ? "error" : "pending" : "ready",
				source: this.#source,
				total: this.#models.size,
				exposed: this.list().length,
				lastRefresh: this.#updatedAt ? new Date(this.#updatedAt).toISOString() : void 0,
				lastError: this.#lastError || void 0,
				writtenAt: (/* @__PURE__ */ new Date()).toISOString()
			};
			await writeFile(path, JSON.stringify(status, null, 2), "utf8");
		} catch {}
	}
	async #capabilitiesWithin(timeoutMs) {
		let timer;
		try {
			return await Promise.race([this.#fetchCapabilities(), new Promise((resolve) => {
				timer = setTimeout(() => resolve(null), timeoutMs);
				timer.unref?.();
			})]);
		} finally {
			if (timer) clearTimeout(timer);
		}
	}
	async #fetchCapabilities() {
		const url = this.#config.capabilitiesUrl;
		if (!url) return null;
		const now = Date.now();
		if (this.#capabilities && now - this.#capabilitiesAt < CAPABILITY_TTL_MS) return this.#capabilities;
		try {
			const signal = timeoutSignal(Math.min(this.#config.fetchTimeoutMs ?? DEFAULT_FETCH_TIMEOUT_MS, 8e3), this.#abort.signal);
			const res = await fetch(url, {
				headers: {
					accept: "application/json",
					...this.#config.capabilityUserAgent ? { "user-agent": this.#config.capabilityUserAgent } : {}
				},
				signal
			});
			if (!res.ok) return this.#capabilities;
			const capabilities = parseOpenCodeCapabilities(await res.json());
			this.#capabilities = capabilities;
			this.#capabilitiesAt = now;
			return capabilities;
		} catch {
			return this.#capabilities;
		}
	}
	/** Load the cache, refresh once, retry while empty, then poll every five minutes. */
	async start() {
		if (this.#startPromise) return this.#startPromise;
		this.#startPromise = (async () => {
			await this.#cacheLoaded;
			if (this.#stopped) return;
			const first = this.refresh();
			this.#firstRefresh = first;
			await first;
			const retryMs = this.#config.startupRetryMs ?? 15e3;
			let attempts = 0;
			while (this.#source !== "network" && attempts < 4 && !this.#stopped && retryMs > 0) {
				attempts += 1;
				await new Promise((resolve) => {
					setTimeout(resolve, retryMs).unref?.();
				});
				if (this.#stopped) return;
				await this.refresh();
			}
			if (this.#stopped) return;
			this.#timer = setInterval(() => void this.refresh(), 3e5);
			this.#timer.unref?.();
		})();
		return this.#startPromise;
	}
	/** Bounded wait for the cache and the first network attempt. */
	async ready(timeoutMs = DEFAULT_READY_TIMEOUT_MS) {
		await withTimeout(this.#cacheLoaded, timeoutMs);
		if (this.#firstRefresh) await withTimeout(this.#firstRefresh, timeoutMs);
	}
	stop() {
		this.#stopped = true;
		if (this.#timer) {
			clearInterval(this.#timer);
			this.#timer = null;
		}
		this.#abort.abort(/* @__PURE__ */ new Error("catalog stopped"));
	}
	async refresh() {
		if (this.#stopped) return;
		if (this.#refreshing) return this.#refreshing;
		this.#refreshing = this.#doRefresh();
		try {
			await this.#refreshing;
		} finally {
			this.#refreshing = null;
		}
	}
	async #doRefresh() {
		try {
			const headers = {
				accept: "application/json",
				...this.#config.headers?.() ?? {}
			};
			const signal = timeoutSignal(this.#config.fetchTimeoutMs ?? DEFAULT_FETCH_TIMEOUT_MS, this.#abort.signal);
			const res = await fetch(this.#config.url, {
				headers,
				signal
			});
			if (!res.ok) throw new Error(`HTTP ${res.status}${res.statusText ? ` ${res.statusText}` : ""}`);
			let models = decodeModels(await res.json());
			if (models.length === 0) throw new Error("empty model list");
			if (this.#config.capabilitiesUrl) {
				const capabilities = await this.#capabilitiesWithin(2500);
				if (capabilities) models = enrichZenModels(models, capabilities);
			}
			if (this.#stopped) return;
			this.#models = new Map(models.map((m) => [m.id, m]));
			this.#updatedAt = Date.now();
			this.#lastError = "";
			this.#source = "network";
			this.#saveCache(models);
			this.#writeStatus();
			this.#onRefresh?.(models.length);
		} catch (error) {
			if (this.#stopped) return;
			this.#lastError = errorText$1(error);
			if (this.#models.size === 0 && this.#config.staticFallback) {
				this.#models = new Map(this.#config.staticFallback.map((m) => [m.id, m]));
				this.#source = "fallback";
			}
			this.#config.onError?.(this.#lastError);
			this.#writeStatus();
		}
	}
	list() {
		const predicate = this.#config.freePredicate ?? isFreeModel;
		return [...this.#models.values()].filter((model) => predicate(model) && supportsTools(model) && isTextOutputModel(model)).map((model) => model.id).sort((a, b) => {
			const rank = (id) => id === "kilo-auto/free" || id === "openrouter/free" ? 0 : 1;
			return rank(a) - rank(b) || a.localeCompare(b);
		});
	}
	get(id) {
		return this.#models.get(id);
	}
	get lastError() {
		return this.#lastError;
	}
	get source() {
		return this.#source;
	}
	get updatedAt() {
		return this.#updatedAt;
	}
};

//#endregion
//#region src/adapter.ts
const STREAM_IDLE_TIMEOUT_MS = 12e4;
const MAX_IMAGE_BYTES = 5 * 1024 * 1024;
const CATALOG_READY_TIMEOUT_MS = 2e3;
const DEFAULT_CONTEXT_WINDOW = 262144;
const DEFAULT_MAX_TOKENS = 32768;
const EFFORT_LABELS = {
	minimal: "Minimal",
	low: "Low",
	medium: "Medium",
	high: "High",
	xhigh: "XHigh",
	max: "Max"
};
function numeric(value, fallback) {
	if (typeof value === "number" && Number.isFinite(value) && value > 0) return value;
	if (typeof value === "string" && value.trim()) {
		const parsed = Number(value);
		if (Number.isFinite(parsed) && parsed > 0) return parsed;
	}
	return fallback;
}
function errorText(error) {
	return error instanceof Error ? error.message : String(error);
}
function toModalities(values) {
	const modalities = [];
	for (const value of values) if ((value === "text" || value === "image") && !modalities.includes(value)) modalities.push(value);
	return modalities.length > 0 ? modalities : ["text"];
}
function flattenText(blocks) {
	if (typeof blocks === "string") return blocks;
	if (!Array.isArray(blocks)) return "";
	return blocks.filter((block) => block.type === "text").map((block) => block.text).join("");
}
function contentHasImage(blocks) {
	if (!Array.isArray(blocks)) return false;
	return blocks.some((block) => block.type === "image" && block.offloaded !== true);
}
function parseArguments(raw) {
	try {
		const parsed = JSON.parse(raw || "{}");
		return parsed !== null && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
	} catch {
		return {};
	}
}
function zeroUsage() {
	return {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 0,
		cost: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			total: 0
		}
	};
}
function mapUsage(usage) {
	return {
		inputTokens: Number.isFinite(usage.input) ? usage.input : 0,
		outputTokens: Number.isFinite(usage.output) ? usage.output : 0,
		...Number.isFinite(usage.totalTokens) ? { totalTokens: usage.totalTokens } : {},
		...usage.cacheRead > 0 ? { cacheReadTokens: usage.cacheRead } : {},
		...usage.cacheWrite > 0 ? { cacheWriteTokens: usage.cacheWrite } : {},
		...typeof usage.reasoning === "number" && usage.reasoning > 0 ? { reasoningTokens: usage.reasoning } : {}
	};
}
/** Classify provider text into Harness failure codes the retry layer understands. */
function classifyError(message) {
	if (/FreeTierError|free tier can only be used from within OpenCode/i.test(message)) return "FREE_TIER_GATED";
	if (isQuotaExceededError(message)) return "QUOTA";
	if (/\b(?:401|403)\b|unauthori[sz]ed|forbidden/i.test(message)) return "AUTH";
	if (/\b429\b|rate.?limit/i.test(message)) return "RATE_LIMIT";
	if (/\b413\b|failed to buffer the request body|payload too large|request body too large/i.test(message)) return "INVALID_REQUEST";
	if (/\b400\b|invalid.?request/i.test(message)) return "INVALID_REQUEST";
	if (/\b5\d\d\b/.test(message)) return "SERVER";
	if (/\btime(?:d)?\s*out\b|timeout/i.test(message)) return "TIMEOUT";
	if (/stream ended (?:before|without)\b/i.test(message)) return "TRANSPORT";
	if (/\b(?:network|connection|socket|fetch)\b|\bECONN[A-Z]+\b|other side closed|terminated|premature close/i.test(message)) return "TRANSPORT";
	return "UPSTREAM";
}
function mapStopReason(message, contextWindow) {
	const detail = message.errorMessage ?? "";
	if (message.stopReason === "error" && isContextWindowExceededError(detail)) return {
		kind: "error",
		failure: {
			message: detail,
			code: CONTEXT_WINDOW_EXCEEDED_CODE
		}
	};
	switch (message.stopReason) {
		case "stop":
			if (message.content.length === 0) return {
				kind: "error",
				failure: {
					message: `model "${message.model}" completed without any content`,
					code: EMPTY_RESPONSE_CODE
				}
			};
			return { kind: "stop" };
		case "length": {
			const used = (message.usage?.input ?? 0) + (message.usage?.output ?? 0);
			if (contextWindow > 0 && used >= contextWindow) return {
				kind: "error",
				failure: {
					message: `output reached the model context window (${used}/${contextWindow} tokens)`,
					code: CONTEXT_WINDOW_EXCEEDED_CODE
				}
			};
			return { kind: "max-tokens" };
		}
		case "toolUse": return { kind: "tool-calls" };
		case "aborted": return {
			kind: "aborted",
			failure: {
				message: detail || "pi-ai stream aborted",
				code: "ABORTED"
			}
		};
		case "error": return {
			kind: "error",
			failure: {
				message: detail || "pi-ai stream error",
				code: classifyError(detail)
			}
		};
		default: return {
			kind: "error",
			failure: {
				message: detail || "pi-ai stream error",
				code: "UPSTREAM"
			}
		};
	}
}
/** `AbortSignal.any` without requiring it in the TypeScript lib. */
function anySignal(signals) {
	const any = AbortSignal.any;
	return any ? any.call(AbortSignal, signals) : signals[0] ?? new AbortController().signal;
}
/** Stable, non-reversible session identity; raw prompt text never reaches a header. */
function sessionSeed(options) {
	if (options.sessionId) return String(options.sessionId);
	for (const message of options.messages) {
		if (message.role !== "user") continue;
		const text = flattenText(message.content);
		if (text) return createHash("sha256").update(text).digest("hex");
	}
	return "my-free-adapter-default-session";
}
function injectStop(payload, stop) {
	if (payload !== null && typeof payload === "object") payload.stop = [...stop];
	return payload;
}
var BaseFreeAdapter = class extends LlmAdapter {
	catalog;
	providerId;
	baseUrl;
	apiKey;
	providerName;
	outputCap;
	provider;
	resolveAttachments;
	resolveFileText;
	#calibration = /* @__PURE__ */ new Map();
	constructor(options) {
		super();
		this.catalog = options.catalog;
		this.providerId = options.providerId;
		this.baseUrl = options.baseUrl.replace(/\/+$/, "");
		this.providerName = options.providerName;
		this.apiKey = options.apiKey?.trim() || options.defaultKey;
		this.outputCap = numeric(options.maxOutputTokens, DEFAULT_MAX_TOKENS);
		this.resolveAttachments = options.resolveAttachments;
		this.resolveFileText = options.resolveFileText;
		this.provider = createProvider({
			id: options.providerId,
			name: options.providerName,
			baseUrl: this.baseUrl,
			auth: { apiKey: {
				name: `${options.providerName} credential`,
				resolve: async () => ({ auth: { apiKey: this.apiKey } })
			} },
			models: [],
			api: {
				"openai-completions": openaiCompletions,
				"openai-responses": openaiResponses
			}
		});
	}
	/** Wire protocol for one catalog record (Zen routes some models to Responses). */
	resolveApiId(_detail) {
		return "openai-completions";
	}
	/** pi-ai compatibility overrides for this gateway dialect. */
	compatFor(_detail, _api) {}
	providerInfo(provider) {
		return {
			id: provider,
			name: this.providerName
		};
	}
	async listModels(provider) {
		await this.catalog.ready(CATALOG_READY_TIMEOUT_MS);
		return this.catalog.list().map((id) => {
			const detail = this.catalog.get(id);
			const info = detail ? modelInfo(detail, this.outputCap) : this.fallbackInfo(id);
			return {
				provider,
				id,
				name: info.name,
				inputModalities: toModalities(info.inputModalities)
			};
		});
	}
	async resolveModel(provider, model, _signal) {
		await this.catalog.ready(CATALOG_READY_TIMEOUT_MS);
		const detail = this.catalog.get(model);
		const info = detail ? modelInfo(detail, this.outputCap) : this.fallbackInfo(model);
		const efforts = info.reasoning && info.reasoningEfforts?.length ? info.reasoningEfforts.map((id) => ({
			id,
			name: EFFORT_LABELS[id] ?? id
		})) : void 0;
		return {
			provider,
			id: model,
			name: info.name,
			inputModalities: toModalities(info.inputModalities),
			context: { contextWindow: info.contextWindow },
			defaultMaxTokens: info.maxTokens,
			...efforts ? { reasoning: { efforts } } : {}
		};
	}
	fallbackInfo(id) {
		return {
			id,
			name: id,
			contextWindow: DEFAULT_CONTEXT_WINDOW,
			maxTokens: Math.min(this.outputCap, DEFAULT_MAX_TOKENS),
			inputModalities: ["text"],
			reasoning: false,
			supportsTools: true
		};
	}
	toPiModel(detail, headers) {
		const info = modelInfo(detail, this.outputCap);
		const pricing = detail.pricing ?? {};
		const api = this.resolveApiId(detail);
		const compat = this.compatFor(detail, api);
		return {
			id: detail.id,
			name: info.name,
			api,
			provider: this.providerId,
			baseUrl: this.baseUrl,
			reasoning: info.reasoning,
			...info.reasoning ? { thinkingLevelMap: {
				off: null,
				low: "low",
				medium: "medium",
				high: "high"
			} } : {},
			input: toModalities(info.inputModalities),
			cost: {
				input: numeric(pricing.prompt, 0),
				output: numeric(pricing.completion, 0),
				cacheRead: 0,
				cacheWrite: 0
			},
			contextWindow: info.contextWindow,
			maxTokens: info.maxTokens,
			headers,
			...compat ? { compat } : {}
		};
	}
	fallbackModel(id, headers) {
		const api = "openai-completions";
		const compat = this.compatFor(void 0, api);
		return {
			id,
			name: id,
			api,
			provider: this.providerId,
			baseUrl: this.baseUrl,
			reasoning: false,
			input: ["text"],
			cost: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0
			},
			contextWindow: DEFAULT_CONTEXT_WINDOW,
			maxTokens: Math.min(this.outputCap, DEFAULT_MAX_TOKENS),
			headers,
			...compat ? { compat } : {}
		};
	}
	async *stream(options) {
		const modelId = options.model.trim();
		await this.catalog.ready(CATALOG_READY_TIMEOUT_MS);
		const detail = this.catalog.get(modelId);
		const info = detail ? modelInfo(detail, this.outputCap) : this.fallbackInfo(modelId);
		const headers = this.buildHeaders(options);
		const rawInput = estimateRequestTokens({
			messages: options.messages,
			system: options.system,
			tools: options.tools
		});
		const budget = resolveWireBudget({
			requested: options.maxTokens,
			modelMaxTokens: info.maxTokens,
			contextWindow: info.contextWindow,
			estimatedInputTokens: rawInput,
			calibration: this.#calibration.get(modelId)
		});
		if (budget.overflow) {
			yield {
				type: "usage",
				usage: {
					inputTokens: budget.effectiveInputTokens,
					outputTokens: 0
				}
			};
			yield {
				type: "finish",
				reason: {
					kind: "error",
					failure: {
						message: overflowMessage(budget.effectiveInputTokens, info.contextWindow),
						code: CONTEXT_WINDOW_EXCEEDED_CODE
					}
				}
			};
			return;
		}
		const wireMax = budget.maxTokens ?? MIN_ANSWER_TOKENS;
		const model = detail ? this.toPiModel(detail, headers) : this.fallbackModel(modelId, headers);
		if (options.messages.some((message) => contentHasImage(message.content)) && !model.input.includes("image")) throw new LlmError(`model "${modelId}" does not accept image input (text-only modality)`, "UNSUPPORTED_CONTENT");
		const context = await this.toPiContext(options, model);
		const effort = String(options.reasoningEffort ?? "");
		const reasoning = [
			"minimal",
			"low",
			"medium",
			"high",
			"xhigh",
			"max"
		].includes(effort) ? effort : void 0;
		const stop = options.stop?.length ? options.stop : void 0;
		const consumer = new AbortController();
		const upstream = options.signal ? anySignal([options.signal, consumer.signal]) : consumer.signal;
		const piOptions = {
			apiKey: this.apiKey,
			maxRetries: 0,
			signal: upstream,
			maxTokens: wireMax,
			...options.temperature !== void 0 ? { temperature: options.temperature } : {},
			...reasoning ? { reasoning } : {},
			...options.sessionId ? { sessionId: String(options.sessionId) } : {},
			...stop ? { onPayload: (payload) => injectStop(payload, stop) } : {}
		};
		let idleTimer;
		const armIdle = () => {
			if (idleTimer) clearTimeout(idleTimer);
			idleTimer = setTimeout(() => consumer.abort(/* @__PURE__ */ new Error("stream idle timeout")), STREAM_IDLE_TIMEOUT_MS);
			idleTimer.unref?.();
		};
		const toolIds = /* @__PURE__ */ new Map();
		let calibrated = false;
		const calibrate = (usage) => {
			const billedInput = (usage.input ?? 0) + (usage.cacheRead ?? 0) + (usage.cacheWrite ?? 0);
			if (calibrated || billedInput <= 0 || rawInput <= 0) return;
			calibrated = true;
			this.#calibration.set(modelId, blendCalibration(this.#calibration.get(modelId), billedInput / rawInput));
		};
		try {
			armIdle();
			const events = this.provider.streamSimple(model, context, piOptions);
			for await (const event of events) {
				armIdle();
				switch (event.type) {
					case "start": break;
					case "text_start":
						yield {
							type: "block-start",
							index: event.contentIndex,
							blockType: "text"
						};
						break;
					case "text_delta":
						yield {
							type: "text-delta",
							index: event.contentIndex,
							text: event.delta
						};
						break;
					case "text_end":
						yield {
							type: "block-end",
							index: event.contentIndex,
							block: {
								type: "text",
								text: event.content
							}
						};
						break;
					case "thinking_start":
						yield {
							type: "block-start",
							index: event.contentIndex,
							blockType: "reasoning"
						};
						break;
					case "thinking_delta":
						yield {
							type: "reasoning-delta",
							index: event.contentIndex,
							text: event.delta
						};
						break;
					case "thinking_end":
						yield {
							type: "block-end",
							index: event.contentIndex,
							block: {
								type: "reasoning",
								text: event.content
							}
						};
						break;
					case "toolcall_start": {
						const partial = event.partial.content[event.contentIndex];
						const call = partial?.type === "toolCall" ? partial : void 0;
						toolIds.set(event.contentIndex, {
							id: call?.id ?? "",
							name: call?.name ?? ""
						});
						yield {
							type: "block-start",
							index: event.contentIndex,
							blockType: "tool-call"
						};
						break;
					}
					case "toolcall_delta": {
						const known = toolIds.get(event.contentIndex);
						yield {
							type: "tool-call-delta",
							index: event.contentIndex,
							id: known?.id ?? "",
							...known?.name ? { name: known.name } : {},
							argumentsDelta: event.delta
						};
						break;
					}
					case "toolcall_end":
						yield {
							type: "block-end",
							index: event.contentIndex,
							block: {
								type: "tool-call",
								id: event.toolCall.id,
								name: event.toolCall.name,
								arguments: JSON.stringify(event.toolCall.arguments ?? {})
							}
						};
						break;
					case "done":
						calibrate(event.message.usage);
						yield {
							type: "usage",
							usage: mapUsage(event.message.usage)
						};
						yield {
							type: "finish",
							reason: mapStopReason(event.message, info.contextWindow)
						};
						return;
					case "error":
						calibrate(event.error.usage);
						yield {
							type: "usage",
							usage: mapUsage(event.error.usage)
						};
						yield {
							type: "finish",
							reason: mapStopReason(options.signal?.aborted ? {
								...event.error,
								stopReason: "aborted"
							} : event.error, info.contextWindow)
						};
						return;
				}
			}
			throw new LlmError("pi-ai event stream ended without done/error", "STREAM_CLOSED");
		} catch (error) {
			if (options.signal?.aborted) throw new LlmError("request aborted by caller", "ABORTED", { cause: error });
			if (consumer.signal.aborted) throw new LlmError(`upstream stream was idle for ${STREAM_IDLE_TIMEOUT_MS}ms`, "TIMEOUT", { cause: error });
			throw error;
		} finally {
			if (idleTimer) clearTimeout(idleTimer);
			consumer.abort(/* @__PURE__ */ new Error("stream consumer finished"));
		}
	}
	async toPiContext(options, model) {
		const systemParts = [];
		if (typeof options.system === "string" && options.system.trim()) systemParts.push(options.system.trim());
		const messages = [];
		const toolNames = /* @__PURE__ */ new Map();
		let index = 0;
		for (; index < options.messages.length; index += 1) {
			const message = options.messages[index];
			if (message.role !== "system") break;
			const text = flattenText(message.content);
			if (text) systemParts.push(text);
		}
		for (; index < options.messages.length; index += 1) {
			const message = options.messages[index];
			if (message.role === "system" || message.role === "developer") {
				const text = flattenText(message.content);
				if (text) systemParts.push(text);
				continue;
			}
			if (message.role === "assistant") {
				const assistant = this.toPiAssistant(message, model, toolNames);
				if (assistant.content.length > 0) messages.push(assistant);
				continue;
			}
			if (message.role === "tool") {
				const content$1 = await this.contentToPi(message.content, model);
				messages.push({
					role: "toolResult",
					toolCallId: message.toolCallId,
					toolName: toolNames.get(message.toolCallId) ?? "unknown",
					content: typeof content$1 === "string" ? [{
						type: "text",
						text: content$1 || "(no output)"
					}] : content$1.length > 0 ? content$1 : [{
						type: "text",
						text: "(no output)"
					}],
					isError: message.isError ?? false,
					timestamp: 0
				});
				continue;
			}
			const content = await this.contentToPi(message.content, model);
			messages.push({
				role: "user",
				content: typeof content === "string" && content.length === 0 ? "(empty message)" : content,
				timestamp: 0
			});
		}
		const systemPrompt = systemParts.join("\n\n").trim();
		const tools = options.tools?.map((tool) => ({
			name: tool.name,
			description: tool.description,
			parameters: tool.parameters
		}));
		return {
			...systemPrompt ? { systemPrompt } : {},
			messages,
			...tools && tools.length > 0 ? { tools } : {}
		};
	}
	toPiAssistant(message, model, toolNames) {
		const content = [];
		for (const block of message.content) switch (block.type) {
			case "text":
				if (block.text) content.push({
					type: "text",
					text: block.text
				});
				break;
			case "reasoning":
				if (block.text) content.push({
					type: "thinking",
					thinking: block.text
				});
				break;
			case "tool-call":
				toolNames.set(block.id, block.name);
				content.push({
					type: "toolCall",
					id: block.id,
					name: block.name,
					arguments: parseArguments(block.arguments)
				});
				break;
			default: break;
		}
		const stopReason = content.some((block) => block.type === "toolCall") ? "toolUse" : "stop";
		return {
			role: "assistant",
			content,
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: zeroUsage(),
			stopReason,
			timestamp: 0
		};
	}
	async contentToPi(blocks, model) {
		if (typeof blocks === "string") return blocks;
		const parts = [];
		let hasImage = false;
		for (const block of blocks) switch (block.type) {
			case "text":
				if (block.text.length > 0) parts.push({
					type: "text",
					text: block.text
				});
				break;
			case "image":
				if (block.offloaded === true) {
					parts.push({
						type: "text",
						text: this.offloadedImageText(block)
					});
					break;
				}
				if (!model.input.includes("image")) throw new LlmError(`model "${model.id}" does not accept image input (text-only modality)`, "UNSUPPORTED_CONTENT");
				parts.push(await this.loadImage(block));
				hasImage = true;
				break;
			case "file":
				parts.push({
					type: "text",
					text: this.fileHandleText(block.attachment)
				});
				break;
			default: break;
		}
		if (!hasImage) return parts.map((part) => part.type === "text" ? part.text : "").join("");
		return parts;
	}
	offloadedImageText(block) {
		return `[image omitted from this request: ${block.attachment.name ?? block.attachment.attachmentId}]`;
	}
	fileHandleText(ref) {
		try {
			const text = this.resolveFileText?.(ref);
			if (text && text.length > 0) return text;
		} catch {}
		return `[file: ${ref.name}, ${ref.bytes} bytes]`;
	}
	async loadImage(block) {
		const store = this.resolveAttachments?.();
		if (!store) throw new LlmError("image input requires the durable attachment service", "UNSUPPORTED_CONTENT");
		try {
			const request = await store.readImageRequest(block.attachment, {
				width: block.attachment.width,
				height: block.attachment.height,
				maxBytes: MAX_IMAGE_BYTES
			});
			return {
				type: "image",
				data: Buffer.from(request.data).toString("base64"),
				mimeType: request.mediaType
			};
		} catch (error) {
			throw new LlmError(`unable to read image attachment: ${errorText(error)}`, "UNSUPPORTED_CONTENT", { cause: error });
		}
	}
};
/** Kilo's keyless gateway: any bearer is accepted; "anonymous" is its guest token. */
var KiloFreeAdapter = class extends BaseFreeAdapter {
	constructor(options) {
		super({
			...options,
			defaultKey: options.defaultKey || "anonymous"
		});
	}
	compatFor(_detail, _api) {
		return {
			supportsDeveloperRole: false,
			supportsStrictMode: false,
			supportsStore: false,
			maxTokensField: "max_tokens",
			thinkingFormat: "openrouter",
			supportsReasoningEffort: true,
			supportsLongCacheRetention: false
		};
	}
	buildHeaders(_options) {
		return attributionHeaders();
	}
};
/** OpenCode Zen: free routes are opt-in because the gateway now gates its free tier. */
var ZenFreeAdapter = class ZenFreeAdapter extends BaseFreeAdapter {
	static RESPONSES_MODEL_IDS = new Set(["muse-spark-1.2-contributor-free"]);
	constructor(options) {
		super({
			...options,
			defaultKey: options.defaultKey || "public"
		});
	}
	static zenModelApi(modelId) {
		const id = modelId.trim().toLowerCase();
		if (ZenFreeAdapter.RESPONSES_MODEL_IDS.has(id)) return "openai-responses";
		return /(?:^|[-_:])responses(?:[-_:]|$)/.test(id) ? "openai-responses" : "openai-completions";
	}
	resolveApiId(detail) {
		return detail ? ZenFreeAdapter.zenModelApi(detail.id) : "openai-completions";
	}
	compatFor(_detail, api) {
		if (api === "openai-responses") return {
			supportsDeveloperRole: false,
			supportsStrictMode: false,
			supportsLongCacheRetention: false
		};
		return {
			supportsDeveloperRole: false,
			supportsStrictMode: false,
			supportsStore: false,
			maxTokensField: "max_tokens",
			thinkingFormat: "openai",
			supportsReasoningEffort: true,
			supportsLongCacheRetention: false
		};
	}
	static canonicalSessionId(seed) {
		const value = seed.trim();
		if (/^ses_[0-9a-f]{12}[0-9A-Za-z]{14}$/.test(value)) return value;
		const digest = createHash("sha256").update(value).digest();
		let tail = "";
		const alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
		for (const byte of digest.subarray(6, 20)) tail += alphabet[byte % 62];
		return `ses_${digest.subarray(0, 6).toString("hex")}${tail}`;
	}
	buildHeaders(options) {
		const session = ZenFreeAdapter.canonicalSessionId(sessionSeed(options));
		return {
			"user-agent": "opencode/1.18.35 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.14",
			"x-opencode-client": "cli",
			"x-opencode-session": session,
			"x-opencode-session-id": session,
			"x-session-affinity": session,
			"X-Session-Id": session,
			"x-opencode-project": "dsh-my-free-adapter",
			"authorization": `Bearer ${this.apiKey}`
		};
	}
};

//#endregion
//#region src/index.ts
const name = "my-free-adapter";
const inject = ["llm"];
const DEFAULT_MAX_OUTPUT_TOKENS = 32768;
const Config = Schema.object({
	kiloEnabled: Schema.boolean().default(true),
	kiloProviderId: Schema.string().default("my-kilo"),
	kiloBaseUrl: Schema.string().default(KILO_GATEWAY_BASE_URL),
	kiloApiKey: Schema.string().default(""),
	zenEnabled: Schema.boolean().default(false),
	zenProviderId: Schema.string().default("my-zen"),
	zenBaseUrl: Schema.string().default(OPENCODE_ZEN_BASE_URL),
	zenApiKey: Schema.string().default("public"),
	zenCapabilitiesUrl: Schema.string().default(OPENCODE_CAPABILITIES_URL),
	maxOutputTokens: Schema.number().default(DEFAULT_MAX_OUTPUT_TOKENS),
	dataDir: Schema.string().default("")
});
function nonEmpty(value, fallback) {
	const trimmed = value?.trim() ?? "";
	return trimmed.length > 0 ? trimmed : fallback;
}
function outputCap(value) {
	return Number.isFinite(value) && value > 0 ? Math.floor(value) : DEFAULT_MAX_OUTPUT_TOKENS;
}
function apply(ctx, config) {
	const dataDir = nonEmpty(config.dataDir, join(homedir(), ".my-free-adapter"));
	const maxOutputTokens = outputCap(config.maxOutputTokens);
	const catalogs = [];
	const kiloEnabled = config.kiloEnabled !== false;
	const zenEnabled = config.zenEnabled === true;
	if (!kiloEnabled && !zenEnabled) {
		ctx.logger.warn("[my-free] both Kilo and Zen routes are disabled; no provider registered");
		return;
	}
	const attachments = () => ctx.get("attachments");
	const fileText = (ref) => ctx.llm.fileRequestText(ref);
	if (kiloEnabled) {
		const providerId = nonEmpty(config.kiloProviderId, "my-kilo");
		const baseUrl = nonEmpty(config.kiloBaseUrl, KILO_GATEWAY_BASE_URL);
		const catalog = new ModelCatalog({
			url: `${baseUrl.replace(/\/+$/, "")}/models`,
			cachePath: join(dataDir, "kilo-models.json"),
			staticFallback: KILO_STATIC_FALLBACK,
			gatewayMaxOutputTokens: maxOutputTokens,
			startupRetryMs: 15e3,
			statusPath: join(dataDir, "kilo-status.json"),
			label: "kilo",
			fetchTimeoutMs: 3e4,
			headers: () => attributionHeaders(),
			onError: (message) => ctx.logger.warn(`[my-free] Kilo catalog refresh failed: ${message}`)
		}, (count) => ctx.logger.info(`[my-free] Kilo catalog: ${count} models`));
		catalogs.push(catalog);
		const adapter = new KiloFreeAdapter({
			catalog,
			providerId,
			baseUrl,
			providerName: "Kilo Free Gateway",
			apiKey: config.kiloApiKey,
			defaultKey: "anonymous",
			maxOutputTokens,
			resolveAttachments: attachments,
			resolveFileText: fileText
		});
		try {
			ctx.llm.registerAdapter([providerId], adapter);
			catalog.start();
			ctx.logger.info(`[my-free] Kilo adapter registered: ${providerId}`);
		} catch (error) {
			catalog.stop();
			ctx.logger.error(`[my-free] could not register Kilo provider "${providerId}": ${String(error)}`);
		}
	}
	if (zenEnabled) {
		const providerId = nonEmpty(config.zenProviderId, "my-zen");
		const baseUrl = nonEmpty(config.zenBaseUrl, OPENCODE_ZEN_BASE_URL);
		const capabilitiesUrl = nonEmpty(config.zenCapabilitiesUrl, OPENCODE_CAPABILITIES_URL);
		const catalog = new ModelCatalog({
			url: `${baseUrl.replace(/\/+$/, "")}/models`,
			cachePath: join(dataDir, "zen-models.json"),
			staticFallback: ZEN_STATIC_FALLBACK,
			freePredicate: isFreeModel,
			gatewayMaxOutputTokens: maxOutputTokens,
			capabilitiesUrl: capabilitiesUrl || void 0,
			capabilityUserAgent: "opencode/1.18.35 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.14",
			startupRetryMs: 15e3,
			statusPath: join(dataDir, "zen-status.json"),
			label: "zen",
			fetchTimeoutMs: 2e4,
			onError: (message) => ctx.logger.warn(`[my-free] Zen catalog refresh failed: ${message}`),
			headers: () => ({
				"user-agent": "opencode/1.18.35 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.14",
				"x-opencode-client": "cli",
				"authorization": `Bearer ${nonEmpty(config.zenApiKey, "public")}`
			})
		}, (count) => ctx.logger.info(`[my-free] Zen catalog: ${count} models`));
		catalogs.push(catalog);
		const adapter = new ZenFreeAdapter({
			catalog,
			providerId,
			baseUrl,
			providerName: "OpenCode Zen (free, opt-in)",
			apiKey: config.zenApiKey,
			defaultKey: "public",
			maxOutputTokens,
			resolveAttachments: attachments,
			resolveFileText: fileText
		});
		try {
			ctx.llm.registerAdapter([providerId], adapter);
			catalog.start();
			ctx.logger.info(`[my-free] Zen adapter registered: ${providerId}`);
			ctx.logger.warn("[my-free] OpenCode Zen now answers free-tier requests from third-party clients with 403 FreeTierError; keep zenEnabled only if your client passes its gate");
		} catch (error) {
			catalog.stop();
			ctx.logger.error(`[my-free] could not register Zen provider "${providerId}": ${String(error)}`);
		}
	}
	ctx.effect(() => () => {
		for (const catalog of catalogs) catalog.stop();
	});
}

//#endregion
export { Config, apply, inject, name };