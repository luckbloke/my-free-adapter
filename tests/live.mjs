/**
 * Live smoke tests for the free adapters against the real gateways.
 *
 * Run with: node tests/live.mjs
 * These are not part of the published package.
 */
import assert from 'node:assert/strict'
import { mkdir } from 'node:fs/promises'
import { join } from 'node:path'
import {
  ModelCatalog,
  KILO_GATEWAY_BASE_URL,
  KILO_STATIC_FALLBACK,
  ZEN_STATIC_FALLBACK,
  isFreeModel,
} from '../src/catalog.ts'
import { KiloFreeAdapter } from '../src/adapter.ts'

const CACHE_DIR = join(process.cwd(), '.cache', 'live-tests')
await mkdir(CACHE_DIR, { recursive: true })

function timeout(ms) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(new Error(`test timed out after ${ms}ms`)), ms)
  timer.unref?.()
  return { signal: controller.signal, clear: () => clearTimeout(timer) }
}

async function collect(adapter, options, ms = 120_000) {
  const guard = timeout(ms)
  const chunks = []
  try {
    for await (const chunk of adapter.stream({ ...options, signal: guard.signal })) chunks.push(chunk)
  } finally {
    guard.clear()
  }
  return chunks
}

function summarize(chunks) {
  const text = chunks.filter((c) => c.type === 'text-delta').map((c) => c.text).join('')
  const reasoning = chunks.filter((c) => c.type === 'reasoning-delta').map((c) => c.text).join('')
  const toolCalls = chunks
    .filter((c) => c.type === 'block-end' && c.block.type === 'tool-call')
    .map((c) => c.block)
  const toolDeltas = chunks.filter((c) => c.type === 'tool-call-delta')
  const usage = chunks.find((c) => c.type === 'usage')?.usage
  const finish = chunks.find((c) => c.type === 'finish')?.reason
  return { text, reasoning, toolCalls, toolDeltas, usage, finish }
}

const catalog = new ModelCatalog({
  url: `${KILO_GATEWAY_BASE_URL}/models`,
  cachePath: join(CACHE_DIR, 'kilo-models.json'),
  statusPath: join(CACHE_DIR, 'kilo-status.json'),
  staticFallback: KILO_STATIC_FALLBACK,
  gatewayMaxOutputTokens: 32_768,
  startupRetryMs: 2_000,
  fetchTimeoutMs: 30_000,
  label: 'kilo-test',
}, (count) => console.log(`[catalog] refreshed ${count} models`))

console.log('== Kilo catalog ==')
await catalog.start()
await catalog.ready(20_000)
const ids = catalog.list()
console.log(`free/tool/text models: ${ids.length}`)
console.log(ids.slice(0, 15).join('\n'))
assert.ok(ids.length > 0, 'Kilo catalog must expose at least one free tool-capable model')
assert.ok(ids.includes('kilo-auto/free'), 'kilo-auto/free must be listed')

const adapter = new KiloFreeAdapter({
  catalog,
  providerId: 'my-kilo',
  baseUrl: KILO_GATEWAY_BASE_URL,
  providerName: 'Kilo Free Gateway (test)',
  defaultKey: 'anonymous',
  maxOutputTokens: 32_768,
})

console.log('\n== listModels ==')
const listed = await adapter.listModels('my-kilo')
console.log(`adapter advertises ${listed.length} models; first: ${listed[0]?.id}`)
assert.equal(listed.length, ids.length)

const preferred = ['nvidia/nemotron-3.5-lightning:free', 'kilo-auto/free', 'openrouter/free']
const model = preferred.find((id) => ids.includes(id)) ?? ids[0]
const resolved = await adapter.resolveModel('my-kilo', model)
console.log('\n== resolveModel ==')
console.log(JSON.stringify(resolved, null, 2))
assert.equal(resolved.provider, 'my-kilo')
assert.ok(resolved.context.contextWindow > 0)
assert.ok(resolved.defaultMaxTokens > 0)

console.log(`\n== text stream (${model}) ==`)
const textChunks = await collect(adapter, {
  provider: 'my-kilo',
  model,
  messages: [{ role: 'user', content: [{ type: 'text', text: 'Reply with exactly the word OK and nothing else.' }] }],
  maxTokens: 64,
})
const textResult = summarize(textChunks)
console.log({ text: textResult.text.slice(0, 200), finish: textResult.finish, usage: textResult.usage })
assert.ok(textChunks.some((c) => c.type === 'usage'), 'usage chunk must be emitted')
assert.ok(textChunks.at(-1)?.type === 'finish', 'finish chunk must be last')

console.log(`\n== tool-call stream (${model}) ==`)
const toolChunks = await collect(adapter, {
  provider: 'my-kilo',
  model,
  messages: [{
    role: 'user',
    content: [{ type: 'text', text: 'Call the get_time tool with an empty object. Do not answer in text.' }],
  }],
  tools: [{
    name: 'get_time',
    description: 'Return the current time.',
    parameters: { type: 'object', properties: {}, additionalProperties: false },
  }],
  maxTokens: 128,
})
const toolResult = summarize(toolChunks)
console.log({
  text: toolResult.text.slice(0, 100),
  toolCalls: toolResult.toolCalls,
  deltaIds: toolResult.toolDeltas.map((d) => d.id),
  finish: toolResult.finish,
})
for (const call of toolResult.toolCalls) {
  assert.ok(call.id, 'tool call block must carry a non-empty id')
  assert.equal(call.name, 'get_time')
  assert.doesNotThrow(() => JSON.parse(call.arguments))
}
for (const delta of toolResult.toolDeltas) {
  assert.ok(delta.id, 'tool-call delta must carry the provider id')
}

console.log(`\n== tool history conversion (${model}) ==`)
const historyChunks = await collect(adapter, {
  provider: 'my-kilo',
  model,
  messages: [
    { role: 'user', content: [{ type: 'text', text: 'What is the value?' }] },
    {
      role: 'assistant',
      content: [
        { type: 'text', text: 'Let me check.' },
        { type: 'tool-call', id: 'call_test_1', name: 'get_time', arguments: '{}' },
      ],
    },
    {
      role: 'tool',
      toolCallId: 'call_test_1',
      content: [{ type: 'text', text: 'The value is 42.' }],
    },
    { role: 'user', content: [{ type: 'text', text: 'Reply with only the number from the tool result.' }] },
  ],
  tools: [{
    name: 'get_time',
    description: 'Return the current time.',
    parameters: { type: 'object', properties: {}, additionalProperties: false },
  }],
  maxTokens: 64,
})
const historyResult = summarize(historyChunks)
console.log({ text: historyResult.text.slice(0, 200), finish: historyResult.finish })
assert.ok(!historyResult.finish?.failure, 'tool history request must not fail')
assert.ok(historyResult.text.includes('42'), 'model must see the tool result')

console.log('\n== Zen free predicate (offline) ==')
const zenIds = ZEN_STATIC_FALLBACK.map((model) => model.id)
for (const id of zenIds) {
  assert.ok(isFreeModel({ id }), `${id} must classify as free`)
}
assert.ok(!isFreeModel({ id: 'claude-opus-5' }), 'paid model must not classify as free')
assert.ok(!isFreeModel({ id: 'test' }), 'test catalog entry must not classify as free')
assert.ok(isFreeModel({ id: 'some-model', pricing: { prompt: '0', completion: '0' } }), 'zero pricing is free')

catalog.stop()
console.log('\nAll live tests passed.')
