/**
 * Loads the built bundle with a fake Cordis context, the same shape the DSH
 * loader supplies. Verifies exports, registration, model listing and disposal.
 *
 * Run with: node tests/smoke.mjs
 */
import assert from 'node:assert/strict'
import { apply, Config, inject, name } from '../lib/index.js'

assert.equal(name, 'my-free-adapter')
assert.deepEqual(inject, ['llm'])
assert.equal(typeof apply, 'function')
assert.equal(typeof Config, 'function')

const registrations = []
const logs = []
const disposers = []

const ctx = {
  llm: {
    registerAdapter(providers, adapter) {
      registrations.push({ providers, adapter })
      return () => {}
    },
    fileRequestText(ref) {
      return `[file ${ref.name}]`
    },
  },
  get(service) {
    return service === 'attachments' ? undefined : undefined
  },
  logger: {
    info: (message) => logs.push(['info', String(message)]),
    warn: (message) => logs.push(['warn', String(message)]),
    error: (message) => logs.push(['error', String(message)]),
  },
  effect(callback) {
    disposers.push(callback())
    return callback
  },
}

apply(ctx, {
  kiloEnabled: true,
  kiloProviderId: 'my-kilo',
  kiloBaseUrl: 'https://api.kilo.ai/api/gateway',
  kiloApiKey: '',
  zenEnabled: false,
  zenProviderId: 'my-zen',
  zenBaseUrl: 'https://opencode.ai/zen/v1',
  zenApiKey: 'public',
  zenCapabilitiesUrl: 'https://models.opencode.ai/api.json',
  maxOutputTokens: 32768,
  dataDir: '.cache/smoke',
})

assert.equal(registrations.length, 1, 'exactly one adapter is registered')
assert.deepEqual(registrations[0].providers, ['my-kilo'])
assert.equal(registrations[0].adapter.providerInfo('my-kilo').name, 'Kilo Free Gateway')

const models = await registrations[0].adapter.listModels('my-kilo')
console.log(`registered models: ${models.length}; sample: ${models.slice(0, 3).map((m) => m.id).join(', ')}`)
assert.ok(models.length > 0, 'built bundle must advertise models')

const resolved = await registrations[0].adapter.resolveModel('my-kilo', models[0].id)
assert.equal(resolved.provider, 'my-kilo')
assert.ok(resolved.context.contextWindow > 0)

const streamModel = models.find((m) => m.id === 'nvidia/nemotron-3.5-lightning:free')?.id ?? models[0].id
const chunks = []
for await (const chunk of registrations[0].adapter.stream({
  provider: 'my-kilo',
  model: streamModel,
  messages: [{ role: 'user', content: [{ type: 'text', text: 'Reply with exactly OK.' }] }],
  maxTokens: 64,
})) chunks.push(chunk)
assert.ok(chunks.some((c) => c.type === 'usage'), 'built bundle must emit usage')
assert.equal(chunks.at(-1)?.type, 'finish', 'built bundle must end with finish')
console.log(`streamed ${streamModel}: ${chunks.length} chunks, finish=${JSON.stringify(chunks.at(-1).reason)}`)

for (const dispose of disposers) dispose()
for (const [, level, message] of logs) if (level === 'error') throw new Error(message)
console.log(logs.map(([level, message]) => `${level}: ${message}`).join('\n'))
console.log('Smoke test passed.')
