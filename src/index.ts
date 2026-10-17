import Schema from '@deepseek-ai/schemastery'
import { attributionHeaders } from '@deepseek-ai/dsh-llm'
import { homedir } from 'node:os'
import { join } from 'node:path'
import type { Context } from '@deepseek-ai/cordis'
import {
  KILO_GATEWAY_BASE_URL,
  KILO_STATIC_FALLBACK,
  ModelCatalog,
  OPENCODE_CAPABILITIES_URL,
  OPENCODE_ZEN_BASE_URL,
  ZEN_STATIC_FALLBACK,
  isFreeModel,
} from './catalog.ts'
import { KiloFreeAdapter, ZenFreeAdapter } from './adapter.ts'
import type { AttachmentStoreLike, FileRefLike } from './adapter.ts'

export const name = 'my-free-adapter'
export const inject = ['llm']

const DEFAULT_MAX_OUTPUT_TOKENS = 32_768

export const Config = Schema.object({
  kiloEnabled: Schema.boolean().default(true),
  kiloProviderId: Schema.string().default('my-kilo'),
  kiloBaseUrl: Schema.string().default(KILO_GATEWAY_BASE_URL),
  kiloApiKey: Schema.string().default(''),
  zenEnabled: Schema.boolean().default(false),
  zenProviderId: Schema.string().default('my-zen'),
  zenBaseUrl: Schema.string().default(OPENCODE_ZEN_BASE_URL),
  zenApiKey: Schema.string().default('public'),
  zenCapabilitiesUrl: Schema.string().default(OPENCODE_CAPABILITIES_URL),
  maxOutputTokens: Schema.number().default(DEFAULT_MAX_OUTPUT_TOKENS),
  dataDir: Schema.string().default(''),
})

export interface ConfigShape {
  kiloEnabled: boolean
  kiloProviderId: string
  kiloBaseUrl: string
  kiloApiKey: string
  zenEnabled: boolean
  zenProviderId: string
  zenBaseUrl: string
  zenApiKey: string
  zenCapabilitiesUrl: string
  maxOutputTokens: number
  dataDir: string
}

function nonEmpty(value: string | undefined, fallback: string): string {
  const trimmed = value?.trim() ?? ''
  return trimmed.length > 0 ? trimmed : fallback
}

function outputCap(value: number): number {
  return Number.isFinite(value) && value > 0 ? Math.floor(value) : DEFAULT_MAX_OUTPUT_TOKENS
}

export function apply(ctx: Context, config: ConfigShape): void {
  const dataDir = nonEmpty(config.dataDir, join(homedir(), '.my-free-adapter'))
  const maxOutputTokens = outputCap(config.maxOutputTokens)
  const catalogs: ModelCatalog[] = []

  const kiloEnabled = config.kiloEnabled !== false
  const zenEnabled = config.zenEnabled === true

  if (!kiloEnabled && !zenEnabled) {
    ctx.logger.warn('[my-free] both Kilo and Zen routes are disabled; no provider registered')
    return
  }

  const attachments = (): AttachmentStoreLike | undefined =>
    ctx.get('attachments') as AttachmentStoreLike | undefined
  const fileText = (ref: FileRefLike): string =>
    ctx.llm.fileRequestText(ref as never)

  // ---- Kilo (keyless, the always-on default) ----
  if (kiloEnabled) {
    const providerId = nonEmpty(config.kiloProviderId, 'my-kilo')
    const baseUrl = nonEmpty(config.kiloBaseUrl, KILO_GATEWAY_BASE_URL)
    const catalog = new ModelCatalog({
      url: `${baseUrl.replace(/\/+$/, '')}/models`,
      cachePath: join(dataDir, 'kilo-models.json'),
      staticFallback: KILO_STATIC_FALLBACK,
      gatewayMaxOutputTokens: maxOutputTokens,
      startupRetryMs: 15_000,
      statusPath: join(dataDir, 'kilo-status.json'),
      label: 'kilo',
      fetchTimeoutMs: 30_000,
      headers: () => attributionHeaders(),
      onError: (message) => ctx.logger.warn(`[my-free] Kilo catalog refresh failed: ${message}`),
    }, (count) => ctx.logger.info(`[my-free] Kilo catalog: ${count} models`))
    catalogs.push(catalog)

    const adapter = new KiloFreeAdapter({
      catalog,
      providerId,
      baseUrl,
      providerName: 'Kilo Free Gateway',
      apiKey: config.kiloApiKey,
      defaultKey: 'anonymous',
      maxOutputTokens,
      resolveAttachments: attachments,
      resolveFileText: fileText,
    })

    try {
      ctx.llm.registerAdapter([providerId], adapter)
      void catalog.start()
      ctx.logger.info(`[my-free] Kilo adapter registered: ${providerId}`)
    } catch (error) {
      catalog.stop()
      ctx.logger.error(`[my-free] could not register Kilo provider "${providerId}": ${String(error)}`)
    }
  }

  // ---- OpenCode Zen (opt-in: free tier now enforces the OpenCode client) ----
  if (zenEnabled) {
    const providerId = nonEmpty(config.zenProviderId, 'my-zen')
    const baseUrl = nonEmpty(config.zenBaseUrl, OPENCODE_ZEN_BASE_URL)
    const capabilitiesUrl = nonEmpty(config.zenCapabilitiesUrl, OPENCODE_CAPABILITIES_URL)
    const catalog = new ModelCatalog({
      url: `${baseUrl.replace(/\/+$/, '')}/models`,
      cachePath: join(dataDir, 'zen-models.json'),
      staticFallback: ZEN_STATIC_FALLBACK,
      freePredicate: isFreeModel,
      gatewayMaxOutputTokens: maxOutputTokens,
      capabilitiesUrl: capabilitiesUrl || undefined,
      capabilityUserAgent: 'opencode/1.18.35 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.14',
      startupRetryMs: 15_000,
      statusPath: join(dataDir, 'zen-status.json'),
      label: 'zen',
      fetchTimeoutMs: 20_000,
      onError: (message) => ctx.logger.warn(`[my-free] Zen catalog refresh failed: ${message}`),
      headers: () => ({
        'user-agent': 'opencode/1.18.35 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.14',
        'x-opencode-client': 'cli',
        'authorization': `Bearer ${nonEmpty(config.zenApiKey, 'public')}`,
      }),
    }, (count) => ctx.logger.info(`[my-free] Zen catalog: ${count} models`))
    catalogs.push(catalog)

    const adapter = new ZenFreeAdapter({
      catalog,
      providerId,
      baseUrl,
      providerName: 'OpenCode Zen (free, opt-in)',
      apiKey: config.zenApiKey,
      defaultKey: 'public',
      maxOutputTokens,
      resolveAttachments: attachments,
      resolveFileText: fileText,
    })

    try {
      ctx.llm.registerAdapter([providerId], adapter)
      void catalog.start()
      ctx.logger.info(`[my-free] Zen adapter registered: ${providerId}`)
      ctx.logger.warn(
        '[my-free] OpenCode Zen now answers free-tier requests from third-party clients with '
        + '403 FreeTierError; keep zenEnabled only if your client passes its gate',
      )
    } catch (error) {
      catalog.stop()
      ctx.logger.error(`[my-free] could not register Zen provider "${providerId}": ${String(error)}`)
    }
  }

  ctx.effect(() => () => {
    for (const catalog of catalogs) catalog.stop()
  })
}
