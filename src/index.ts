import Schema from '@deepseek-ai/schemastery'
import { ModelCatalog, KILO_GATEWAY_BASE_URL } from './catalog.ts'
import { KiloFreeAdapter } from './adapter.ts'

export const name = 'my-free-adapter'
export const inject = ['llm']

export const Config = Schema.object({
  providerId: Schema.string().default('my-free'),
  baseUrl: Schema.string().default(KILO_GATEWAY_BASE_URL),
})

export function apply(ctx: any, config: { providerId: string; baseUrl: string }) {
  const catalog = new ModelCatalog(`${config.baseUrl.replace(/\/+$/, '')}/models`)
  const adapter = new KiloFreeAdapter(catalog, config.providerId, config.baseUrl)
  const dispose = ctx.llm.registerAdapter([config.providerId], adapter)
  void catalog.start()
  ctx.effect(() => () => {
    catalog.stop()
    dispose?.()
  })
}