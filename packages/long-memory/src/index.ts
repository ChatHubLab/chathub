/* eslint-disable @typescript-eslint/naming-convention */
import { Context, Logger, Schema } from 'koishi'
import { ClientConfig } from 'koishi-plugin-chatluna/llm-core/platform/config'
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'
import { createLogger } from 'koishi-plugin-chatluna/utils/logger'
import { plugins } from './plugin'
import { ChatLunaLongMemoryService } from './service/memory'

export let logger: Logger

export function apply(ctx: Context, config: Config) {
    logger = createLogger(ctx, 'chatluna-long-memory')
    const plugin = new ChatLunaPlugin<ClientConfig, Config>(
        ctx,
        config,
        'long-memory',
        false
    )

    ctx.on('ready', async () => {
        plugin.registerToService()

        ctx.plugin(ChatLunaLongMemoryService, config)

        ctx.inject(['chatluna_long_memory'], (ctx) => {
            ctx.on('ready', async () => {
                await plugins(ctx, config, plugin)
            })
        })
    })
}

export interface Config extends ChatLunaPlugin.Config {
    longMemoryNewQuestionSearch: boolean
    longMemorySimilarity: number
    longMemoryTFIDFThreshold: number
    longMemoryDuplicateThreshold: number
    longMemoryDuplicateCheck: boolean
    longMemoryInterval: number
    longMemoryExtractModel: string
    longMemoryLayer: string[]
}

export const Config: Schema<Config> = Schema.intersect([
    Schema.object({
        longMemoryInterval: Schema.number().default(3).min(1).max(10),
        longMemoryNewQuestionSearch: Schema.boolean().default(false),
        longMemorySimilarity: Schema.percent()
            .min(0)
            .max(1)
            .step(0.01)
            .default(0.3),
        longMemoryTFIDFThreshold: Schema.percent()
            .min(0)
            .max(1)
            .step(0.01)
            .default(0),
        longMemoryDuplicateCheck: Schema.boolean().default(true),
        longMemoryDuplicateThreshold: Schema.percent()
            .min(0)
            .max(1)
            .step(0.01)
            .default(0.8),
        longMemoryLayer: Schema.array(
            Schema.union([
                Schema.const('Global'),
                Schema.const('Preset'),
                Schema.const('Preset_User'),
                Schema.const('User')
            ])
        )
            .role('checkbox')
            .default(['Preset_User']),
        longMemoryExtractModel: Schema.dynamic('model').default('无')
    })
]).i18n({
    'zh-CN': require('./locales/zh-CN.schema.yml'),
    'en-US': require('./locales/en-US.schema.yml')
}) as Schema<Config>

export const inject = ['chatluna']

export const name = 'chatluna-long-memory'

export * from './types'
export * from './service/memory'
