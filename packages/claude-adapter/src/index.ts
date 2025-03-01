import { ClientConfig } from 'koishi-plugin-chatluna/llm-core/platform/config'
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'
import { createLogger } from 'koishi-plugin-chatluna/utils/logger'
import { Context, Logger, Schema } from 'koishi'
import { ClaudeClient } from './client'

export let logger: Logger
export function apply(ctx: Context, config: Config) {
    const plugin = new ChatLunaPlugin<ClientConfig, Config>(
        ctx,
        config,
        'claude'
    )

    logger = createLogger(ctx, 'chatluna-claude-adapter')

    ctx.on('ready', async () => {
        plugin.registerToService()

        await plugin.parseConfig((config) =>
            config.apiKeys.map((apiKey) => {
                return {
                    apiKey: apiKey[0],
                    apiEndpoint: apiKey[1],
                    platform: 'claude',
                    chatLimit: config.chatTimeLimit,
                    timeout: config.timeout,
                    maxRetries: config.maxRetries,
                    concurrentMaxSize: config.chatConcurrentMaxSize
                }
            })
        )

        plugin.registerClient(
            (_, clientConfig) =>
                new ClaudeClient(ctx, config, clientConfig, plugin)
        )

        await plugin.initClients()
    })
}

export interface Config extends ChatLunaPlugin.Config {
    apiKeys: [string, string][]
    maxTokens: number
    temperature: number
    presencePenalty: number
    frequencyPenalty: number
}

export const Config: Schema<Config> = Schema.intersect([
    ChatLunaPlugin.Config,
    Schema.object({
        apiKeys: Schema.array(
            Schema.tuple([
                Schema.string().role('secret'),
                Schema.string().default('https://api.anthropic.com/v1')
            ])
        ).default([['', 'https://api.anthropic.com/v1']])
    }),
    Schema.object({
        maxTokens: Schema.number().min(16).max(200000).step(16).default(8000),
        temperature: Schema.percent().min(0).max(2).step(0.1).default(0.8),
        presencePenalty: Schema.number().min(-2).max(2).step(0.1).default(0.2),
        frequencyPenalty: Schema.number().min(-2).max(2).step(0.1).default(0.2)
    })
]).i18n({
    'zh-CN': require('./locales/zh-CN.schema.yml'),
    'en-US': require('./locales/en-US.schema.yml')
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
}) as any

export const inject = ['chatluna']

export const name = 'chatluna-claude-adapter'
