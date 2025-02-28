import { Context, Schema } from 'koishi'
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'
import { OllamaClient } from './client'

export function apply(ctx: Context, config: Config) {
    const plugin = new ChatLunaPlugin(ctx, config, 'ollama')

    ctx.on('ready', async () => {
        plugin.registerToService()

        await plugin.parseConfig((config) => {
            return config.apiEndpoints.map((apiEndpoint) => {
                return {
                    apiKey: '',
                    apiEndpoint,
                    platform: 'ollama',
                    chatLimit: config.chatTimeLimit,
                    timeout: config.timeout,
                    maxRetries: config.maxRetries,
                    concurrentMaxSize: config.chatConcurrentMaxSize
                }
            })
        })

        plugin.registerClient(
            (_, clientConfig) =>
                new OllamaClient(ctx, config, clientConfig, plugin)
        )

        await plugin.initClients()
    })
}

export interface Config extends ChatLunaPlugin.Config {
    apiEndpoints: string[]
    maxTokens: number
    temperature: number
    presencePenalty: number
    frequencyPenalty: number
    supportImage: boolean
    keepAlive: boolean
}

export const Config: Schema<Config> = Schema.intersect([
    ChatLunaPlugin.Config,
    Schema.object({
        apiEndpoints: Schema.array(Schema.string()).default([
            'http://127.0.0.1:11434'
        ]),
        supportImage: Schema.boolean().default(true),
        keepAlive: Schema.boolean().default(true)
    }),
    Schema.object({
        maxTokens: Schema.number().min(16).max(128000).step(16).default(4096),
        temperature: Schema.percent().min(0).max(2).step(0.1).default(0.8),
        presencePenalty: Schema.number().min(-2).max(2).step(0.1).default(0.2),
        frequencyPenalty: Schema.number().min(-2).max(2).step(0.1).default(0.2)
    })
]).i18n({
    'zh-CN': require('./locales/zh-CN.schema.yml'),
    'en-US': require('./locales/en-US.schema.yml')
})

export const inject = ['chatluna']

export const name = 'chatluna-ollama-adapter'
