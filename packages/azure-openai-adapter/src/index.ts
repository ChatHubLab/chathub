import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'
import { Context, Logger, Schema } from 'koishi'
import { OpenAIClient } from './client'
import { createLogger } from 'koishi-plugin-chatluna/utils/logger'
import { AzureOpenAIClientConfig } from './types'

export let logger: Logger

export function apply(ctx: Context, config: Config) {
    const plugin = new ChatLunaPlugin<AzureOpenAIClientConfig, Config>(
        ctx,
        config,
        'azure'
    )

    logger = createLogger(ctx, 'chatluna-openai-adapter')

    ctx.on('ready', async () => {
        plugin.registerToService()

        await plugin.parseConfig((config) => {
            return config.apiKeys.map(([apiKey, apiEndpoint]) => {
                return {
                    apiKey,
                    apiEndpoint,
                    // [{model,xx}] => Record<string(model),{}>
                    supportModels: config.supportModels.reduce((acc, value) => {
                        acc[value.model] = value
                        return acc
                    }, {}),
                    platform: 'azure',
                    chatLimit: config.chatTimeLimit,
                    timeout: config.timeout,
                    maxRetries: config.maxRetries,
                    concurrentMaxSize: config.chatConcurrentMaxSize
                }
            })
        })

        plugin.registerClient(
            (_, clientConfig) =>
                new OpenAIClient(ctx, config, clientConfig, plugin)
        )

        await plugin.initClients()
    })
}

export interface Config extends ChatLunaPlugin.Config {
    apiKeys: [string, string][]
    maxTokens: number
    supportModels: {
        model: string
        modelType: string
        modelVersion: string
        contextSize: number
    }[]
    temperature: number
    presencePenalty: number
    frequencyPenalty: number
}

export const Config: Schema<Config> = Schema.intersect([
    ChatLunaPlugin.Config,
    Schema.object({
        apiKeys: Schema.array(
            Schema.tuple([
                Schema.string().role('secret').required(),
                Schema.string().default('https://xxx.openai.azure.com')
            ])
        ).default([['', 'https://xxx.openai.azure.com']])
    }),

    Schema.object({
        supportModels: Schema.array(
            Schema.object({
                model: Schema.string().required(),
                modelType: Schema.union([
                    'LLM 大语言模型',
                    'LLM 大语言模型（函数调用）',
                    'Embeddings 嵌入模型'
                ]).default('LLM 大语言模型'),
                modelVersion: Schema.string().default('2023-03-15-preview'),
                contextSize: Schema.number().default(4096)
            }).role('table')
        ).default([
            {
                model: 'gpt-4o',
                modelType: 'LLM 大语言模型',
                modelVersion: '2023-03-15-preview',
                contextSize: 128000
            }
        ]),
        maxTokens: Schema.number().min(16).max(128000).step(16).default(1024),
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

export const name = 'chatluna-openai-adapter'
