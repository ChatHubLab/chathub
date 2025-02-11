import { Context, Logger, Schema } from 'koishi'
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'
import {
    ChatLunaError,
    ChatLunaErrorCode
} from 'koishi-plugin-chatluna/utils/error'
import { createLogger } from 'koishi-plugin-chatluna/utils/logger'
import { OpenAIClient } from './client'

export let logger: Logger
export const reusable = true

export function apply(ctx: Context, config: Config) {
    if (config.platform == null || config.platform.length < 1) {
        throw new ChatLunaError(
            ChatLunaErrorCode.UNKNOWN_ERROR,
            new Error('Cannot find any platform')
        )
    }

    const platform = config.platform

    const plugin = new ChatLunaPlugin(ctx, config, platform)

    logger = createLogger(ctx, 'chatluna-openai-like-adapter')

    ctx.on('ready', async () => {
        plugin.registerToService()

        await plugin.parseConfig((config) => {
            return config.apiKeys.map(([apiKey, apiEndpoint]) => {
                return {
                    apiKey,
                    apiEndpoint,
                    platform,
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
    pullModels: boolean
    additionalModels: {
        model: string
        modelType: string
        contextSize: number
    }[]
    additionCookies: [string, string][]
    maxTokens: number
    temperature: number
    presencePenalty: number
    platform: string
    frequencyPenalty: number
    googleSearch: boolean
    googleSearchSupportModel: string[]
}

export const Config: Schema<Config> = Schema.intersect([
    ChatLunaPlugin.Config,
    Schema.object({
        platform: Schema.string().default('openai-like'),
        pullModels: Schema.boolean().default(true),
        additionalModels: Schema.array(
            Schema.object({
                model: Schema.string(),
                modelType: Schema.union([
                    'LLM 大语言模型',
                    'LLM 大语言模型（函数调用）',
                    'Embeddings 嵌入模型'
                ]).default('LLM 大语言模型'),
                contextSize: Schema.number().default(4096)
            }).role('table')
        ).default([])
    }),
    Schema.object({
        apiKeys: Schema.array(
            Schema.tuple([
                Schema.string().role('secret'),
                Schema.string().default('https://api.openai.com/v1')
            ])
        ).default([['', 'https://api.openai.com/v1']]),
        additionCookies: Schema.array(
            Schema.tuple([Schema.string(), Schema.string()])
        ).default([])
    }),
    Schema.object({
        maxTokens: Schema.number().min(16).max(2000000).step(16).default(4096),
        temperature: Schema.percent().min(0).max(2).step(0.1).default(0.8),
        presencePenalty: Schema.number().min(-2).max(2).step(0.1).default(0.2),
        frequencyPenalty: Schema.number().min(-2).max(2).step(0.1).default(0.2)
    }),
    Schema.object({
        googleSearch: Schema.boolean().default(false),
        googleSearchSupportModel: Schema.array(Schema.string()).default([
            'gemini-2.0'
        ])
    })
]).i18n({
    'zh-CN': require('./locales/zh-CN.schema.yml'),
    'en-US': require('./locales/en-US.schema.yml')
}) as Schema<Config>

export const inject = ['chatluna']

export const name = 'chatluna-openai-like-adapter'
