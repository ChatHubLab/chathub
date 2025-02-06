import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'
import { Context, Schema } from 'koishi'
import { ZhipuClient } from './client'
import { ZhipuClientConfig } from './types'

export function apply(ctx: Context, config: Config) {
    const plugin = new ChatLunaPlugin<ZhipuClientConfig, Config>(
        ctx,
        config,
        'zhipu'
    )

    ctx.on('ready', async () => {
        plugin.registerToService()

        await plugin.parseConfig((config) => {
            return config.apiKeys.map((apiKey) => {
                return {
                    apiKey,
                    apiEndpoint: '',
                    platform: 'zhipu',
                    chatLimit: config.chatTimeLimit,
                    timeout: config.timeout,
                    maxRetries: config.maxRetries,
                    concurrentMaxSize: config.chatConcurrentMaxSize,
                    webSearch: config.webSearch,
                    codeInterpreter: config.codeInterpreter,
                    retrieval: config.retrieval
                        .filter((item) => item[1])
                        .map((item) => item[0])
                } satisfies ZhipuClientConfig
            })
        })

        plugin.registerClient(
            (_, clientConfig) =>
                new ZhipuClient(ctx, config, clientConfig, plugin)
        )

        await plugin.initClients()
    })
}

export interface Config extends ChatLunaPlugin.Config {
    apiKeys: string[]
    maxTokens: number
    temperature: number
    presencePenalty: number
    knowledgePromptTemplate: string
    frequencyPenalty: number
    retrieval: [string, boolean][]
    codeInterpreter: boolean
    webSearch: boolean
}

export const Config: Schema<Config> = Schema.intersect([
    ChatLunaPlugin.Config,
    Schema.object({
        apiKeys: Schema.array(
            Schema.string().role('secret').required()
        ).default([])
    }),
    Schema.object({
        maxTokens: Schema.number().min(16).max(1024000).step(16).default(4096),
        temperature: Schema.percent().min(0).max(2).step(0.1).default(0.8),
        webSearch: Schema.boolean().default(true),
        retrieval: Schema.array(
            Schema.tuple([Schema.string(), Schema.boolean()])
        ).default([]),
        knowledgePromptTemplate: Schema.string()
            .role('textarea')
            .default(
                `从文档
            """
            {{knowledge}}
            """

            中找问题

            """
            {{question}}
            """

            的答案，找到答案就仅使用文档语句回答问题，找不到答案就用自身知识回答并且告诉用户该信息不是来自文档。
            不要复述问题，直接开始回答`
            ),
        presencePenalty: Schema.number().min(-2).max(2).step(0.1).default(0.2),
        frequencyPenalty: Schema.number().min(-2).max(2).step(0.1).default(0.2)
    })
]).i18n({
    'zh-CN': require('./locales/zh-CN.schema.yml'),
    'en-US': require('./locales/en-US.schema.yml')
}) as Schema<Config>

export const inject = ['chatluna']

export const name = 'chatluna-zhipu-adapter'
