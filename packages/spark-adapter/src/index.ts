import { Context, Schema } from 'koishi'
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'
import { SparkClient } from './client'
import { SparkClientConfig } from './types'

export function apply(ctx: Context, config: Config) {
    const plugin = new ChatLunaPlugin<SparkClientConfig, Config>(
        ctx,
        config,
        'spark'
    )

    ctx.on('ready', async () => {
        await plugin.registerToService()

        await plugin.parseConfig((config) => {
            return config.appConfigs.map(([appId, apiSecret, apiKey]) => {
                return {
                    apiKey,
                    appId,
                    apiSecret,
                    apiEndpoint: undefined,
                    platform: 'spark',
                    chatLimit: config.chatTimeLimit,
                    timeout: config.timeout,
                    maxRetries: config.maxRetries,
                    concurrentMaxSize: config.chatConcurrentMaxSize
                }
            })
        })

        await plugin.registerClient(
            (_, clientConfig) =>
                new SparkClient(ctx, config, clientConfig, plugin)
        )

        await plugin.initClients()
    })
}

export interface Config extends ChatLunaPlugin.Config {
    appConfigs: [string, string, string][]
    maxTokens: number
    temperature: number
    assistants: [string, string][]
}

export const Config: Schema<Config> = Schema.intersect([
    ChatLunaPlugin.Config,
    Schema.object({
        appConfigs: Schema.array(
            Schema.tuple([
                Schema.string().description('讯飞星火应用的 APP ID').required(),
                Schema.string()
                    .role('secret')
                    .description('讯飞星火应用的 API Secret')
                    .required(),
                Schema.string()
                    .description('讯飞星火应用配置的 API Key')
                    .role('secret')
                    .required()
            ])
        ).description('讯飞星火平台配置 (API Id,API Secret,API Key)'),
        assistants: Schema.array(
            Schema.tuple([
                Schema.string().description('讯飞星火助手的名称').required(),
                Schema.string()
                    .role('secret')
                    .description('讯飞星火助手的链接')
                    .required()
            ])
        ).description(
            '讯飞星火助手配置 (名称,API 链接) (如了星火助手，则不要在上方的配置填入多个 API KEY，只能填入和星火助手绑定的应用的 API KEY，否则可能导致找不到相关助手）'
        )
    }).description('请求设置'),

    Schema.object({
        maxTokens: Schema.number()
            .description(
                '回复的最大 Token 数（16~8192，必须是16的倍数）（注意如果你目前使用的模型的最大 Token 为 8000 及以上的话才建议设置超过 1024 token）'
            )
            .min(16)
            .max(16000)
            .step(16)
            .default(1024),
        temperature: Schema.percent()
            .description('回复温度，越高越随机')
            .min(0.1)
            .max(1)
            .step(0.01)
            .default(0.8)
    }).description('模型设置')
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
]) as any

export const inject = ['chatluna']

export const name = 'chatluna-spark-adapter'
