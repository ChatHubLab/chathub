import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'
import { Context, Schema } from 'koishi'
import { DifyClientConfig } from './types'
import { DifyClient } from './client'

export function apply(ctx: Context, config: Config) {
    const plugin = new ChatLunaPlugin<DifyClientConfig, Config>(
        ctx,
        config,
        'dify'
    )

    ctx.on('ready', async () => {
        plugin.registerToService()

        await plugin.parseConfig((config) => {
            return [
                {
                    apiKey: '',
                    apiEndpoint: config.apiURL,
                    platform: 'dify',
                    chatLimit: config.chatTimeLimit,
                    timeout: config.timeout,
                    maxRetries: config.maxRetries,
                    concurrentMaxSize: config.chatConcurrentMaxSize,
                    // mark as Map<string,...>
                    additionalModel: new Map(
                        config.additionalModels.map((model) => [
                            model.workflowName,
                            model
                        ])
                    )
                }
            ]
        })

        plugin.registerClient(
            (_, clientConfig) =>
                new DifyClient(ctx, config, clientConfig, plugin)
        )

        await plugin.initClients()
    })
}

export interface Config extends ChatLunaPlugin.Config {
    apiURL: string

    additionalModels: {
        apiKey: string
        workflowName: string
        workflowType: string
    }[]
    maxTokens: number
    temperature: number
}

export const Config: Schema<Config> = Schema.intersect([
    ChatLunaPlugin.Config,
    Schema.object({
        apiURL: Schema.string().default(''),
        additionalModels: Schema.array(
            Schema.object({
                apiKey: Schema.string().role('secret'),
                workflowName: Schema.string(),
                workflowType: Schema.union([
                    'Agent',
                    'Workflow',
                    'ChatBot'
                ]).default('ChatBot')
            }).role('table')
        ).default([])
    })
]).i18n({
    'zh-CN': require('./locales/zh-CN.schema.yml'),
    'en-US': require('./locales/en-US.schema.yml')
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
}) as any

export const inject = ['chatluna']

export const name = 'chatluna-dify-adapter'
