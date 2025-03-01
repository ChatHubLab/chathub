/* eslint-disable max-len */
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'
import { Context, Logger, Schema } from 'koishi'
import { plugin as plugins } from './plugin'
import { createLogger } from 'koishi-plugin-chatluna/utils/logger'

export let logger: Logger
export function apply(ctx: Context, config: Config) {
    const plugin = new ChatLunaPlugin(ctx, config, 'plugin-common', false)
    logger = createLogger(ctx, 'chatluna-plugin-common')

    ctx.on('ready', async () => {
        plugin.registerToService()
        await plugins(ctx, config, plugin)
    })
}

export interface Config extends ChatLunaPlugin.Config {
    request: boolean
    requestMaxOutputLength: number
    fs: boolean
    fsScopePath: string
    bilibili: boolean
    bilibiliTempTimeout: number
    group: boolean
    groupScopeSelector: string[]
    command: boolean
    commandList: {
        command: string
        description: string
        selector: string[]
        confirm: boolean
    }[]
    chat: boolean
    think: boolean
    cron: boolean
    send: boolean
    draw: boolean
    music: boolean
    actions: boolean
    drawPrompt: string
    drawCommand: string
    codeSandbox: boolean
    codeSandboxAPIKey: string
    knowledge: boolean
    knowledgeId: string[]
    thinkModel: string
    actionsList: {
        name: string
        description: string
        openAPISpec: string
        headers: Record<string, string>
        selector: string[]
    }[]
}

export const Config: Schema<Config> = Schema.intersect([
    ChatLunaPlugin.Config,
    Schema.object({
        request: Schema.boolean().default(true),
        fs: Schema.boolean().default(false),
        group: Schema.boolean().default(false),
        command: Schema.boolean().default(false),
        chat: Schema.boolean().default(false),
        think: Schema.boolean().default(false),
        cron: Schema.boolean().default(false),
        send: Schema.boolean().default(false),
        draw: Schema.boolean().default(false),
        codeSandbox: Schema.boolean().default(false),
        actions: Schema.boolean().default(false),
        knowledge: Schema.boolean().default(false),
        music: Schema.boolean().default(false)
    }),
    Schema.union([
        Schema.object({
            request: Schema.const(true).required(),
            requestMaxOutputLength: Schema.number()
                .min(500)
                .max(8600)
                .default(2000)
        }),
        Schema.object({})
    ]),
    Schema.union([
        Schema.object({
            fs: Schema.const(true).required(),
            fsScopePath: Schema.string().default('')
        }),
        Schema.object({})
    ]),
    Schema.union([
        Schema.object({
            think: Schema.const(true).required(),
            thinkModel: Schema.dynamic('model')
        }),
        Schema.object({})
    ]),
    Schema.union([
        Schema.object({
            command: Schema.const(true).required(),
            commandList: Schema.array(
                Schema.object({
                    command: Schema.string(),
                    description: Schema.string(),
                    selector: Schema.array(Schema.string()).role('table'),
                    confirm: Schema.boolean().default(true)
                })
            )
        }),
        Schema.object({})
    ]),
    Schema.union([
        Schema.object({
            codeSandbox: Schema.const(true).required(),
            codeSandboxAPIKey: Schema.string()
        }),
        Schema.object({})
    ]),
    Schema.union([
        Schema.object({
            group: Schema.const(true).required(),
            groupScopeSelector: Schema.array(Schema.string())
        }),
        Schema.object({})
    ]),
    Schema.union([
        Schema.object({
            draw: Schema.const(true).required(),
            drawPrompt: Schema.string()
                .role('textarea')
                .default(
                    `1girl, solo, female only, full body, masterpiece, highly detailed, game CG, spring, cherry blossoms, floating sakura, beautiful sky, park, extremely delicate and beautiful girl, high school girl, black blazer jacket, plaid skirt\nshort_hair, blunt_bangs, white_hair/pink_eyes, two-tone hair, gradient hair, by Masaaki Sasamoto, best quality, masterpiece, highres, red-eyeshadow, lipstick.`
                ),
            drawCommand: Schema.string().default('nai {prompt}')
        }),
        Schema.object({})
    ]),
    Schema.union([
        Schema.object({
            knowledge: Schema.const(true).required(),
            knowledgeId: Schema.array(Schema.string())
        }),
        Schema.object({})
    ]),
    Schema.union([
        Schema.object({
            actions: Schema.const(true).required(),
            actionsList: Schema.array(
                Schema.object({
                    name: Schema.string(),
                    description: Schema.string(),
                    headers: Schema.dict(String).default({}).role('table'),
                    selector: Schema.array(Schema.string())
                        .default([])
                        .role('table'),
                    openAPISpec: Schema.string().role('textarea')
                })
            ).role('table')
        }),
        Schema.object({})
    ])
]).i18n({
    'zh-CN': require('./locales/zh-CN.schema.yml'),
    'en-US': require('./locales/en-US.schema.yml')
}) as Schema<Config>

export const inject = ['chatluna']

export const name = 'chatluna-plugin-common'
