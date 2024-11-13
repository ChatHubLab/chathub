import { Awaitable, Computed, Schema } from 'koishi'

export interface Config {
    botName: string
    isNickname: boolean
    allowPrivate: boolean
    isForwardMsg: boolean
    allowChatWithRoomName: boolean
    msgCooldown: number
    randomReplyFrequency: number
    messageCount: number
    isLog: boolean

    isReplyWithAt: boolean
    proxyAddress: string
    isProxy: boolean
    outputMode: string
    sendThinkingMessage: boolean
    sendThinkingMessageTimeout: number
    showThoughtMessage: boolean
    splitMessage: boolean
    blackList: Computed<Awaitable<boolean>>
    censor: boolean
    autoDelete: boolean
    autoDeleteTimeout: number

    longMemory: boolean
    privateChatWithoutCommand: boolean
    allowAtReply: boolean
    streamResponse: boolean

    defaultEmbeddings: string
    defaultVectorStore: string

    defaultChatMode: string
    defaultModel: string
    defaultPreset: string

    autoCreateRoomFromUser: boolean

    authUserDefaultGroup: Computed<Awaitable<[number, number, string]>>
    authSystem: boolean

    voiceSpeakId: number

    longMemorySimilarity: number
    longMemoryInterval: number
    longMemoryExtractModel: string
}

export const Config: Schema<Config> = Schema.intersect([
    Schema.object({
        botName: Schema.string().default('香草'),
        isNickname: Schema.boolean().default(true)
    }),

    Schema.object({
        allowPrivate: Schema.boolean().default(true),
        allowAtReply: Schema.boolean().default(true),
        isReplyWithAt: Schema.boolean().default(false),
        isForwardMsg: Schema.boolean().default(false),
        privateChatWithoutCommand: Schema.boolean().default(true),
        allowChatWithRoomName: Schema.boolean().default(false),
        randomReplyFrequency: Schema.percent()
            .min(0)
            .max(1)
            .step(0.01)
            .default(0)
    }),

    Schema.object({
        sendThinkingMessage: Schema.boolean().default(true),
        sendThinkingMessageTimeout: Schema.number().default(15000),
        msgCooldown: Schema.number().min(0).max(3600).step(1).default(0),
        showThoughtMessage: Schema.boolean().default(false)
    }),

    Schema.object({
        outputMode: Schema.dynamic('output-mode').default('text'),
        splitMessage: Schema.boolean().default(false),
        censor: Schema.boolean().default(false),
        streamResponse: Schema.boolean().default(false)
    }),

    Schema.object({
        blackList: Schema.union([Schema.boolean(), Schema.any().hidden()])
            .role('computed')
            .default(false)
    }),

    Schema.object({
        longMemory: Schema.dynamic('long-memory').default(false),
        longMemorySimilarity: Schema.percent()
            .min(0)
            .max(1)
            .step(0.01)
            .default(0.3),
        longMemoryInterval: Schema.number().default(3).min(1).max(10),
        longMemoryExtractModel: Schema.dynamic('model').default('无')
    }),

    Schema.object({
        messageCount: Schema.number()
            .role('slider')
            .min(2)
            .max(100)
            .step(1)
            .default(40),
        autoDelete: Schema.boolean().default(false),
        autoDeleteTimeout: Schema.number()
            .default(86400 * 10)
            .min(86400)
    }),

    Schema.object({
        defaultEmbeddings: Schema.dynamic('embeddings').default('无'),
        defaultVectorStore: Schema.dynamic('vector-store').default('无')
    }),

    Schema.object({
        autoCreateRoomFromUser: Schema.boolean().default(false),
        defaultChatMode: Schema.dynamic('chat-mode').default('chat'),
        defaultModel: Schema.dynamic('model').default('无'),
        defaultPreset: Schema.dynamic('preset').default('chatgpt')
    }),

    Schema.object({
        authSystem: Schema.boolean().experimental().hidden().default(false),
        isProxy: Schema.boolean().default(false),
        voiceSpeakId: Schema.number().default(0),
        isLog: Schema.boolean().default(false)
    }),

    Schema.union([
        Schema.object({
            isProxy: Schema.const(true).required(),
            proxyAddress: Schema.string().default('')
        }),
        Schema.object({})
    ]),

    Schema.union([
        Schema.object({
            authSystem: Schema.const(true).required(),
            authUserDefaultGroup: Schema.union([
                Schema.tuple([
                    Schema.number().default(0),
                    Schema.number().default(1.0),
                    Schema.string().default('guest')
                ]),
                Schema.any().hidden()
            ])
                .role('computed')
                .default([0, 1.0, 'guest'])
        }),
        Schema.object({})
    ])
]).i18n({
    'zh-CN': require('./locales/zh-CN.schema'),
    'en-US': require('./locales/en-US.schema')
}) as Schema<Config>
