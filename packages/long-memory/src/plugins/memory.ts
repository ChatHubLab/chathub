import { Context } from 'koishi'
import { HumanMessage } from '@langchain/core/messages'
import crypto from 'crypto'
import {
    Config,
    logger,
    MemoryRetrievalLayerType
} from 'koishi-plugin-chatluna-long-memory'
import { getMessageContent } from 'koishi-plugin-chatluna/utils/string'
import { VectorStoreMemoryLayer } from '../utils/layer'
import {
    extractMemoriesFromChat,
    generateNewQuestion,
    selectChatHistory
} from '../utils/chat-history'
import { enhancedMemoryToDocument, isMemoryExpired } from '../utils/memory'

// 应用函数
export function apply(ctx: Context, config: Config) {
    // 获取或创建记忆层
    function createMemoryLayers(
        message: HumanMessage,
        conversationId: string
    ): Promise<VectorStoreMemoryLayer[]> {
        const defaultLayerTypes = ctx.chatluna_long_memory.defaultLayerTypes

        return Promise.all(
            defaultLayerTypes.map(async (layerType) => {
                const memoryId = resolveLongMemoryId(message, layerType)
                const layer = new VectorStoreMemoryLayer(ctx, config, {
                    type: layerType,
                    userId: message.id,
                    presetId: message.additional_kwargs?.preset as string,
                    memoryId
                })

                await layer.initialize()
                return layer
            })
        )
    }

    // 在聊天前处理长期记忆
    ctx.on(
        'chatluna/before-chat',
        async (conversationId, message, promptVariables, chatInterface) => {
            if (ctx.chatluna_long_memory.defaultLayerTypes.length === 0) {
                return
            }

            const layers = await ctx.chatluna_long_memory.getOrPutMemoryLayers(
                conversationId,
                () => createMemoryLayers(message, conversationId)
            )

            let searchContent =
                (message.additional_kwargs['raw_content'] as string | null) ??
                getMessageContent(message.content)

            if (config.longMemoryNewQuestionSearch) {
                const chatHistory = await selectChatHistory(
                    await chatInterface.chatHistory.getMessages(),
                    message.id,
                    config.longMemoryInterval
                )

                searchContent = await generateNewQuestion(
                    ctx,
                    config,
                    chatHistory,
                    searchContent
                )

                if (searchContent === '[skip]') {
                    logger?.debug(
                        `Don't search long memory for user: ${message.id}`
                    )
                    return
                }
            }

            logger?.debug(`Long memory search: ${searchContent}`)

            // 使用记忆层检索记忆
            const memories = await Promise.all(
                layers.map((layer) => layer.retrieveMemory(searchContent))
            )

            // 过滤掉过期的记忆
            const validMemories = memories
                .flat()
                .filter((memory) => !isMemoryExpired(memory))

            logger?.debug(`Long memory: ${JSON.stringify(validMemories)}`)

            promptVariables['long_memory'] = validMemories.map(
                enhancedMemoryToDocument
            )
        }
    )

    // 在聊天后处理长期记忆
    ctx.on(
        'chatluna/after-chat',
        async (
            conversationId,
            sourceMessage,
            _,
            promptVariables,
            chatInterface
        ) => {
            if (config.longMemoryExtractModel === '无') {
                logger?.warn(
                    'Long memory extract model is not set, skip long memory'
                )
                return undefined
            }

            if (
                !ctx.chatluna_long_memory.defaultLayerTypes.includes(
                    'preset-user'
                )
            ) {
                logger?.warn(
                    'Long memory preset-user layer is not supported, only support preset-user layer'
                )
                return undefined
            }

            const chatCount = promptVariables['chatCount'] as number

            if (chatCount % config.longMemoryInterval !== 0) return undefined

            const chatHistory = await selectChatHistory(
                await chatInterface.chatHistory.getMessages(),
                sourceMessage.id ?? undefined,
                config.longMemoryInterval
            )

            // 提取记忆
            const memories = await extractMemoriesFromChat(
                ctx,
                config,
                chatInterface,
                chatHistory
            )

            if (memories.length === 0) return

            // 添加记忆到记忆层
            await ctx.chatluna_long_memory.addMemories(conversationId, memories)
        }
    )
}

function resolveLongMemoryId(
    message: HumanMessage,

    layerType: MemoryRetrievalLayerType
) {
    const presetId = message.additional_kwargs?.preset as string

    const userId = message.id

    let hash = crypto.createHash('sha256')

    switch (layerType) {
        case 'user':
            hash = hash.update(`${userId}`)
            break
        case 'preset':
            hash = hash.update(`${presetId}`)
            break
        case 'preset-user':
            hash = hash.update(`${presetId}-${userId}`)
            break
        case 'global':
        default:
            hash = hash.update('global')
            break
    }

    const hex = hash.digest('hex')

    return hex
}
