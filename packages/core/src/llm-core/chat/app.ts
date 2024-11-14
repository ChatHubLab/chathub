import { BaseChatMessageHistory } from '@langchain/core/chat_history'
import { Embeddings } from '@langchain/core/embeddings'
import { ChainValues } from '@langchain/core/utils/types'
import { Context } from 'koishi'
import { parseRawModelName } from 'koishi-plugin-chatluna/llm-core/utils/count_tokens'
import { BufferMemory } from 'koishi-plugin-chatluna/llm-core/memory/langchain'
import { logger } from 'koishi-plugin-chatluna'
import { ConversationRoom } from '../../types'
import {
    ChatLunaError,
    ChatLunaErrorCode
} from 'koishi-plugin-chatluna/utils/error'
import { ChatHubLLMCallArg, ChatHubLLMChainWrapper } from '../chain/base'
import { KoishiChatMessageHistory } from 'koishi-plugin-chatluna/llm-core/memory/message'
import { emptyEmbeddings } from 'koishi-plugin-chatluna/llm-core/model/in_memory'
import {
    PlatformEmbeddingsClient,
    PlatformModelAndEmbeddingsClient,
    PlatformModelClient
} from 'koishi-plugin-chatluna/llm-core/platform/client'
import {
    ClientConfig,
    ClientConfigWrapper
} from 'koishi-plugin-chatluna/llm-core/platform/config'
import {
    ChatHubBaseEmbeddings,
    ChatLunaChatModel
} from 'koishi-plugin-chatluna/llm-core/platform/model'
import { PlatformService } from 'koishi-plugin-chatluna/llm-core/platform/service'
import { ModelInfo } from 'koishi-plugin-chatluna/llm-core/platform/types'
import { AIMessage, HumanMessage } from '@langchain/core/messages'
import { PresetTemplate } from 'koishi-plugin-chatluna/llm-core/prompt'
import { getMessageContent } from 'koishi-plugin-chatluna/utils/string'
import type { HandlerResult } from '../../utils/types'

export class ChatInterface {
    private _input: ChatInterfaceInput
    private _chatHistory: KoishiChatMessageHistory
    private _chains: Record<string, ChatHubLLMChainWrapper> = {}
    private _embeddings: Embeddings

    private _errorCountsMap: Record<string, number[]> = {}
    private _chatCount = 0

    constructor(
        public ctx: Context,
        input: ChatInterfaceInput
    ) {
        this._input = input
    }

    private async handleChatError(
        error: unknown,
        config: ClientConfigWrapper
    ): Promise<never> {
        const configMD5 = config.md5()

        if (
            error instanceof ChatLunaError &&
            error.errorCode === ChatLunaErrorCode.API_UNSAFE_CONTENT
        ) {
            throw error
        }

        this._errorCountsMap[configMD5] = this._errorCountsMap[configMD5] ?? []
        const errorTimes = this._errorCountsMap[configMD5]

        // Add current error timestamp
        errorTimes.push(Date.now())

        // Keep only recent errors
        if (errorTimes.length > config.value.maxRetries * 3) {
            this._errorCountsMap[configMD5] = errorTimes.slice(
                -config.value.maxRetries * 3
            )
        }

        // Check if we need to disable the config
        const recentErrors = errorTimes.slice(-config.value.maxRetries)
        if (
            recentErrors.length >= config.value.maxRetries &&
            checkRange(recentErrors, 1000 * 60 * 20)
        ) {
            await this.disableConfig(config)
        }

        throw error instanceof ChatLunaError
            ? error
            : new ChatLunaError(ChatLunaErrorCode.UNKNOWN_ERROR, error as Error)
    }

    private async disableConfig(config: ClientConfigWrapper): Promise<void> {
        const configMD5 = config.md5()
        delete this._chains[configMD5]
        delete this._errorCountsMap[configMD5]

        const service = this.ctx.chatluna.platform
        await service.makeConfigStatus(config.value, false)
    }

    async chat(arg: ChatHubLLMCallArg): Promise<ChainValues> {
        const [wrapper, config] = await this.createChatHubLLMChainWrapper()

        try {
            await this.ctx.parallel(
                'chatluna/before-chat',
                arg.conversationId,
                arg.message,
                arg.variables,
                this,
                wrapper
            )

            const additionalArgs = await this._chatHistory.getAdditionalArgs()
            arg.variables = { ...additionalArgs, ...arg.variables }

            const response = await this.processChat(arg, wrapper)

            delete this._errorCountsMap[config.md5()]
            return response
        } catch (error) {
            await this.handleChatError(error, config)
        }
    }

    private async processChat(
        arg: ChatHubLLMCallArg,
        wrapper: ChatHubLLMChainWrapper
    ): Promise<ChainValues> {
        const response = (await wrapper.call(arg)) as {
            message: AIMessage
        } & ChainValues
        this._chatCount++

        // Process response
        this.ctx.parallel(
            'chatluna/after-chat',
            arg.conversationId,
            arg.message,
            response.message as AIMessage,
            { ...arg.variables, chatCount: this._chatCount },
            this,
            wrapper
        )

        // Handle post-processing if needed
        if (arg.postHandler) {
            const handlerResult = await this.handlePostProcessing(arg, response)
            response.message.content = handlerResult.displayContent
            await this._chatHistory.overrideAdditionalArgs(
                handlerResult.variables
            )
        }

        const messageContent = getMessageContent(response.message.content)

        // Update chat history
        if (messageContent.trim().length > 0) {
            await this.chatHistory.addMessage(arg.message)
            await this.chatHistory.addMessage(response.message)
        }

        return response
    }

    private async handlePostProcessing(
        arg: ChatHubLLMCallArg,
        response: { message: AIMessage } & ChainValues
    ): Promise<HandlerResult> {
        logger.debug(`original content: %c`, response.message.content)

        return await arg.postHandler.handler(
            arg.session,
            getMessageContent(response.message.content)
        )
    }

    async createChatHubLLMChainWrapper(): Promise<
        [ChatHubLLMChainWrapper, ClientConfigWrapper]
    > {
        const service = this.ctx.chatluna.platform
        const [llmPlatform, llmModelName] = parseRawModelName(this._input.model)
        const currentLLMConfig = await service.randomConfig(llmPlatform)

        if (this._chains[currentLLMConfig.md5()]) {
            return [this._chains[currentLLMConfig.md5()], currentLLMConfig]
        }

        let embeddings: Embeddings

        let llm: ChatLunaChatModel
        let modelInfo: ModelInfo
        let historyMemory: BufferMemory

        try {
            embeddings = await this._initEmbeddings(service)
        } catch (error) {
            if (error instanceof ChatLunaError) {
                throw error
            }
            throw new ChatLunaError(
                ChatLunaErrorCode.EMBEDDINGS_INIT_ERROR,
                error
            )
        }

        try {
            ;[llm, modelInfo] = await this._initModel(
                service,
                currentLLMConfig.value,
                llmModelName
            )
        } catch (error) {
            if (error instanceof ChatLunaError) {
                throw error
            }
            throw new ChatLunaError(ChatLunaErrorCode.MODEL_INIT_ERROR, error)
        }

        embeddings = (await this._checkChatMode(modelInfo)) ?? embeddings

        try {
            await this._createChatHistory()
        } catch (error) {
            if (error instanceof ChatLunaError) {
                throw error
            }
            throw new ChatLunaError(
                ChatLunaErrorCode.CHAT_HISTORY_INIT_ERROR,
                error
            )
        }

        try {
            historyMemory = this._createHistoryMemory()
        } catch (error) {
            if (error instanceof ChatLunaError) {
                throw error
            }
            throw new ChatLunaError(ChatLunaErrorCode.UNKNOWN_ERROR, error)
        }

        const chatChain = await service.createChatChain(this._input.chatMode, {
            botName: this._input.botName,
            model: llm,
            embeddings,
            historyMemory,
            preset: this._input.preset,
            vectorStoreName: this._input.vectorStoreName
        })

        this._chains[currentLLMConfig.md5()] = chatChain
        this._embeddings = embeddings

        return [chatChain, currentLLMConfig]
    }

    get chatHistory(): BaseChatMessageHistory {
        return this._chatHistory
    }

    get chatMode(): string {
        return this._input.chatMode
    }

    get embeddings(): Embeddings {
        return this._embeddings
    }

    get preset(): Promise<PresetTemplate> {
        return this._input.preset()
    }

    async delete(ctx: Context, room: ConversationRoom): Promise<void> {
        await this.clearChatHistory()

        for (const chain of Object.values(this._chains)) {
            await chain.model.clearContext()
        }

        this._chains = {}

        await ctx.database.remove('chathub_conversation', {
            id: room.conversationId
        })

        await ctx.database.remove('chathub_room', {
            roomId: room.roomId
        })
        await ctx.database.remove('chathub_room_member', {
            roomId: room.roomId
        })
        await ctx.database.remove('chathub_room_group_member', {
            roomId: room.roomId
        })
    }

    async clearChatHistory(): Promise<void> {
        if (this._chatHistory == null) {
            await this._createChatHistory()
        }

        await this.ctx.root.parallel(
            'chatluna/clear-chat-history',
            this._input.conversationId,
            this
        )

        await this._chatHistory.clear()

        for (const chain of Object.values(this._chains)) {
            await chain.model.clearContext()
        }
    }

    private async _initEmbeddings(
        service: PlatformService
    ): Promise<ChatHubBaseEmbeddings> {
        if (
            this._input.chatMode === 'chat' &&
            this._input.longMemory === false
        ) {
            return emptyEmbeddings
        }

        if (this._input.embeddings == null) {
            logger.warn(
                'Embeddings are empty, falling back to fake embeddings. Try check your config.'
            )
            return emptyEmbeddings
        }

        const [platform, modelName] = parseRawModelName(this._input.embeddings)

        logger.info(`init embeddings for %c`, this._input.embeddings)

        const client = await service.randomClient(platform)

        if (client == null || client instanceof PlatformModelClient) {
            logger.warn(
                `Platform ${platform} is not supported, falling back to fake embeddings`
            )
            return emptyEmbeddings
        }

        if (client instanceof PlatformEmbeddingsClient) {
            return client.createModel(modelName)
        } else if (client instanceof PlatformModelAndEmbeddingsClient) {
            const model = client.createModel(modelName)

            if (model instanceof ChatLunaChatModel) {
                logger.warn(
                    `Model ${modelName} is not an embeddings model, falling back to fake embeddings`
                )
                return emptyEmbeddings
            }

            return model
        }
    }

    private async _initModel(
        service: PlatformService,
        config: ClientConfig,
        llmModelName: string
    ): Promise<[ChatLunaChatModel, ModelInfo]> {
        const platform = await service.getClient(config)

        const llmInfo = (await platform.getModels()).find(
            (model) => model.name === llmModelName
        )

        const llmModel = platform.createModel(llmModelName)

        if (llmModel instanceof ChatLunaChatModel) {
            return [llmModel, llmInfo]
        }
    }

    private async _checkChatMode(modelInfo: ModelInfo) {
        if (
            // default check
            (!modelInfo.supportMode?.includes(this._input.chatMode) &&
                // all
                !modelInfo.supportMode?.includes('all')) ||
            // func call with plugin browsing
            (!modelInfo.functionCall && this._input.chatMode === 'plugin')
        ) {
            logger.warn(
                `Chat mode ${this._input.chatMode} is not supported by model ${this._input.model}, falling back to chat mode`
            )

            this._input.chatMode = 'chat'
            const embeddings = emptyEmbeddings

            return embeddings
        }

        return undefined
    }

    private async _createChatHistory(): Promise<BaseChatMessageHistory> {
        if (this._chatHistory != null) {
            return this._chatHistory
        }

        this._chatHistory = new KoishiChatMessageHistory(
            this.ctx,
            this._input.conversationId,
            this._input.maxMessagesCount
        )

        await this._chatHistory.loadConversation()

        return this._chatHistory
    }

    private _createHistoryMemory() {
        return new BufferMemory({
            returnMessages: true,
            inputKey: 'input',
            outputKey: 'output',
            chatHistory: this._chatHistory,
            humanPrefix: 'user',
            aiPrefix: this._input.botName
        })
    }
}

export interface ChatInterfaceInput {
    chatMode: string
    botName?: string
    preset?: () => Promise<PresetTemplate>
    model: string
    embeddings?: string
    vectorStoreName?: string
    conversationId: string
    maxMessagesCount: number
    longMemory: boolean
}

function checkRange(times: number[], delayTime: number) {
    const first = times[0]
    const last = times[times.length - 1]

    return last - first < delayTime
}

declare module 'koishi' {
    interface Events {
        'chatluna/before-chat': (
            conversationId: string,
            message: HumanMessage,
            promptVariables: ChainValues,
            chatInterface: ChatInterface,
            chain: ChatHubLLMChainWrapper
        ) => Promise<void>
        'chatluna/after-chat': (
            conversationId: string,
            sourceMessage: HumanMessage,
            responseMessage: AIMessage,
            promptVariables: ChainValues,
            chatInterface: ChatInterface,
            chain: ChatHubLLMChainWrapper
        ) => Promise<void>
        'chatluna/clear-chat-history': (
            conversationId: string,
            chatInterface: ChatInterface
        ) => Promise<void>
    }
}
