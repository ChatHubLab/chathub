import { AIMessage, HumanMessage } from '@langchain/core/messages'
import fs from 'fs'
import {
    Awaitable,
    Computed,
    Context,
    Dict,
    Schema,
    Service,
    Session
} from 'koishi'
import { ChatInterface } from 'koishi-plugin-chatluna/llm-core/chat/app'
import path from 'path'
import { LRUCache } from 'lru-cache'
import { v4 as uuidv4 } from 'uuid'
import { Cache } from '../cache'
import { ChatChain } from '../chains/chain'
import { ChatLunaLLMChainWrapper } from 'koishi-plugin-chatluna/llm-core/chain/base'
import { BasePlatformClient } from 'koishi-plugin-chatluna/llm-core/platform/client'
import {
    ClientConfig,
    ClientConfigPool,
    ClientConfigPoolMode
} from 'koishi-plugin-chatluna/llm-core/platform/config'
import {
    ChatHubBaseEmbeddings,
    ChatLunaChatModel
} from 'koishi-plugin-chatluna/llm-core/platform/model'
import { PlatformService } from 'koishi-plugin-chatluna/llm-core/platform/service'
import {
    ChatLunaTool,
    CreateChatLunaLLMChainParams,
    CreateVectorStoreFunction,
    ModelType,
    PlatformClientNames
} from 'koishi-plugin-chatluna/llm-core/platform/types'
import { parseRawModelName } from 'koishi-plugin-chatluna/llm-core/utils/count_tokens'
import { PresetService } from 'koishi-plugin-chatluna/preset'
import { ConversationRoom, Message } from '../types'
import {
    ChatLunaError,
    ChatLunaErrorCode
} from 'koishi-plugin-chatluna/utils/error'
import { RequestIdQueue } from 'koishi-plugin-chatluna/utils/queue'
import { MessageTransformer } from './message_transform'
import { ChatEvents } from './types'
import { chatLunaFetch, ws } from 'koishi-plugin-chatluna/utils/request'
import * as fetchType from 'undici/types/fetch'
import { ClientOptions, WebSocket } from 'ws'
import { ClientRequestArgs } from 'http'
import { Config } from '../config'
import { DefaultRenderer } from '../render'
import type { PostHandler } from '../utils/types'
import { withResolver } from 'koishi-plugin-chatluna/utils/promise'

export class ChatLunaService extends Service {
    private _plugins: Record<string, ChatLunaPlugin> = {}
    private _chatInterfaceWrapper: ChatInterfaceWrapper
    private readonly _chain: ChatChain
    private readonly _keysCache: Cache<'chathub/keys', string>
    private readonly _preset: PresetService
    private readonly _platformService: PlatformService
    private readonly _messageTransformer: MessageTransformer
    private readonly _renderer: DefaultRenderer

    constructor(
        public readonly ctx: Context,
        public config: Config
    ) {
        super(ctx, 'chatluna')
        this._chain = new ChatChain(ctx, config)
        this._keysCache = new Cache(this.ctx, config, 'chathub/keys')
        this._preset = new PresetService(ctx, config, this._keysCache)
        this._platformService = new PlatformService(ctx)
        this._messageTransformer = new MessageTransformer()
        this._renderer = new DefaultRenderer(ctx, config)

        this._createTempDir()
        this._defineDatabase()
    }

    async registerPlugin(plugin: ChatLunaPlugin) {
        const platformName = plugin.platformName

        if (this._plugins[platformName]) {
            throw new ChatLunaError(
                ChatLunaErrorCode.PLUGIN_ALREADY_REGISTERED,
                new Error(`Plugin ${platformName} already registered`)
            )
        }

        this._plugins[platformName] = plugin

        this.logger.success(`register plugin %c`, plugin.platformName)
    }

    async awaitLoadPlatform(
        plugin: ChatLunaPlugin | string,
        timeout: number = 30000
    ) {
        const pluginName =
            typeof plugin === 'string' ? plugin : plugin.platformName
        const { promise, resolve, reject } = withResolver<void>()

        // 提前检测，如果已经加载，则直接返回
        if (
            this._platformService.getModels(pluginName, ModelType.all).length >
            0
        ) {
            resolve()
            return promise
        }

        // 添加超时处理
        const timeoutId = setTimeout(() => {
            dispose()
            reject(
                new Error(`Timeout waiting for platform ${pluginName} to load`)
            )
        }, timeout)

        const dispose = this.ctx.on(
            'chatluna/model-added',
            (service, platform) => {
                if (platform === pluginName) {
                    clearTimeout(timeoutId)
                    resolve()
                    dispose()
                }
            }
        )

        return promise
    }

    unregisterPlugin(plugin: ChatLunaPlugin | string) {
        const platformName =
            typeof plugin === 'string' ? plugin : plugin.platformName

        const targetPlugin = this._plugins[platformName]

        // If not found the plugin, return directly
        /* if (!targetPlugin && withError) {
            throw new ChatLunaError(
                ChatLunaErrorCode.PLUGIN_NOT_FOUND,
                new Error(`Plugin ${platformName} not found`)
            )
        } else */ if (!targetPlugin) {
            return
        }

        const platform = targetPlugin.platformName

        this._chatInterfaceWrapper?.dispose(platform)

        targetPlugin.dispose()

        delete this._plugins[platform]

        this.logger.success('unregister plugin %c', targetPlugin.platformName)
    }

    getPlugin(platformName: string) {
        return this._plugins[platformName]
    }

    chat(
        session: Session,
        room: ConversationRoom,
        message: Message,
        event: ChatEvents,
        stream: boolean = false,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        variables: Record<string, any> = {},
        postHandler?: PostHandler,
        requestId: string = uuidv4()
    ) {
        const chatInterfaceWrapper =
            this._chatInterfaceWrapper ?? this._createChatInterfaceWrapper()

        return chatInterfaceWrapper.chat(
            session,
            room,
            message,
            event,
            stream,
            requestId,
            variables,
            postHandler
        )
    }

    async stopChat(room: ConversationRoom, requestId: string) {
        const chatInterfaceWrapper = this.queryInterfaceWrapper(room, false)

        if (chatInterfaceWrapper == null) {
            return undefined
        }

        return chatInterfaceWrapper.stopChat(requestId)
    }

    queryInterfaceWrapper(room: ConversationRoom, autoCreate: boolean = true) {
        return (
            this._chatInterfaceWrapper ??
            (autoCreate ? this._createChatInterfaceWrapper() : undefined)
        )
    }

    async clearChatHistory(room: ConversationRoom) {
        const chatBridger =
            this._chatInterfaceWrapper ?? this._createChatInterfaceWrapper()

        return chatBridger.clearChatHistory(room)
    }

    getCachedInterfaceWrapper() {
        return this._chatInterfaceWrapper
    }

    async clearCache(room: ConversationRoom) {
        const chatBridger =
            this._chatInterfaceWrapper ?? this._createChatInterfaceWrapper()

        return chatBridger.clearCache(room)
    }

    async createChatModel(platformName: string, model: string) {
        const service = this._platformService

        const client = await service.randomClient(platformName)

        if (client == null) {
            throw new ChatLunaError(
                ChatLunaErrorCode.MODEL_ADAPTER_NOT_FOUND,
                new Error(`The platform ${platformName} no available`)
            )
        }

        return client.createModel(model) as ChatLunaChatModel
    }

    randomChatModel(platformName: string, model: string) {
        return async () => await this.createChatModel(platformName, model)
    }

    async createEmbeddings(platformName: string, modelName: string) {
        const service = this._platformService

        const client = await service.randomClient(platformName)

        if (client == null) {
            throw new ChatLunaError(
                ChatLunaErrorCode.MODEL_ADAPTER_NOT_FOUND,
                new Error(`The platform ${platformName} no available`)
            )
        }

        const model = client.createModel(modelName)

        if (model instanceof ChatHubBaseEmbeddings) {
            return model
        }

        throw new ChatLunaError(
            ChatLunaErrorCode.MODEL_NOT_FOUND,
            new Error(`The model ${modelName} is not embeddings`)
        )
    }

    randomEmbeddings(platformName: string, modelName: string) {
        return async () => await this.createEmbeddings(platformName, modelName)
    }

    get platform() {
        return this._platformService
    }

    get cache() {
        return this._keysCache
    }

    get preset() {
        return this._preset
    }

    get chatChain() {
        return this._chain
    }

    get messageTransformer() {
        return this._messageTransformer
    }

    get renderer() {
        return this._renderer
    }

    protected async stop(): Promise<void> {
        for (const plugin of Object.values(this._plugins)) {
            this.unregisterPlugin(plugin)
        }
        this._chatInterfaceWrapper?.dispose()
        this._platformService.dispose()
    }

    private _createTempDir() {
        // create dir data/chathub/temp use fs
        // ?
        const tempPath = path.resolve(this.ctx.baseDir, 'data/chathub/temp')
        if (!fs.existsSync(tempPath)) {
            fs.mkdirSync(tempPath, { recursive: true })
        }
    }

    private _defineDatabase() {
        const ctx = this.ctx

        ctx.database.extend(
            'chathub_conversation',
            {
                id: {
                    type: 'char',
                    length: 255
                },
                latestId: {
                    type: 'char',
                    length: 255,
                    nullable: true
                },
                additional_kwargs: {
                    type: 'text',
                    nullable: true
                },
                updatedAt: {
                    type: 'timestamp',
                    nullable: false,
                    initial: new Date()
                }
            },
            {
                autoInc: false,
                primary: 'id',
                unique: ['id']
            }
        )

        ctx.database.extend(
            'chathub_message',
            {
                id: {
                    type: 'char',
                    length: 255
                },
                text: 'text',
                parent: {
                    type: 'char',
                    length: 255,
                    nullable: true
                },
                role: {
                    type: 'char',
                    length: 20
                },
                conversation: {
                    type: 'char',
                    length: 255
                },
                additional_kwargs: {
                    type: 'text',
                    nullable: true
                },
                rawId: {
                    type: 'char',
                    length: 255,
                    nullable: true
                }
            },
            {
                autoInc: false,
                primary: 'id',
                unique: ['id']
                /*  foreign: {
                 conversation: ['chathub_conversaion', 'id']
             } */
            }
        )

        ctx.database.extend(
            'chathub_room',
            {
                roomId: {
                    type: 'integer'
                },
                roomName: 'string',
                conversationId: {
                    type: 'char',
                    length: 255,
                    nullable: true
                },

                roomMasterId: {
                    type: 'char',
                    length: 255
                },
                visibility: {
                    type: 'char',
                    length: 20
                },
                preset: {
                    type: 'char',
                    length: 255
                },
                model: {
                    type: 'char',
                    length: 100
                },
                chatMode: {
                    type: 'char',
                    length: 20
                },
                password: {
                    type: 'char',
                    length: 100
                },
                autoUpdate: {
                    type: 'boolean',
                    initial: false
                },
                updatedTime: {
                    type: 'timestamp',
                    nullable: false,
                    initial: new Date()
                }
            },
            {
                autoInc: false,
                primary: 'roomId',
                unique: ['roomId']
            }
        )

        ctx.database.extend(
            'chathub_room_member',
            {
                userId: {
                    type: 'char',
                    length: 255
                },
                roomId: {
                    type: 'integer'
                },
                roomPermission: {
                    type: 'char',
                    length: 50
                },
                mute: {
                    type: 'boolean',
                    initial: false
                }
            },
            {
                autoInc: false,
                primary: ['userId', 'roomId']
            }
        )

        ctx.database.extend(
            'chathub_room_group_member',
            {
                groupId: {
                    type: 'char',
                    length: 255
                },
                roomId: {
                    type: 'integer'
                },
                roomVisibility: {
                    type: 'char',
                    length: 20
                }
            },
            {
                autoInc: false,
                primary: ['groupId', 'roomId']
            }
        )

        ctx.database.extend(
            'chathub_user',
            {
                userId: {
                    type: 'char',
                    length: 255
                },
                defaultRoomId: {
                    type: 'integer'
                },
                groupId: {
                    type: 'char',
                    length: 255,
                    nullable: true
                }
            },
            {
                autoInc: false,
                primary: ['userId', 'groupId']
            }
        )
    }

    private _createChatInterfaceWrapper(): ChatInterfaceWrapper {
        const chatBridger = new ChatInterfaceWrapper(this)
        this._chatInterfaceWrapper = chatBridger
        return chatBridger
    }

    static inject = ['database']
}

export class ChatLunaPlugin<
    R extends ClientConfig = ClientConfig,
    T extends ChatLunaPlugin.Config = ChatLunaPlugin.Config
> {
    private _disposables: (() => void)[] = []

    private _supportModels: string[] = []

    private readonly _platformConfigPool: ClientConfigPool<R>

    private _platformService: PlatformService

    constructor(
        protected ctx: Context,
        public readonly config: T,
        public platformName: PlatformClientNames,
        createConfigPool: boolean = true
    ) {
        ctx.once('dispose', async () => {
            ctx.chatluna.unregisterPlugin(this)
        })

        if (createConfigPool) {
            this._platformConfigPool = new ClientConfigPool<R>(
                ctx,
                config.configMode === 'default'
                    ? ClientConfigPoolMode.AlwaysTheSame
                    : ClientConfigPoolMode.LoadBalancing
            )
        }

        this._platformService = ctx.chatluna.platform
    }

    async parseConfig(f: (config: T) => R[]) {
        const configs = f(this.config)

        for (const config of configs) {
            await this._platformConfigPool.addConfig(config)
        }
    }

    async initClients() {
        this._platformService.registerConfigPool(
            this.platformName,
            this._platformConfigPool
        )

        try {
            await this._platformService.createClients(this.platformName)
        } catch (e) {
            this.ctx.chatluna.unregisterPlugin(this)

            throw e
        }

        this._supportModels = this._supportModels.concat(
            this._platformService
                .getModels(this.platformName, ModelType.llm)
                .map((model) => `${this.platformName}/${model.name}`)
        )
    }

    async initClientsWithPool<A extends ClientConfig = R>(
        platformName: PlatformClientNames,
        pool: ClientConfigPool<A>,
        createConfigFunc: (config: T) => A[]
    ) {
        const configs = createConfigFunc(this.config)

        for (const config of configs) {
            await pool.addConfig(config)
        }

        this._platformService.registerConfigPool(platformName, pool)

        try {
            await this._platformService.createClients(platformName)
        } catch (e) {
            this.ctx.chatluna.unregisterPlugin(this)

            throw e
        }

        this._supportModels = this._supportModels.concat(
            this._platformService
                .getModels(platformName, ModelType.llm)
                .map((model) => `${platformName}/${model.name}`)
        )
    }

    get supportedModels(): readonly string[] {
        return this._supportModels
    }

    dispose() {
        while (this._disposables.length > 0) {
            const disposable = this._disposables.pop()
            disposable()
        }
    }

    registerConfigPool(
        platformName: PlatformClientNames,
        configPool: ClientConfigPool
    ) {
        this._platformService.registerConfigPool(platformName, configPool)
    }

    registerToService() {
        this.ctx.chatluna.registerPlugin(this)
    }

    registerClient(
        func: (
            ctx: Context,
            config: R
        ) => BasePlatformClient<R, ChatHubBaseEmbeddings | ChatLunaChatModel>,
        platformName: string = this.platformName
    ) {
        const disposable = this._platformService.registerClient(
            platformName,
            func
        )

        this._disposables.push(disposable)
    }

    registerVectorStore(name: string, func: CreateVectorStoreFunction) {
        const disposable = this._platformService.registerVectorStore(name, func)
        this._disposables.push(disposable)
    }

    registerTool(name: string, tool: ChatLunaTool) {
        const disposable = this._platformService.registerTool(name, tool)
        this._disposables.push(disposable)
    }

    registerChatChainProvider(
        name: string,
        description: Dict<string>,
        func: (
            params: CreateChatLunaLLMChainParams
        ) => Promise<ChatLunaLLMChainWrapper>
    ) {
        const disposable = this._platformService.registerChatChain(
            name,
            description,
            func
        )
        this._disposables.push(disposable)
    }

    async fetch(info: fetchType.RequestInfo, init?: fetchType.RequestInit) {
        const proxyMode = this.config.proxyMode

        switch (proxyMode) {
            case 'system':
                return chatLunaFetch(info, init)
            case 'off':
                return chatLunaFetch(info, init, 'null')
            case 'on':
                return chatLunaFetch(info, init, this.config.proxyAddress)
            default:
                return chatLunaFetch(info, init)
        }
    }

    ws(url: string, options?: ClientOptions | ClientRequestArgs): WebSocket {
        const proxyMode = this.config.proxyMode

        let webSocket: WebSocket

        switch (proxyMode) {
            case 'system':
                webSocket = ws(url, options)
                break
            case 'off':
                webSocket = ws(url, options, 'null')
                break
            case 'on':
                webSocket = ws(url, options, this.config.proxyAddress)
                break
            default:
                webSocket = ws(url, options)
                break
        }

        this.ctx.effect(() => webSocket.close)

        webSocket.on('error', (err) => {
            this.ctx.logger.error(err)
        })

        return webSocket
    }
}

type ChatHubChatBridgerInfo = {
    chatInterface: ChatInterface
    room: ConversationRoom
}

class ChatInterfaceWrapper {
    private _conversations: LRUCache<string, ChatHubChatBridgerInfo> =
        new LRUCache({
            max: 40
        })

    private _modelQueue = new RequestIdQueue()
    private _conversationQueue = new RequestIdQueue()
    private _platformService: PlatformService

    private _requestIdMap: Map<string, AbortController> = new Map()
    private _platformToConversations: Map<string, string[]> = new Map()

    constructor(private _service: ChatLunaService) {
        this._platformService = _service.platform
    }

    async chat(
        session: Session,
        room: ConversationRoom,
        message: Message,
        event: ChatEvents,
        stream: boolean,
        requestId: string,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        variables: Record<string, any> = {},
        postHandler?: PostHandler
    ): Promise<Message> {
        const { conversationId, model: fullModelName } = room
        const [platform] = parseRawModelName(fullModelName)
        const config = this._platformService.getConfigs(platform)[0]

        try {
            // Add to queues
            await Promise.all([
                this._conversationQueue.add(conversationId, requestId),
                this._modelQueue.add(platform, requestId)
            ])

            const currentQueueLength =
                await this._conversationQueue.getQueueLength(conversationId)
            await event['llm-queue-waiting'](currentQueueLength)

            // Wait for our turn
            await Promise.all([
                this._conversationQueue.wait(conversationId, requestId, 0),
                this._modelQueue.wait(
                    platform,
                    requestId,
                    config.value.concurrentMaxSize
                )
            ])

            // Track conversation
            const conversationIds =
                this._platformToConversations.get(platform) ?? []
            conversationIds.push(conversationId)
            this._platformToConversations.set(platform, conversationIds)

            const { chatInterface } =
                this._conversations.get(conversationId) ??
                (await this._createChatInterface(room))

            const abortController = new AbortController()
            this._requestIdMap.set(requestId, abortController)

            const humanMessage = new HumanMessage({
                content: message.content,
                name: message.name,
                id: session.userId,
                additional_kwargs: {
                    ...message.additional_kwargs,
                    preset: room.preset
                }
            })

            const chainValues = await chatInterface.chat({
                message: humanMessage,
                events: event,
                stream,
                conversationId,
                session,
                variables,
                signal: abortController.signal,
                postHandler
            })

            return {
                content: (chainValues.message as AIMessage).content as string,
                additionalReplyMessages: (
                    chainValues.additionalReplyMessages as string[]
                )?.map((content) => ({ content }))
            }
        } finally {
            // Clean up resources
            await Promise.all([
                this._modelQueue.remove(platform, requestId),
                this._conversationQueue.remove(conversationId, requestId)
            ])
            this._requestIdMap.delete(requestId)
        }
    }

    stopChat(requestId: string) {
        const abortController = this._requestIdMap.get(requestId)
        if (!abortController) {
            return false
        }
        abortController.abort()
        return true
    }

    async query(
        room: ConversationRoom,
        create: boolean = false
    ): Promise<ChatInterface> {
        const { conversationId } = room

        const { chatInterface } = this._conversations.get(conversationId) ?? {}

        if (chatInterface == null && create) {
            return this._createChatInterface(room).then(
                (result) => result.chatInterface
            )
        }

        return chatInterface
    }

    async clearChatHistory(room: ConversationRoom) {
        const { conversationId } = room
        const requestId = uuidv4()

        try {
            await this._conversationQueue.add(conversationId, requestId)
            await this._conversationQueue.wait(conversationId, requestId, 0)

            const chatInterface = await this.query(room, true)
            await chatInterface.clearChatHistory()
            this._conversations.delete(conversationId)
        } finally {
            await this._conversationQueue.remove(conversationId, requestId)
        }
    }

    async clearCache(room: ConversationRoom) {
        const { conversationId } = room
        const requestId = uuidv4()

        try {
            await this._conversationQueue.add(conversationId, requestId)
            await this._conversationQueue.wait(conversationId, requestId, 0)

            const chatInterface = await this.query(room)

            await this._service.ctx.root.parallel(
                'chatluna/clear-chat-history',
                conversationId,
                chatInterface
            )

            return this._conversations.delete(conversationId)
        } finally {
            await this._conversationQueue.remove(conversationId, requestId)
        }
    }

    getCachedConversations(): [string, ChatHubChatBridgerInfo][] {
        return Array.from(this._conversations.entries())
    }

    async delete(room: ConversationRoom) {
        const { conversationId } = room
        const requestId = uuidv4()

        try {
            await this._conversationQueue.add(conversationId, requestId)
            await this._conversationQueue.wait(conversationId, requestId, 1)

            const chatInterface = await this.query(room)
            if (!chatInterface) return

            await chatInterface.delete(this._service.ctx, room)
            await this.clearCache(room)
        } finally {
            await this._conversationQueue.remove(conversationId, requestId)
        }
    }

    dispose(platform?: string) {
        // Terminate all related requests
        for (const controller of this._requestIdMap.values()) {
            controller.abort()
        }

        if (!platform) {
            // Clean up all resources
            this._conversations.clear()
            this._requestIdMap.clear()
            this._platformToConversations.clear()
            return
        }

        // Clean up resources for specific platform
        const conversationIds = this._platformToConversations.get(platform)
        if (!conversationIds?.length) return

        for (const conversationId of conversationIds) {
            this._conversations.delete(conversationId)
            // Terminate platform-related requests
            const controller = this._requestIdMap.get(conversationId)
            if (controller) {
                controller.abort()
                this._requestIdMap.delete(conversationId)
            }
        }

        this._platformToConversations.delete(platform)
    }

    private async _createChatInterface(
        room: ConversationRoom
    ): Promise<ChatHubChatBridgerInfo> {
        const config = this._service.config

        const chatInterface = new ChatInterface(this._service.ctx.root, {
            chatMode: room.chatMode,
            botName: config.botName,
            preset: async () => {
                return await this._service.preset.getPreset(room.preset)
            },
            model: room.model,
            conversationId: room.conversationId,
            embeddings:
                config.defaultEmbeddings && config.defaultEmbeddings.length > 0
                    ? config.defaultEmbeddings
                    : undefined,
            vectorStoreName:
                config.defaultVectorStore &&
                config.defaultVectorStore.length > 0
                    ? config.defaultVectorStore
                    : undefined,
            maxMessagesCount: config.messageCount
        })

        const result = {
            chatInterface,
            room
        }

        this._conversations.set(room.conversationId, result)

        return result
    }
}

// eslint-disable-next-line @typescript-eslint/no-namespace
export namespace ChatLunaPlugin {
    export interface Config {
        chatConcurrentMaxSize?: number
        chatTimeLimit?: Computed<Awaitable<number>>
        timeout?: number
        configMode: string
        maxRetries: number
        proxyMode: string
        proxyAddress: string
    }

    export const Config: Schema<ChatLunaPlugin.Config> = Schema.intersect([
        Schema.object({
            chatConcurrentMaxSize: Schema.number().min(1).max(8).default(3),
            chatTimeLimit: Schema.union([
                Schema.number().min(1).max(2000),
                Schema.any().hidden()
            ])
                .role('computed')
                .default(200),
            configMode: Schema.union([
                Schema.const('default'),
                Schema.const('balance')
            ]).default('default'),
            maxRetries: Schema.number().min(1).max(6).default(3),
            timeout: Schema.number().default(300 * 1000),
            proxyMode: Schema.union([
                Schema.const('system'),
                Schema.const('off'),
                Schema.const('on')
            ]).default('system')
        }),
        Schema.union([
            Schema.object({
                proxyMode: Schema.const('on').required(),
                proxyAddress: Schema.string().default('')
            }),
            Schema.object({})
        ])
    ]).i18n({
        'zh-CN': require('../locales/zh-CN.schema.plugin.yml'),
        'en-US': require('../locales/en-US.schema.plugin.yml')
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
    }) as any
}
