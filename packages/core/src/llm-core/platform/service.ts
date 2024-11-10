import { Context, Dict, sleep } from 'koishi'
import {
    BasePlatformClient,
    PlatformEmbeddingsClient,
    PlatformModelAndEmbeddingsClient,
    PlatformModelClient
} from 'koishi-plugin-chatluna/llm-core/platform/client'
import {
    ClientConfig,
    ClientConfigPool
} from 'koishi-plugin-chatluna/llm-core/platform/config'
import {
    ChatHubChainInfo,
    ChatHubTool,
    CreateChatHubLLMChainParams,
    CreateVectorStoreFunction,
    CreateVectorStoreParams,
    ModelInfo,
    ModelType,
    PlatformClientNames
} from 'koishi-plugin-chatluna/llm-core/platform/types'
import { ChatHubLLMChainWrapper } from '../chain/base'
import { LRUCache } from 'lru-cache'
import { ChatLunaSaveableVectorStore } from 'koishi-plugin-chatluna/llm-core/model/base'

export class PlatformService {
    private _platformClients: Record<string, BasePlatformClient> = {}
    private _createClientFunctions: Record<
        string,
        (ctx: Context, config: ClientConfig) => BasePlatformClient
    > = {}

    private _configPools: Record<string, ClientConfigPool> = {}
    private _tools: Record<string, ChatHubTool> = {}
    private _models: Record<string, ModelInfo[]> = {}
    private _chatChains: Record<string, ChatHubChainInfo> = {}
    private _vectorStore: Record<string, CreateVectorStoreFunction> = {}

    private _tmpVectorStores = new LRUCache<
        string,
        ChatLunaSaveableVectorStore
    >({
        max: 20
    })

    constructor(private ctx: Context) {
        this.ctx.on('chatluna/clear-chat-history', async (conversationId) => {
            this._tmpVectorStores.clear()
        })
    }

    registerClient(
        name: PlatformClientNames,
        createClientFunction: (
            ctx: Context,
            config: ClientConfig
        ) => BasePlatformClient
    ) {
        if (this._createClientFunctions[name]) {
            throw new Error(`Client ${name} already exists`)
        }
        this._createClientFunctions[name] = createClientFunction
        return () => this.unregisterClient(name)
    }

    registerConfigPool(name: string, configPool: ClientConfigPool) {
        if (this._configPools[name]) {
            throw new Error(`Config pool ${name} already exists`)
        }
        this._configPools[name] = configPool
    }

    registerTool(name: string, toolCreator: ChatHubTool) {
        this._tools[name] = toolCreator
        this.ctx.emit('chatluna/tool-updated', this)
        return () => this.unregisterTool(name)
    }

    unregisterTool(name: string) {
        delete this._tools[name]
        this.ctx.emit('chatluna/tool-updated', this)
    }

    unregisterClient(platform: PlatformClientNames) {
        const configPool = this._configPools[platform]

        if (!configPool) {
            throw new Error(`Config pool ${platform} not found`)
        }

        const configs = configPool.getConfigs()

        delete this._models[platform]

        for (const config of configs) {
            const client = this.getClientForCache(config.value)

            if (client == null) {
                continue
            }

            delete this._platformClients[
                this._getClientConfigAsKey(config.value)
            ]

            if (client instanceof PlatformModelClient) {
                this.ctx.emit('chatluna/model-removed', this, platform, client)
            } else if (client instanceof PlatformEmbeddingsClient) {
                this.ctx.emit(
                    'chatluna/embeddings-removed',
                    this,
                    platform,
                    client
                )
            } else if (client instanceof PlatformModelAndEmbeddingsClient) {
                this.ctx.emit(
                    'chatluna/embeddings-removed',
                    this,
                    platform,
                    client
                )
                this.ctx.emit('chatluna/model-removed', this, platform, client)
            }
        }

        delete this._configPools[platform]
        delete this._createClientFunctions[platform]
    }

    unregisterVectorStore(name: string) {
        delete this._vectorStore[name]
        this.ctx.emit('chatluna/vector-store-removed', this, name)
    }

    registerVectorStore(
        name: string,
        vectorStoreRetrieverCreator: CreateVectorStoreFunction
    ) {
        this._vectorStore[name] = vectorStoreRetrieverCreator
        this.ctx.emit('chatluna/vector-store-added', this, name)
        return () => this.unregisterVectorStore(name)
    }

    registerChatChain(
        name: string,
        description: Dict<string>,
        createChatChainFunction: (
            params: CreateChatHubLLMChainParams
        ) => Promise<ChatHubLLMChainWrapper>
    ) {
        this._chatChains[name] = {
            name,
            description,
            createFunction: createChatChainFunction
        }
        this.ctx.emit('chatluna/chat-chain-added', this, this._chatChains[name])
        return () => this.unregisterChatChain(name)
    }

    unregisterChatChain(name: string) {
        const chain = this._chatChains[name]
        delete this._chatChains[name]
        this.ctx.emit('chatluna/chat-chain-removed', this, chain)
    }

    getModels(platform: PlatformClientNames, type: ModelType) {
        const models = this._models[platform] ?? []

        return models
            .filter((m) => type === ModelType.all || m.type === type)
            .sort((a, b) => {
                if (!a?.name || !b?.name) return 0
                return a.name.localeCompare(b.name, undefined, {
                    numeric: true,
                    sensitivity: 'base'
                })
            })
    }

    getTools() {
        return Object.keys(this._tools)
    }

    getConfigs(platform: string) {
        return this._configPools[platform]?.getConfigs() ?? []
    }

    resolveModel(platform: PlatformClientNames, name: string) {
        return this._models[platform]?.find((m) => m.name === name)
    }

    getAllModels(type: ModelType) {
        const allModel: string[] = []

        for (const platform in this._models) {
            const models = this._models[platform]

            for (const model of models) {
                if (type === ModelType.all || model.type === type) {
                    allModel.push(platform + '/' + model.name)
                }
            }
        }

        return allModel.sort()
    }

    getVectorStoreRetrievers() {
        return Object.keys(this._vectorStore)
    }

    getChatChains() {
        return Object.values(this._chatChains)
    }

    makeConfigStatus(config: ClientConfig, isAvailable: boolean) {
        const platform = config.platform
        const pool = this._configPools[platform]

        if (!pool) {
            throw new Error(`Config pool ${platform} not found`)
        }

        return pool.markConfigStatus(config, isAvailable)
    }

    async createVectorStore(name: string, params: CreateVectorStoreParams) {
        const vectorStoreRetriever = this._vectorStore[name]

        if (!vectorStoreRetriever) {
            throw new Error(`Vector store retriever ${name} not found`)
        }

        const key = params.key

        if (key == null) {
            return await vectorStoreRetriever(params)
        }

        const cacheVectorStore = this._tmpVectorStores.get(key)

        if (cacheVectorStore) {
            return cacheVectorStore
        }

        const vectorStore = await vectorStoreRetriever(params)

        this._tmpVectorStores.set(key, vectorStore)
        return vectorStore
    }

    async randomConfig(platform: string, lockConfig: boolean = false) {
        return this._configPools[platform]?.getConfig(lockConfig)
    }

    async randomClient(platform: string, lockConfig: boolean = false) {
        const config = await this.randomConfig(platform, lockConfig)

        if (!config) {
            return undefined
        }

        const client = await this.getClient(config.value)

        return client
    }

    getClientForCache(config: ClientConfig) {
        return this._platformClients[this._getClientConfigAsKey(config)]
    }

    async getClient(config: ClientConfig) {
        return (
            this.getClientForCache(config) ??
            (await this.createClient(config.platform, config))
        )
    }

    async refreshClient(
        client: BasePlatformClient,
        platform: string,
        config: ClientConfig
    ) {
        const isAvailable = await client.isAvailable()

        const pool = this._configPools[platform]

        await pool.markConfigStatus(config, isAvailable)

        if (!isAvailable) {
            return undefined
        }

        const models = await client.getModels()

        if (models == null) {
            await pool.markConfigStatus(config, false)

            return undefined
        }

        const availableModels = this._models[platform] ?? []

        await sleep(1)
        // filter existing models
        this._models[platform] = availableModels.concat(
            models.filter(
                (m) => !availableModels.some((am) => am.name === m.name)
            )
        )

        if (client instanceof PlatformModelClient) {
            this.ctx.emit('chatluna/model-added', this, platform, client)
        } else if (client instanceof PlatformEmbeddingsClient) {
            this.ctx.emit('chatluna/embeddings-added', this, platform, client)
        } else if (client instanceof PlatformModelAndEmbeddingsClient) {
            this.ctx.emit('chatluna/embeddings-added', this, platform, client)
            this.ctx.emit('chatluna/model-added', this, platform, client)
        }
    }

    async createClient(platform: string, config: ClientConfig) {
        const createClientFunction = this._createClientFunctions[platform]

        if (!createClientFunction) {
            throw new Error(`Create client function ${platform} not found`)
        }

        const client = createClientFunction(this.ctx, config)

        await this.refreshClient(client, platform, config)

        return client
    }

    async createClients(platform: string) {
        const configPool = this._configPools[platform]

        if (!configPool) {
            throw new Error(`Config pool ${platform} not found`)
        }

        const configs = configPool.getConfigs()

        const clients: BasePlatformClient[] = []

        for (const config of configs) {
            const client = await this.createClient(platform, config.value)

            if (client == null) {
                continue
            }

            clients.push(client)
            this._platformClients[this._getClientConfigAsKey(config.value)] =
                client
        }

        return clients
    }

    getTool(name: string) {
        return this._tools[name]
    }

    createChatChain(name: string, params: CreateChatHubLLMChainParams) {
        const chatChain = this._chatChains[name]

        if (!chatChain) {
            throw new Error(`Chat chain ${name} not found`)
        }

        return chatChain.createFunction(params)
    }

    private _getClientConfigAsKey(config: ClientConfig) {
        return `${config.platform}/${config.apiKey}/${config.apiEndpoint}/${config.maxRetries}/${config.concurrentMaxSize}/${config.timeout}`
    }
}

declare module 'koishi' {
    interface Events {
        'chatluna/chat-chain-added': (
            service: PlatformService,
            chain: ChatHubChainInfo
        ) => void
        'chatluna/model-added': (
            service: PlatformService,
            platform: PlatformClientNames,
            client: BasePlatformClient | BasePlatformClient[]
        ) => void
        'chatluna/embeddings-added': (
            service: PlatformService,
            platform: PlatformClientNames,
            client: BasePlatformClient | BasePlatformClient[]
        ) => void
        'chatluna/vector-store-added': (
            service: PlatformService,
            name: string
        ) => void
        'chatluna/chat-chain-removed': (
            service: PlatformService,
            chain: ChatHubChainInfo
        ) => void
        'chatluna/model-removed': (
            service: PlatformService,
            platform: PlatformClientNames,
            client: BasePlatformClient
        ) => void
        'chatluna/vector-store-removed': (
            service: PlatformService,
            name: string
        ) => void
        'chatluna/embeddings-removed': (
            service: PlatformService,
            platform: PlatformClientNames,
            client: BasePlatformClient | BasePlatformClient[]
        ) => void
        'chatluna/tool-updated': (service: PlatformService) => void
    }
}
