import { Tool } from 'langchain/tools';
import { BasePlatformClient, PlatformEmbeddingsClient, PlatformModelClient } from './client';
import { ChatHubChatModel, ChatHubEmbeddings } from './model';
import { ChatHubChainInfo, CreateChatHubLLMChainParams, CreateToolFunction, CreateToolParams, CreateVectorStoreRetrieverFunction, ModelInfo, ModelType, PlatformClientNames } from './types';
import AwaitEventEmitter from 'await-event-emitter';
import { record } from 'zod';
import { ClientConfig, ClientConfigPool } from './config';
import { Context } from 'koishi';
import { ChatHubLLMChain } from '../chain/base';

export class PlatformService {
    private static _platformClients: Map<ClientConfig, BasePlatformClient> = new Map()
    private static _createClientFunctions: Record<string, (ctx: Context, config: ClientConfig) => BasePlatformClient> = {}
    private static _configPools: Record<string, ClientConfigPool> = {}
    private static _tools: Record<string, Tool> = {}
    private static _toolCreators: Record<string, CreateToolFunction> = {}
    private static _models: Record<string, ModelInfo[]> = {}
    private static _chatChains: Record<string, ChatHubChainInfo> = {}
    private static _vectorStoreRetrievers: Record<string, CreateVectorStoreRetrieverFunction> = {}
    private static _eventEmitter = new AwaitEventEmitter()

    constructor(public ctx: Context) {

    }

    registerClient(name: PlatformClientNames, createClientFunction: (ctx: Context, config: ClientConfig) => BasePlatformClient) {
        PlatformService._createClientFunctions[name] = createClientFunction
        return () => this.unregisterClient(name)
    }

    registerConfigPool(name: string, configPool: ClientConfigPool) {
        PlatformService._configPools[name] = configPool
    }

    registerTool(name: string, toolCreator: CreateToolFunction) {
        PlatformService._toolCreators[name] = toolCreator
        return () => this.unregisterTool(name)
    }

    unregisterTool(name: string) { 
        delete PlatformService._toolCreators[name]
    }

    unregisterClient(name: PlatformClientNames) { 
        delete PlatformService._createClientFunctions[name]
        delete PlatformService._configPools[name]
        delete PlatformService._models[name]
        delete PlatformService._platformClients[name]
    }

    unregisterVectorStoreRetriever(name: string) { 
        delete PlatformService._vectorStoreRetrievers[name]
    }

    async registerVectorStoreRetriever(name: string, vectorStoreRetrieverCreator: CreateVectorStoreRetrieverFunction) {
        PlatformService._vectorStoreRetrievers[name] = vectorStoreRetrieverCreator
        await PlatformService.emit('vector-store-retriever-added', this, name)
        return () => this.unregisterVectorStoreRetriever(name)
    }

    async registerChatChain(name: string, description: string, createChatChainFunction: (params: CreateChatHubLLMChainParams) => Promise<ChatHubLLMChain>) {
        PlatformService._chatChains[name] = {
            name,
            description,
            createFunction: createChatChainFunction
        }
        await PlatformService.emit('chat-chain-added', this, PlatformService._chatChains[name])
        return () => this.unregisterChatChain(name)
    }

    unregisterChatChain(name: string) {
        delete PlatformService._chatChains[name]
    }

    getModels(platform: PlatformClientNames, type: ModelType) {
        return PlatformService._models[platform]?.filter(m => type === ModelType.all || m.type === type) ?? []
    }

    getAllModels(type: ModelType) {
        const allModel = []

        for (const platform in PlatformService._models) {
            const models = PlatformService._models[platform]

            for (const model of models) {
                if (type === ModelType.all || model.type === type) {
                    allModel.push(model)
                }
            }
        }

        return allModel
    }

    async createVectorStoreRetriever(name: string, params: CreateToolParams) {
        let vectorStoreRetriever = PlatformService._vectorStoreRetrievers[name]

        if (!vectorStoreRetriever) {
            throw new Error(`Vector store retriever ${name} not found`)
        }

        return await vectorStoreRetriever(params)

    }


    async randomClient(platform: PlatformClientNames) {
        const config = PlatformService._configPools[platform].getConfig()

        if (!config) {
            return null
        }

        const client = await this.getClient(config.value)

        return client
    }

    async getClient(config: ClientConfig) {
        return PlatformService._platformClients.get(config) ?? await this.createClient(config.platform, config)
    }

    async createClient(platform: PlatformClientNames, config: ClientConfig) {
        const createClientFunction = PlatformService._createClientFunctions[platform]

        if (!createClientFunction) {
            throw new Error(`Create client function ${platform} not found`)
        }

        const client = createClientFunction(this.ctx, config)

        const isAvailable = await client.isAvailable()

        const pool = PlatformService._configPools[platform]

        await pool.markConfigStatus(config, isAvailable)

        if (!isAvailable) {
            return null
        }

        const models = await client.getModels()

        const availableModels = PlatformService._models[platform] ?? []


        // filter existing models
        PlatformService._models[platform] = availableModels
            .concat(
                models
                    .filter(m => !availableModels.some(am => am.name === m.name))
            )

        if (client instanceof PlatformModelClient) {
            await PlatformService.emit('model-added', this, platform, client)
        } else if (client instanceof PlatformEmbeddingsClient) {
            await PlatformService.emit('embeddings-added', this, platform, client)
        }

        return client
    }

    async createClients(platform: PlatformClientNames) {
        const configPool = PlatformService._configPools[platform]

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
            PlatformService._platformClients.set(config.value, client)
        }

        return clients
    }

    async createTool(name: string, params: CreateToolParams) {
        let tool = PlatformService._tools[name]

        if (tool) {
            return tool
        }

        let toolCreator = PlatformService._toolCreators[name]

        if (!toolCreator) {
            throw new Error(`Tool ${name} not found`)
        }

        tool = await toolCreator(params)

        PlatformService._tools[name] = tool

        return tool
    }

    static on<T extends keyof PlatformServiceEvents>(eventName: T, func: PlatformServiceEvents[T]) {
        PlatformService._eventEmitter.on(eventName, func)
    }

    static emit<T extends keyof PlatformServiceEvents>(eventName: T, ...args: Parameters<PlatformServiceEvents[T]>) {
        return PlatformService._eventEmitter.emit(eventName, ...args)
    }

}


interface PlatformServiceEvents {
    'chat-chain-added': (service: PlatformService, chain: ChatHubChainInfo) => Promise<void>
    'model-added': (service: PlatformService, platform: PlatformClientNames, client: PlatformModelClient | PlatformModelClient[]) => Promise<void>
    'embeddings-added': (service: PlatformService, platform: PlatformClientNames, client: PlatformEmbeddingsClient | PlatformEmbeddingsClient[]) => Promise<void>
    'vector-store-retriever-added': (service: PlatformService, name: string) => Promise<void>
}
