import { Config, logger } from '..'
import { VectorStoreRetriever } from '@langchain/core/vectorstores'
import { ChatLunaSaveableVectorStore } from 'koishi-plugin-chatluna/llm-core/model/base'
import { Context } from 'koishi'
import {
    EnhancedMemory,
    MemoryRetrievalLayerInfo,
    MemoryRetrievalLayerType
} from '../types'
import {
    filterSimilarMemoryByBM25,
    filterSimilarMemoryByVectorStore
} from './similarity'
import {
    documentToEnhancedMemory,
    enhancedMemoryToDocument,
    isMemoryExpired
} from './memory'
import { parseRawModelName } from 'koishi-plugin-chatluna/llm-core/utils/count_tokens'
import { ScoreThresholdRetriever } from 'koishi-plugin-chatluna/llm-core/retrievers'
import { createHash } from 'crypto'

// Interface for memory retrieval layer
interface MemoryRetrievalLayer {
    // Retrieve memory based on the search content
    retrieveMemory(searchContent: string): Promise<EnhancedMemory[]>
    // Add new memories
    addMemories(memories: EnhancedMemory[]): Promise<void>
    // Initialize the layer
    initialize(): Promise<void>
}

// Base class for memory retrieval layer
export abstract class BaseMemoryRetrievalLayer<
    T extends MemoryRetrievalLayerType = MemoryRetrievalLayerType
> implements MemoryRetrievalLayer
{
    protected vectorStore: ChatLunaSaveableVectorStore
    protected retriever: VectorStoreRetriever<ChatLunaSaveableVectorStore>

    constructor(
        protected ctx: Context,
        protected config: Config,
        public info: MemoryRetrievalLayerInfo<T>
    ) {}

    abstract retrieveMemory(searchContent: string): Promise<EnhancedMemory[]>
    abstract addMemories(memories: EnhancedMemory[]): Promise<void>
    abstract deleteMemories(memoryIds: string[]): Promise<void>
    abstract initialize(): Promise<void>
    abstract clearMemories(): Promise<void>

    async cleanupExpiredMemories(): Promise<void> {}
}

// Standard vector store-based memory retrieval layer
export class VectorStoreMemoryLayer<
    T extends MemoryRetrievalLayerType = MemoryRetrievalLayerType
> extends BaseMemoryRetrievalLayer<T> {
    constructor(
        protected ctx: Context,
        protected config: Config,
        public info: MemoryRetrievalLayerInfo<T>
    ) {
        super(ctx, config, info)

        ctx.setInterval(
            async () => {
                await this.cleanupExpiredMemories()
            },
            1000 * 60 * 5
        )
    }

    async initialize(): Promise<void> {
        const { type: layerType, memoryId, userId } = this.info

        logger?.info(
            `init layer(${layerType}) ${memoryId} for ${userId != null ? `user ${userId}` : `global`}`
        )

        this.retriever = await createVectorStoreRetriever(
            this.ctx,
            this.config,
            memoryId
        )
        this.vectorStore = this.retriever.vectorStore
    }

    async retrieveMemory(searchContent: string): Promise<EnhancedMemory[]> {
        let memory = await this.retriever.invoke(searchContent)

        if (this.config.longMemoryTFIDFThreshold > 0) {
            memory = filterSimilarMemoryByBM25(
                memory,
                searchContent,
                this.config.longMemoryTFIDFThreshold
            )
        }

        return memory
            .map(documentToEnhancedMemory)
            .sort((a, b) => b.importance - a.importance)
    }

    async addMemories(memories: EnhancedMemory[]): Promise<void> {
        if (!this.vectorStore) {
            logger?.warn('Vector store not initialized')
            return
        }

        if (
            this.config.longMemoryDuplicateThreshold < 1 &&
            this.config.longMemoryDuplicateCheck
        ) {
            memories = await filterSimilarMemoryByVectorStore(
                memories,
                this.vectorStore,
                this.config.longMemoryDuplicateThreshold
            )
        }

        if (memories.length === 0) return

        await this.vectorStore.addDocuments(
            memories.map(enhancedMemoryToDocument)
        )

        if (this.vectorStore instanceof ChatLunaSaveableVectorStore) {
            logger?.debug('saving vector store')
            try {
                await this.vectorStore.save()
            } catch (e) {
                console.error(e)
            }
        }
    }

    async clearMemories(): Promise<void> {
        if (!this.vectorStore) {
            return
        }

        await this.vectorStore.delete({ deleteAll: true })
    }

    async deleteMemories(memoryIds: string[]): Promise<void> {
        // 检查向量存储是否支持删除操作
        if (typeof this.vectorStore.delete === 'function') {
            // 删除指定ID的记忆
            await this.vectorStore.delete({ ids: memoryIds })

            // 保存向量存储
            if (this.vectorStore instanceof ChatLunaSaveableVectorStore) {
                await this.vectorStore.save()
            }

            logger?.debug(`Deleted ${memoryIds.length} expired memories`)
        } else {
            logger?.warn('Vector store does not support deletion')
        }
    }

    async cleanupExpiredMemories(): Promise<void> {
        if (!this.vectorStore) {
            return
        }

        try {
            // 获取所有记忆
            const allMemories = await this.vectorStore.similaritySearch(
                'test',
                1000
            )

            // 找出过期的记忆
            const expiredMemoriesIds: string[] = []

            for (const doc of allMemories) {
                const memory = documentToEnhancedMemory(doc)
                if (isMemoryExpired(memory) && doc.metadata?.raw_id) {
                    expiredMemoriesIds.push(doc.metadata.raw_id)
                }
            }

            if (expiredMemoriesIds.length > 0) {
                logger?.info(
                    `Found ${expiredMemoriesIds.length} expired memories to delete`
                )

                await this.deleteMemories(expiredMemoriesIds)
            }
        } catch (e) {
            logger?.error(`Error cleaning up expired memories: ${e}`)
        }
    }
}

async function createVectorStoreRetriever(
    ctx: Context,
    config: Config,
    longMemoryId: string
): Promise<VectorStoreRetriever<ChatLunaSaveableVectorStore>> {
    const [platform, model] = parseRawModelName(
        ctx.chatluna.config.defaultEmbeddings
    )
    const embeddingModel = await ctx.chatluna.createEmbeddings(platform, model)

    const vectorStore = await ctx.chatluna.platform.createVectorStore(
        ctx.chatluna.config.defaultVectorStore,
        {
            embeddings: embeddingModel,
            key: longMemoryId
        }
    )

    const retriever = ScoreThresholdRetriever.fromVectorStore(vectorStore, {
        minSimilarityScore: Math.min(0.1, config.longMemorySimilarity - 0.3), // Finds results with at least this similarity score
        maxK: 50, // The maximum K value to use. Use it based to your chunk size to make sure you don't run out of tokens
        kIncrement: 2, // How much to increase K by each time. It'll fetch N results, then N + kIncrement, then N + kIncrement * 2, etc.,
        searchType: 'mmr'
    })

    return retriever
}

export function sortMemoryRetrievalLayerType(
    a: MemoryRetrievalLayerType,
    b: MemoryRetrievalLayerType
): number {
    const order: MemoryRetrievalLayerType[] = [
        'preset-user',
        'preset',
        'user',
        'global'
    ]
    return order.indexOf(a) - order.indexOf(b)
}

export function createMemoryLayers(
    ctx: Context,
    presetId: string,
    userId: string,
    layerTypes?: MemoryRetrievalLayerType[]
): Promise<VectorStoreMemoryLayer[]> {
    const resolveLongMemoryId = (
        presetId: string,
        userId: string,
        layerType: MemoryRetrievalLayerType
    ) => {
        let hash = createHash('sha256')

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

    layerTypes = layerTypes ?? ctx.chatluna_long_memory.defaultLayerTypes

    return Promise.all(
        layerTypes.map(async (layerType) => {
            const memoryId = resolveLongMemoryId(presetId, userId, layerType)
            const layer = new VectorStoreMemoryLayer(
                ctx,
                ctx.chatluna_long_memory.config,
                {
                    type: layerType,
                    userId,
                    presetId,
                    memoryId
                }
            )

            await layer.initialize()
            return layer
        })
    )
}
