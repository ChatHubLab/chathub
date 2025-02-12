import { RedisVectorStore } from '@langchain/redis'
import { Context, Logger } from 'koishi'
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'
import { createLogger } from 'koishi-plugin-chatluna/utils/logger'
import { Config } from '..'
import { ChatLunaSaveableVectorStore } from 'koishi-plugin-chatluna/llm-core/model/base'

let logger: Logger

export async function apply(
    ctx: Context,
    config: Config,
    plugin: ChatLunaPlugin
) {
    logger = createLogger(ctx, 'chatluna-vector-store-service')

    if (!config.vectorStore.includes('redis')) {
        return
    }

    await importRedis()

    plugin.registerVectorStore('redis', async (params) => {
        const embeddings = params.embeddings

        const client = await createClient(config.redisUrl)

        await client.connect()

        const vectorStore = new RedisVectorStore(embeddings, {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            redisClient: client,
            indexName: params.key ?? 'chatluna'
        })
        const testVector = await embeddings.embedDocuments(['test'])

        try {
            await vectorStore.createIndex(testVector[0].length)
        } catch (e) {
            try {
                await vectorStore.dropIndex(true)
                await vectorStore.createIndex(testVector[0].length)
            } catch (e) {
                logger.error(e)
            }
            logger.error(e)
        }

        try {
            await vectorStore.similaritySearchVectorWithScore(testVector[0], 1)
        } catch (e) {
            try {
                await vectorStore.dropIndex(true)
                await vectorStore.createIndex(testVector[0].length)
            } catch (e) {
                logger.error(e)
            }
            logger.error(e)
        }

        logger.debug(`Loading redis store from %c`, vectorStore.indexName)

        const wrapperStore = new ChatLunaSaveableVectorStore<RedisVectorStore>(
            vectorStore,
            {
                async deletableFunction(store, options) {
                    if (options.deleteAll) {
                        // await vectorStore.dropIndex(true)
                        await client.ft.dropIndex(vectorStore.indexName, {
                            DD: true
                        })

                        return
                    }

                    const ids: string[] = []
                    if (options.ids) {
                        ids.push(...options.ids)
                    }

                    if (options.documents) {
                        const documentIds = options.documents
                            ?.map((document) => {
                                return document.metadata?.raw_id as
                                    | string
                                    | undefined
                            })
                            .filter((id): id is string => id != null)

                        ids.push(...documentIds)
                    }

                    if (ids.length < 1) {
                        return
                    }

                    for (const id of ids) {
                        await client.del(store.keyPrefix + id)
                    }
                },
                async addDocumentsFunction(store, documents, options) {
                    let keys = options?.keys ?? []

                    keys = documents.map((document, i) => {
                        const id = keys[i] ?? crypto.randomUUID()

                        document.metadata = {
                            ...document.metadata,
                            raw_id: id
                        }

                        return store.keyPrefix + id
                    })

                    await store.addDocuments(documents, {
                        keys,
                        batchSize: options?.batchSize
                    })
                },
                async freeFunction() {
                    await client.disconnect()
                },
                async saveableFunction(store) {}
            }
        )

        return wrapperStore
    })
}

async function createClient(url: string) {
    const redis = await importRedis()

    return redis.createClient({
        url
    })
}

async function importRedis() {
    try {
        const any = await import('redis')

        return any
    } catch (err) {
        logger.error(err)
        throw new Error(
            'Please install redis as a dependency with, e.g. `npm install -S redis`'
        )
    }
}
