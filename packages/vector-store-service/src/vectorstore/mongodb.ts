import { MongoDBAtlasVectorSearch } from '@langchain/mongodb'
import { Context, Logger } from 'koishi'
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'
import { createLogger } from 'koishi-plugin-chatluna/utils/logger'
import { Config } from '..'
import { ChatLunaSaveableVectorStore } from 'koishi-plugin-chatluna/llm-core/model/base'
import { MongoClient, ObjectId } from 'mongodb'

let logger: Logger

export async function apply(
    ctx: Context,
    config: Config,
    plugin: ChatLunaPlugin
) {
    logger = createLogger(ctx, 'chatluna-vector-store-service')

    if (!config.vectorStore.includes('mongodb')) {
        return
    }

    await importMongoDB()

    plugin.registerVectorStore('mongodb', async (params) => {
        const embeddings = params.embeddings

        const client = new MongoClient(config.mongodbUrl)
        await client.connect()

        ctx.on('dispose', async () => {
            await client.close()
            logger.info('MongoDB connection closed')
        })

        const collection = client
            .db(config.mongodbDbName)
            .collection(config.mongodbCollectionName)

        const vectorStore = new MongoDBAtlasVectorSearch(embeddings, {
            collection,
            indexName: params.key ?? 'vector_index',
            textKey: 'text',
            embeddingKey: 'embedding'
        })

        const wrapperStore =
            new ChatLunaSaveableVectorStore<MongoDBAtlasVectorSearch>(
                vectorStore,
                {
                    async deletableFunction(_store, options) {
                        if (options.deleteAll) {
                            await collection.deleteMany({})
                            return
                        }

                        const ids: string[] = []
                        if (options.ids) {
                            ids.push(...options.ids)
                        }

                        if (options.documents) {
                            const documentIds = options.documents
                                ?.map(
                                    (document) =>
                                        document.metadata?.raw_id as
                                            | string
                                            | undefined
                                )
                                .filter((id): id is string => id != null)

                            ids.push(...documentIds)
                        }

                        if (ids.length > 0) {
                            await collection.deleteMany({
                                _id: { $in: ids.map((id) => new ObjectId(id)) }
                            })
                        }
                    },
                    async addDocumentsFunction(
                        store,
                        documents,
                        options: { ids?: string[] }
                    ) {
                        let keys = options?.ids ?? []

                        keys = documents.map((document, i) => {
                            const id = keys[i] ?? crypto.randomUUID()
                            document.metadata = {
                                ...document.metadata,
                                raw_id: id
                            }
                            return id
                        })

                        await store.addDocuments(documents)
                    },
                    async saveableFunction(_store) {
                        await client.close()
                        logger.info('MongoDB connection closed during save')
                    }
                }
            )

        return wrapperStore
    })
}

async function importMongoDB() {
    try {
        const { MongoClient } = await import('mongodb')
        return { MongoClient }
    } catch (err) {
        logger.error(err)
        throw new Error(
            'Please install mongodb as a dependency with, e.g. `npm install -S mongodb`'
        )
    }
}
