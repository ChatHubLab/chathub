import { Context, Logger } from 'koishi'
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'
import { createLogger } from 'koishi-plugin-chatluna/utils/logger'
import { Config } from '..'
import { ChatLunaSaveableVectorStore } from 'koishi-plugin-chatluna/llm-core/model/base'
import type { Milvus } from '@langchain/community/vectorstores/milvus'

let logger: Logger

export async function apply(
    ctx: Context,
    config: Config,
    plugin: ChatLunaPlugin
) {
    logger = createLogger(ctx, 'chatluna-vector-store-service')

    if (!config.vectorStore.includes('milvus')) {
        return
    }

    const MilvusClass = await importMilvus()

    plugin.registerVectorStore('milvus', async (params) => {
        const embeddings = params.embeddings

        const vectorStore = new MilvusClass(embeddings, {
            collectionName: 'chatluna_collection',
            partitionName: params.key ?? 'chatluna',
            url: config.milvusUrl,
            autoId: false,
            username: config.milvusUsername,
            password: config.milvusPassword,
            textFieldMaxLength: 3000
        })

        logger.debug(`Loading milvus store from %c`, params.key ?? 'chatluna')

        const testVector = await embeddings.embedDocuments(['test'])

        try {
            await vectorStore.ensureCollection(testVector, [
                {
                    pageContent: 'test',
                    metadata: {
                        raw_id: 'z'.repeat(100),
                        source: 'z'.repeat(100)
                    }
                }
            ])

            await vectorStore.ensurePartition()

            await vectorStore.similaritySearchVectorWithScore(testVector[0], 10)
        } catch (e) {
            try {
                await vectorStore.client.releasePartitions({
                    collection_name: 'chatluna_collection',
                    partition_names: [params.key ?? 'chatluna']
                })

                await vectorStore.client.releaseCollection({
                    collection_name: 'chatluna_collection'
                })

                await vectorStore.client.dropPartition({
                    collection_name: 'chatluna_collection',
                    partition_name: params.key ?? 'chatluna'
                })

                await vectorStore.client.dropCollection({
                    collection_name: 'chatluna_collection'
                })

                await vectorStore.ensureCollection(testVector, [
                    {
                        pageContent: 'test',
                        metadata: {
                            raw_id: 'z'.repeat(100),
                            source: 'z'.repeat(100)
                        }
                    }
                ])

                await vectorStore.ensurePartition()
            } catch (e) {
                logger.error(e)
            }
            logger.error(e)
        }

        const wrapperStore = new ChatLunaSaveableVectorStore<Milvus>(
            vectorStore,
            {
                async deletableFunction(store, options) {
                    if (options.deleteAll) {
                        await vectorStore.client.releasePartitions({
                            collection_name: 'chatluna_collection',
                            partition_names: [params.key ?? 'chatluna']
                        })

                        await vectorStore.client.releaseCollection({
                            collection_name: 'chatluna_collection'
                        })

                        await vectorStore.client.dropPartition({
                            collection_name: 'chatluna_collection',
                            partition_name: params.key ?? 'chatluna'
                        })

                        await vectorStore.client.dropCollection({
                            collection_name: 'chatluna_collection'
                        })
                        return
                    }

                    const ids: string[] = []
                    if (options.ids) {
                        ids.push(
                            ...options.ids.map((id) => id.replaceAll('-', '_'))
                        )
                    }

                    if (options.documents) {
                        const documentIds = options.documents
                            ?.map((document) => {
                                const id = document.metadata?.raw_id as
                                    | string
                                    | undefined

                                return id != null
                                    ? id.replaceAll('-', '_')
                                    : undefined
                            })
                            .filter((id): id is string => id != null)

                        ids.push(...documentIds)
                    }

                    if (ids.length < 1) {
                        return
                    }

                    const client = store.client

                    const deleteResp = await client.delete({
                        collection_name: store.collectionName,
                        partition_name: params.key ?? 'chatluna',
                        ids
                    })

                    if (deleteResp.status.error_code !== 'Success') {
                        throw new Error(
                            `Error deleting data with ids: ${JSON.stringify(deleteResp)}`
                        )
                    }
                },
                async addDocumentsFunction(store, documents, options) {
                    let ids = options?.ids ?? []

                    ids = documents.map((document, i) => {
                        const id = ids[i] ?? crypto.randomUUID()

                        document.metadata = {
                            source: 'unknown',
                            ...document.metadata,
                            raw_id: id
                        }

                        return id.replaceAll('-', '_')
                    })

                    await store.addDocuments(documents, {
                        ids
                    })
                }
            }
        )

        return wrapperStore
    })
}

async function importMilvus() {
    try {
        await import('@zilliz/milvus2-sdk-node')

        const store = await import('@langchain/community/vectorstores/milvus')

        return store.Milvus
    } catch (err) {
        logger.error(err)
        throw new Error(
            'Please install milvus as a dependency with, e.g. `npm install -S @zilliz/milvus2-sdk-node`'
        )
    }
}
