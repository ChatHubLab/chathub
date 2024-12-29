import { Context, Logger } from 'koishi'
import { ChatLunaSaveableVectorStore } from 'koishi-plugin-chatluna/llm-core/model/base'
import { FaissStore } from '@langchain/community/vectorstores/faiss'
import path from 'path'
import fs from 'fs/promises'
import { createLogger } from 'koishi-plugin-chatluna/utils/logger'
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'
import { Config } from '..'
import crypto from 'crypto'

let logger: Logger

export async function apply(
    ctx: Context,
    config: Config,
    plugin: ChatLunaPlugin
) {
    logger = createLogger(ctx, 'chatluna-vector-store-service')

    if (!config.vectorStore.includes('faiss')) {
        return
    }

    await FaissStore.importFaiss()

    plugin.registerVectorStore('faiss', async (params) => {
        const embeddings = params.embeddings
        let faissStore: FaissStore

        const directory = path.join(
            'data/chathub/vector_store/faiss',
            params.key ?? 'chatluna'
        )

        try {
            await fs.access(directory)
        } catch {
            await fs.mkdir(directory, { recursive: true })
        }

        const jsonFile = path.join(directory, 'docstore.json')

        logger.debug(`Loading faiss store from %c`, directory)

        try {
            await fs.access(jsonFile)
            faissStore = await FaissStore.load(directory, embeddings)

            // test the embeddings dimension
            const testEmbedding = await embeddings.embedQuery('test')
            if (testEmbedding.length !== faissStore.index.getDimension()) {
                logger.error(
                    `embeddings dimension mismatch: ${testEmbedding.length} !== ${faissStore.index.getDimension()}. Please check the embeddings.`
                )
                // faissStore = undefined
            }
        } catch (e) {
            faissStore = await FaissStore.fromTexts(
                ['sample'],
                [' '],
                embeddings
            )

            try {
                await faissStore.save(directory)
            } catch (e) {
                logger.error(e)
            }
        }

        if (faissStore == null) {
            throw new Error('failed to load faiss store')
        }

        const wrapperStore = new ChatLunaSaveableVectorStore<FaissStore>(
            faissStore,
            {
                async saveableFunction(store) {
                    await store.save(directory)
                },
                async deletableFunction(store, options) {
                    if (options.deleteAll) {
                        await fs.rm(directory, { recursive: true })
                        return
                    }

                    const ids: string[] = []
                    if (options.ids) {
                        ids.push(...options.ids)
                    }

                    if (options.documents) {
                        const ids = options.documents
                            ?.map((document) => {
                                return document.metadata?.raw_id as
                                    | string
                                    | undefined
                            })
                            .filter((id) => id != null)

                        ids.push(...ids)
                    }

                    if (ids.length > 0) {
                        await store.delete({ ids })
                    }
                },
                async addDocumentsFunction(store, documents, options) {
                    let ids = options?.ids ?? []

                    ids = documents.map((document, i) => {
                        const id = ids[i] ?? crypto.randomUUID()

                        document.metadata = {
                            ...document.metadata,
                            raw_id: id
                        }

                        return id
                    })

                    await store.addDocuments(documents, {
                        ids
                    })
                },
                async freeFunction() {
                    faissStore = undefined
                }
            }
        )

        return wrapperStore
    })
}
