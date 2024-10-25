import { Context, Logger } from 'koishi'
import { ChatLunaSaveableVectorStore } from 'koishi-plugin-chatluna/llm-core/model/base'
import { FaissStore } from '@langchain/community/vectorstores/faiss'
import path from 'path'
import fs from 'fs/promises'
import { createLogger } from 'koishi-plugin-chatluna/utils/logger'
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'
import { Config } from '..'

let logger: Logger

export async function apply(
    ctx: Context,
    config: Config,
    plugin: ChatLunaPlugin
) {
    logger = createLogger(ctx, 'chatluna-vector-store-service')

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
                faissStore = undefined
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
                async deletableFunction(store) {
                    await fs.rm(directory, { recursive: true })
                }
            }
        )

        return wrapperStore
    })
}
