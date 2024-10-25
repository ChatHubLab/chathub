import { Context, Logger } from 'koishi'
import { createLogger } from 'koishi-plugin-chatluna/utils/logger'
import type { Table } from 'vectordb'
import path from 'path'
import fs from 'fs/promises'
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'
import { Config } from '..'
import { ChatLunaSaveableVectorStore } from 'koishi-plugin-chatluna/llm-core/model/base'

let logger: Logger

export async function apply(
    ctx: Context,
    config: Config,
    plugin: ChatLunaPlugin
) {
    logger = createLogger(ctx, 'chatluna-vector-store-service')

    plugin.registerVectorStore('lancedb', async (params) => {
        const embeddings = params.embeddings

        const directory = path.join(
            'data/chathub/vector_store/lancedb',
            params.key ?? 'chatluna'
        )

        try {
            await fs.access(directory)
        } catch {
            await fs.mkdir(directory, { recursive: true })
        }

        logger.debug(`Loading lancedb from %c`, directory)

        const client = await (await importLanceDB()).connect(directory)

        const tableNames = await client.tableNames()

        let table: Table<number[]>

        const testVector = await embeddings.embedDocuments(['test'])

        if (tableNames.some((text) => text === 'vectors')) {
            table = await client.openTable('vectors')
        } else {
            table = await client.createTable('vectors', [
                { vector: Array(testVector[0].length), text: 'sample' }
            ])
        }

        const LanceDB = await importLanceDBForLangChain()

        const store = new LanceDB(embeddings, {
            table
        })

        const wrapperStore = new ChatLunaSaveableVectorStore(store, {
            async deletableFunction(store) {
                await client.dropTable('vectors')
                await fs.rm(directory, { recursive: true })
            }
        })

        return wrapperStore
    })
}

async function importLanceDB() {
    try {
        const any = await import('vectordb')

        return any.default
    } catch (err) {
        logger.error(err)
        throw new Error(
            'Please install vectordb as a dependency with, e.g. `npm install -S vectordb`'
        )
    }
}

async function importLanceDBForLangChain() {
    try {
        const { LanceDB } = await import(
            '@langchain/community/vectorstores/lancedb'
        )

        return LanceDB
    } catch (err) {
        logger.error(err)
    }
}
