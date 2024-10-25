import { RedisVectorStore } from '@langchain/redis'
import { Context, Logger } from 'koishi'
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'
import { createLogger } from 'koishi-plugin-chatluna/utils/logger'
import { Config } from '..'

let logger: Logger

export async function apply(
    ctx: Context,
    config: Config,
    plugin: ChatLunaPlugin
) {
    logger = createLogger(ctx, 'chatluna-vector-store-service')

    plugin.registerVectorStore('redis', async (params) => {
        const embeddings = params.embeddings

        const client = await createClient(config.redisUrl)

        await client.connect()

        const vector = new RedisVectorStore(embeddings, {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            redisClient: client,
            indexName: params.key ?? 'chatluna'
        })

        const testVector = await embeddings.embedDocuments(['test'])

        try {
            await vector.createIndex(testVector[0].length)
        } catch (e) {
            try {
                await vector.dropIndex(true)
                await vector.createIndex()
            } catch (e) {
                logger.error(e)
            }
            logger.error(e)
        }

        return vector
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
