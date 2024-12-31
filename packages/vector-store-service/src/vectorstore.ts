import { Context } from 'koishi'
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'
import { Config } from '.'
// import start
import { apply as faiss } from './vectorstore/faiss'
import { apply as lunavdb } from './vectorstore/lunavdb'
import { apply as milvus } from './vectorstore/milvus'
import { apply as mongodb } from './vectorstore/mongodb'
import { apply as redis } from './vectorstore/redis' // import end

export async function vectorStore(
    ctx: Context,
    config: Config,
    plugin: ChatLunaPlugin
) {
    type VectorStore = (
        ctx: Context,
        config: Config,
        plugin: ChatLunaPlugin
    ) => PromiseLike<void> | void

    const middlewares: VectorStore[] =
        // middleware start
        [faiss, lunavdb, milvus, mongodb, redis] // middleware end

    for (const middleware of middlewares) {
        try {
            await middleware(ctx, config, plugin)
        } catch (error) {
            ctx.logger.error(error)
        }
    }
}
