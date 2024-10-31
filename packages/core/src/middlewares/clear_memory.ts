import { Context } from 'koishi'

import { ChainMiddlewareRunStatus, ChatChain } from '../chains/chain'
import { Config } from '../config'
import { parseRawModelName } from 'koishi-plugin-chatluna/llm-core/utils/count_tokens'
import { resolveLongMemoryId } from './search_memory'
import { logger } from '..'

export function apply(ctx: Context, config: Config, chain: ChatChain) {
    const services = ctx.chatluna.platform

    chain
        .middleware('clear_memory', async (session, context) => {
            let {
                command,
                options: { type, room }
            } = context

            if (command !== 'clear_memory')
                return ChainMiddlewareRunStatus.SKIPPED

            if (!type) {
                type = room.preset
            }

            const [platform, modelName] = parseRawModelName(
                config.defaultEmbeddings
            )
            const embeddings = await ctx.chatluna.createEmbeddings(
                platform,
                modelName
            )

            const key = resolveLongMemoryId(type, session.userId)

            try {
                const vectorStore = await services.createVectorStore(
                    config.defaultVectorStore,
                    { embeddings, key }
                )

                await vectorStore.delete({ deleteAll: true })
                await ctx.chatluna.clearCache(room)
                context.message = session.text('.clear_success')
            } catch (error) {
                logger?.error(error)
                context.message = session.text('.clear_failed')
            }

            return ChainMiddlewareRunStatus.STOP
        })
        .after('lifecycle-handle_command')
}

declare module '../chains/chain' {
    interface ChainMiddlewareName {
        clear_memory: never
    }
}
