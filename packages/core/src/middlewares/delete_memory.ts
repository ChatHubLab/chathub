import { Context } from 'koishi'

import { ChainMiddlewareRunStatus, ChatChain } from '../chains/chain'
import { Config } from '../config'
import { parseRawModelName } from 'koishi-plugin-chatluna/llm-core/utils/count_tokens'
import { resolveLongMemoryId } from './search_memory'
import { logger } from '..'

export function apply(ctx: Context, config: Config, chain: ChatChain) {
    const services = ctx.chatluna.platform

    chain
        .middleware('delete_memory', async (session, context) => {
            let {
                command,
                options: { type, room, ids }
            } = context

            if (!type) {
                type = room.preset
            }

            if (command !== 'delete_memory')
                return ChainMiddlewareRunStatus.SKIPPED

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

                await vectorStore.delete({ ids })

                await vectorStore.save()
                await ctx.chatluna.clearCache(room)
                context.message = session.text('.delete_success')
            } catch (error) {
                logger?.error(error)
                context.message = session.text('.delete_failed')
            }

            return ChainMiddlewareRunStatus.STOP
        })
        .after('lifecycle-handle_command')
}

declare module '../chains/chain' {
    interface ChainMiddlewareName {
        delete_memory: never
    }

    interface ChainMiddlewareContextOptions {
        ids?: string[]
    }
}
