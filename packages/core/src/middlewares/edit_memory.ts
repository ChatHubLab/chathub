import { Context } from 'koishi'

import { ChainMiddlewareRunStatus, ChatChain } from '../chains/chain'
import { Config } from '../config'
import { parseRawModelName } from 'koishi-plugin-chatluna/llm-core/utils/count_tokens'
import { resolveLongMemoryId } from './search_memory'
import { logger } from '..'

export function apply(ctx: Context, config: Config, chain: ChatChain) {
    const services = ctx.chatluna.platform

    chain
        .middleware('edit_memory', async (session, context) => {
            let {
                command,
                options: { type, room, memoryId }
            } = context

            if (!type) {
                type = room.preset
            }

            if (command !== 'edit_memory')
                return ChainMiddlewareRunStatus.SKIPPED

            const [platform, modelName] = parseRawModelName(
                config.defaultEmbeddings
            )
            const embeddings = await ctx.chatluna.createEmbeddings(
                platform,
                modelName
            )

            const key = resolveLongMemoryId(type, session.userId)

            const vectorStore = await services.createVectorStore(
                config.defaultVectorStore,
                { embeddings, key }
            )

            await session.send(session.text('.edit_memory_start'))

            const content = await session.prompt()

            try {
                await vectorStore.editDocument(memoryId, {
                    pageContent: content,
                    metadata: {
                        source: 'user_edit_memory'
                    }
                })
                await vectorStore.save()
                await ctx.chatluna.clearCache(room)
                context.message = session.text('.edit_success')
            } catch (error) {
                logger?.error(error)
                context.message = session.text('.edit_failed')
            }

            return ChainMiddlewareRunStatus.STOP
        })
        .after('lifecycle-handle_command')
}

declare module '../chains/chain' {
    interface ChainMiddlewareName {
        edit_memory: never
    }

    interface ChainMiddlewareContextOptions {
        memoryId?: string
    }
}
