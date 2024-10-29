import { Context } from 'koishi'

import { ChainMiddlewareRunStatus, ChatChain } from '../chains/chain'
import { Config } from '../config'
import { parseRawModelName } from 'koishi-plugin-chatluna/llm-core/utils/count_tokens'
import { resolveLongMemoryId } from './search_memory'
import { logger } from '..'

export function apply(ctx: Context, config: Config, chain: ChatChain) {
    const services = ctx.chatluna.platform

    chain
        .middleware('add_memory', async (session, context) => {
            let {
                command,
                options: { type, content, room }
            } = context

            if (!type) {
                type = room.preset
            }

            if (command !== 'add_memory')
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

            try {
                await vectorStore.addDocuments([
                    {
                        pageContent: content,
                        metadata: {
                            source: 'user_memory'
                        }
                    }
                ])

                await vectorStore.save()
                context.message = session.text('.add_success')
            } catch (error) {
                logger?.error(error)
                context.message = session.text('.add_failed')
            }

            return ChainMiddlewareRunStatus.STOP
        })
        .after('lifecycle-handle_command')
}

declare module '../chains/chain' {
    interface ChainMiddlewareName {
        add_memory: never
    }

    interface ChainMiddlewareContextOptions {
        content?: string
    }
}
