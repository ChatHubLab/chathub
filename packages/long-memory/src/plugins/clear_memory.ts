import { Context } from 'koishi'
import { logger } from 'koishi-plugin-chatluna'
import { ChainMiddlewareRunStatus } from 'koishi-plugin-chatluna/chains'
import { MemoryRetrievalLayerType } from '../types'
import { Config } from '..'
import { createMemoryLayers } from '../utils/layer'

export function apply(ctx: Context, config: Config) {
    const chain = ctx.chatluna.chatChain

    chain
        .middleware('clear_memory', async (session, context) => {
            let {
                command,
                options: { type, room, view }
            } = context

            if (command !== 'clear_memory')
                return ChainMiddlewareRunStatus.SKIPPED

            if (!type) {
                type = room.preset
            }

            try {
                const layers = await createMemoryLayers(
                    ctx,
                    type,
                    session.userId,
                    view != null
                        ? [view as MemoryRetrievalLayerType]
                        : ctx.chatluna_long_memory.defaultLayerTypes
                )

                await Promise.all(layers.map((layer) => layer.clearMemories()))

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

declare module 'koishi-plugin-chatluna/chains' {
    interface ChainMiddlewareName {
        clear_memory: never
    }
}
