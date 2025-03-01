import { Context } from 'koishi'
import { logger } from 'koishi-plugin-chatluna'
import { ChainMiddlewareRunStatus } from 'koishi-plugin-chatluna/chains'
import { Config, MemoryRetrievalLayerType } from '../index'
import { createMemoryLayers } from '../utils/layer'

export function apply(ctx: Context, config: Config) {
    const chain = ctx.chatluna.chatChain

    chain
        .middleware('delete_memory', async (session, context) => {
            let {
                command,
                options: { type, room, ids, view }
            } = context

            if (command !== 'delete_memory')
                return ChainMiddlewareRunStatus.SKIPPED

            if (!type) {
                type = room.preset
            }

            let parsedLayerType = MemoryRetrievalLayerType.PRESET_USER

            if (view != null) {
                parsedLayerType = MemoryRetrievalLayerType[view.toUpperCase()]

                if (parsedLayerType == null) {
                    context.message = session.text('.invalid_view', [
                        ['global', 'preset', 'user', 'preset_user'].join(', ')
                    ])
                    return ChainMiddlewareRunStatus.STOP
                }
            }

            try {
                const layers = await createMemoryLayers(
                    ctx,
                    type,
                    session.userId,
                    [parsedLayerType]
                )

                await Promise.all(
                    layers.map((layer) => layer.deleteMemories(ids))
                )

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

declare module 'koishi-plugin-chatluna/chains' {
    interface ChainMiddlewareName {
        delete_memory: never
    }

    interface ChainMiddlewareContextOptions {
        ids?: string[]
    }
}
