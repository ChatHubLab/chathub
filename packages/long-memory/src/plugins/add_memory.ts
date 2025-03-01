import { Context } from 'koishi'
import { ConversationRoom, logger } from 'koishi-plugin-chatluna'
import { ChainMiddlewareRunStatus } from 'koishi-plugin-chatluna/chains'
import { Config, MemoryRetrievalLayerType, MemoryType } from '../index'
import { createMemoryLayers } from '../utils/layer'

export function apply(ctx: Context, config: Config) {
    const chain = ctx.chatluna.chatChain

    chain
        .middleware('add_memory', async (session, context) => {
            let {
                command,
                options: { type, content, room, view }
            } = context

            if (command !== 'add_memory')
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
                    layers.map((layer) =>
                        layer.addMemories([
                            {
                                content,
                                type: MemoryType.PREFERENCE,
                                importance: 10,
                                // 10 years
                                expirationDate: new Date(
                                    Date.now() + 1000 * 60 * 60 * 24 * 365 * 10
                                )
                            }
                        ])
                    )
                )

                await ctx.chatluna.clearCache(room)

                context.message = session.text('.add_success')
            } catch (error) {
                logger?.error(error)
                context.message = session.text('.add_failed')
            }

            return ChainMiddlewareRunStatus.STOP
        })
        .after('lifecycle-handle_command')
}

declare module 'koishi-plugin-chatluna/chains' {
    interface ChainMiddlewareName {
        add_memory: never
    }

    interface ChainMiddlewareContextOptions {
        content?: string
        room?: ConversationRoom
    }
}
