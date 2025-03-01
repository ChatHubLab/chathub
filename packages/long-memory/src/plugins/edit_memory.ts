import { Context } from 'koishi'
import { logger } from 'koishi-plugin-chatluna'
import { ChainMiddlewareRunStatus } from 'koishi-plugin-chatluna/chains'
import { MemoryRetrievalLayerType, MemoryType } from '../types'
import { Config } from '..'
import { createDefaultMemory } from '../utils/memory'
import { createMemoryLayers } from '../utils/layer'
export function apply(ctx: Context, config: Config) {
    const chain = ctx.chatluna.chatChain

    chain
        .middleware('edit_memory', async (session, context) => {
            let {
                command,
                options: { type, room, memoryId, view }
            } = context

            if (command !== 'edit_memory')
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
                await session.send(session.text('.edit_memory_start'))

                const content = await session.prompt()

                const layers = await createMemoryLayers(
                    ctx,
                    type,
                    session.userId,
                    [parsedLayerType]
                )

                await Promise.all(
                    layers.map((layer) => layer.deleteMemories([memoryId]))
                )

                const memory = createDefaultMemory(
                    content,
                    MemoryType.PREFERENCE,
                    10
                )

                await Promise.all(
                    layers.map((layer) => layer.addMemories([memory]))
                )

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

declare module 'koishi-plugin-chatluna/chains' {
    interface ChainMiddlewareName {
        edit_memory: never
    }

    interface ChainMiddlewareContextOptions {
        memoryId?: string
        view?: string
    }
}
