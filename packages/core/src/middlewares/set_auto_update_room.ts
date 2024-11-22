import { Context } from 'koishi'
import { Config } from '../config'
import { ChainMiddlewareRunStatus, ChatChain } from '../chains/chain'
import { checkAdmin, getAllJoinedConversationRoom } from '../chains/rooms'

export function apply(ctx: Context, config: Config, chain: ChatChain) {
    chain
        .middleware('set_auto_update_room', async (session, context) => {
            const { command } = context

            if (command !== 'set_auto_update_room')
                return ChainMiddlewareRunStatus.SKIPPED

            let { room: targetRoom, auto_update_room: autoUpdateRoom } =
                context.options

            if (targetRoom == null && context.options.room_resolve != null) {
                // 尝试完整搜索一次

                const rooms = await getAllJoinedConversationRoom(
                    ctx,
                    session,
                    true
                )

                const roomId = parseInt(context.options.room_resolve?.name)

                targetRoom = rooms.find(
                    (room) =>
                        room.roomName === context.options.room_resolve?.name ||
                        room.roomId === roomId
                )
            }

            if (targetRoom == null) {
                context.message = session.text('.room_not_found')
                return ChainMiddlewareRunStatus.STOP
            }

            if (
                targetRoom.roomMasterId !== session.userId &&
                !(await checkAdmin(session))
            ) {
                context.message = session.text('.not_admin')
                return ChainMiddlewareRunStatus.STOP
            }

            targetRoom.autoUpdate = context.options.auto_update_room

            await ctx.database.upsert('chathub_room', [targetRoom])

            context.message = session.text('.success', [
                targetRoom.roomName,
                autoUpdateRoom
            ])

            return ChainMiddlewareRunStatus.STOP
        })
        .after('lifecycle-handle_command')
}

declare module '../chains/chain' {
    interface ChainMiddlewareName {
        set_auto_update_room: never
    }
    interface ChainMiddlewareContextOptions {
        auto_update_room?: boolean
    }
}
