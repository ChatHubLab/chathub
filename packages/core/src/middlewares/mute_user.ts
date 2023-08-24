import { Context, h } from 'koishi';
import { Config } from '../config';
import { ChainMiddlewareRunStatus, ChatChain } from '../chains/chain';
import { createLogger } from '../llm-core/utils/logger';
import { checkAdmin, deleteConversationRoom, getAllJoinedConversationRoom, getConversationRoomUser, muteUserFromConversationRoom, switchConversationRoom } from '../chains/rooms';
import { ConversationRoom } from '../types';


const logger = createLogger("@dingyi222666/chathub/middlewares/delete_room")

export function apply(ctx: Context, config: Config, chain: ChatChain) {
    chain.middleware("mute_user", async (session, context) => {

        let { command, options: { room } } = context

        if (command !== "muteUser") return ChainMiddlewareRunStatus.SKIPPED


        if (room == null && context.options.room_resolve != null) {
            // 尝试完整搜索一次

            const rooms = await getAllJoinedConversationRoom(ctx, session, true)

            const roomId = parseInt(context.options.room_resolve?.name)

            room = rooms.find(room => room.roomName === context.options.room_resolve?.name || room.roomId === roomId)
        }


        if (room == null) {
            context.message = "未找到指定的房间。"
            return ChainMiddlewareRunStatus.STOP
        }

        const userInfo = await getConversationRoomUser(ctx, session, room, session.userId)

        if (userInfo.roomPermission === "member" && !(await checkAdmin(session))) {
            context.message = `你不是房间 ${room.roomName} 的管理员，无法禁言用户。`
            return ChainMiddlewareRunStatus.STOP
        }


        const targetUser = context.options.resolve_user.id as string[]

        for (const user of targetUser) {
            await muteUserFromConversationRoom(ctx, session, room, user)
        }

        context.message = `已将用户 ${targetUser.join(",")} 在房间 ${room.roomName} 禁言或解除禁言。`

        return ChainMiddlewareRunStatus.STOP
    }).after("lifecycle-handle_command")
}


declare module '../chains/chain' {
    interface ChainMiddlewareName {
        "mute_user": never
    }


}