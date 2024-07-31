import { Context, Logger } from 'koishi'
import { Config } from '../config'

import { ConversationRoom } from '../types'
import { v4 as uuidv4 } from 'uuid'
import { createLogger } from 'koishi-plugin-chatluna/utils/logger'
import { ChainMiddlewareRunStatus, ChatChain } from '../chains/chain'
import {
    createConversationRoom,
    getAllJoinedConversationRoom,
    getConversationRoomCount as getMaxConversationRoomId,
    getTemplateConversationRoom,
    queryJoinedConversationRoom,
    queryPublicConversationRoom,
    switchConversationRoom
} from '../chains/rooms'

let logger: Logger

export function apply(ctx: Context, config: Config, chain: ChatChain) {
    logger = createLogger(ctx)
    chain
        .middleware('resolve_room', async (session, context) => {
            let joinRoom = await queryJoinedConversationRoom(
                ctx,
                session,
                context.options?.room_resolve?.name
            )

            if (joinRoom == null) {
                // 随机加入到一个你已经加入的房间？？？
                const joinedRooms = await getAllJoinedConversationRoom(
                    ctx,
                    session
                )

                if (joinedRooms.length > 0) {
                    joinRoom =
                        // 优先加入自己创建的房间
                        joinedRooms.find(
                            (room) =>
                                room.visibility === 'private' &&
                                room.roomMasterId === session.userId
                        ) ??
                        // 优先加入模版克隆房间
                        joinedRooms.find(
                            (room) => room.visibility === 'template_clone'
                        ) ??
                        joinedRooms[
                            Math.floor(Math.random() * joinedRooms.length)
                        ]

                    await switchConversationRoom(ctx, session, joinRoom.roomId)

                    logger.success(
                        `已为用户 ${session.userId} 自动切换到房间 ${joinRoom.roomName}。`
                    )
                }
            }

            if (
                joinRoom == null &&
                !session.isDirect &&
                (context.command?.length ?? 0) < 1
            ) {
                joinRoom = await queryPublicConversationRoom(ctx, session)
                if (joinRoom != null) {
                    logger.success(
                        `已为用户 ${session.userId} 自动切换到公共房间 ${joinRoom.roomName}。`
                    )
                }
            }

            if (joinRoom == null && (context.command?.length ?? 0) < 1) {
                // 尝试基于模板房间创建模版克隆房间

                const templateRoom = await getTemplateConversationRoom(
                    ctx,
                    config
                )

                if (templateRoom == null) {
                    // 没有就算了。后面需要房间的中间件直接报错就完事。
                    return ChainMiddlewareRunStatus.SKIPPED
                }

                const cloneRoom = structuredClone(templateRoom)

                cloneRoom.conversationId = uuidv4()

                // 如果是群聊的公共房间，那么就房主直接设置为群主，否则就是私聊
                cloneRoom.roomMasterId = session.userId

                cloneRoom.visibility = 'template_clone'

                cloneRoom.roomId = (await getMaxConversationRoomId(ctx)) + 1

                cloneRoom.roomName = session.isDirect
                    ? `${session.username ?? session.userId} 的模版克隆房间`
                    : `${
                          session.event.guild.name ??
                          session.username ??
                          session.event.guild.id.toString()
                      } 的模版克隆房间`

                await createConversationRoom(ctx, session, cloneRoom)

                logger.success(
                    `已为用户 ${session.userId} 自动创建模版克隆房间 ${cloneRoom.roomName}。`
                )

                joinRoom = cloneRoom
            }

            if (
                joinRoom?.visibility === 'template_clone' &&
                joinRoom?.autoUpdate === true
            ) {
                // 直接从配置里面复制

                // 对于 preset，chatModel 的变更，我们需要写入数据库
                let needUpdate = false
                if (
                    joinRoom.preset !== config.defaultPreset ||
                    joinRoom.chatMode !== config.defaultChatMode
                ) {
                    needUpdate = true
                } else if (joinRoom.model !== config.defaultModel) {
                    await ctx.chatluna.clearCache(joinRoom)
                }

                joinRoom.model = config.defaultModel
                joinRoom.preset = config.defaultPreset
                joinRoom.chatMode = config.defaultChatMode

                if (needUpdate) {
                    await ctx.database.upsert('chathub_room', [joinRoom])
                    // 需要提前清空聊天记录
                    await ctx.chatluna.clearChatHistory(joinRoom)
                    logger.debug(
                        `检测到模版房间 ${joinRoom.roomName} 的配置变更，已更新到数据库。`
                    )
                }
            }

            context.options.room = joinRoom

            return ChainMiddlewareRunStatus.CONTINUE
        })
        .after('lifecycle-prepare')
    //  .before("lifecycle-request_model")
}

export type ChatMode = 'plugin' | 'chat' | 'browsing'

declare module '../chains/chain' {
    interface ChainMiddlewareContextOptions {
        room?: ConversationRoom
    }

    interface ChainMiddlewareName {
        resolve_room: never
    }
}
