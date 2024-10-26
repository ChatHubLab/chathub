/* eslint-disable operator-linebreak */
import { Context, h, Logger } from 'koishi'
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

            if (config.allowChatWithRoomName) {
                const needContinue =
                    // 私聊
                    session.isDirect &&
                    config.allowPrivate &&
                    (context.command != null ||
                        config.privateChatWithoutCommand)
                        ? true // 群艾特
                        : session.stripped.appel && config.allowAtReply
                          ? true
                          : // bot名字
                            session.content.startsWith(config.botName) &&
                              config.isNickname
                            ? true
                            : // 命令
                              context.command != null

                // split the chat content
                const splitContent = (context.message as string).split(' ')

                let matchedRoom: ConversationRoom

                // the first word is the room name
                if (splitContent.length > 1) {
                    matchedRoom = await queryJoinedConversationRoom(
                        ctx,
                        session,
                        splitContent.shift()
                    )
                }

                if (matchedRoom == null && !needContinue) {
                    // 无敌了，破需求

                    return ChainMiddlewareRunStatus.STOP
                }

                if (matchedRoom != null) {
                    joinRoom = matchedRoom

                    context.options.inputMessage =
                        await ctx.chatluna.messageTransformer.transform(
                            session,
                            [h.text(splitContent.concat(' '))]
                        )
                }
            }

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
                        // 优先加入自己创建的房间
                        joinedRooms.find(
                            (room) =>
                                room.visibility === 'template_clone' &&
                                room.roomMasterId === session.userId
                        )

                    if (
                        config.autoCreateRoomFromUser !== true &&
                        joinRoom == null
                    ) {
                        joinRoom = // 优先加入模版克隆房间
                            joinedRooms.find(
                                (room) => room.visibility === 'template_clone'
                            ) ??
                            joinedRooms[
                                Math.floor(Math.random() * joinedRooms.length)
                            ]
                    }

                    if (joinRoom != null) {
                        await switchConversationRoom(
                            ctx,
                            session,
                            joinRoom.roomId
                        )

                        logger.success(
                            session.text('chatluna.room.auto_switch', [
                                session.userId,
                                joinRoom.roomName
                            ])
                        )
                    }
                }
            }

            if (
                joinRoom == null &&
                config.autoCreateRoomFromUser !== true &&
                !session.isDirect &&
                (context.command?.length ?? 0) < 1
            ) {
                joinRoom = await queryPublicConversationRoom(ctx, session)
                if (joinRoom != null) {
                    logger.success(
                        session.text('chatluna.room.auto_switch', [
                            session.userId,
                            joinRoom.roomName
                        ])
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

                if (config.autoCreateRoomFromUser) {
                    // 如果是群聊的公共房间，那么就房主直接设置为聊天者，否则就是私聊
                    cloneRoom.roomMasterId = session.userId

                    cloneRoom.visibility = 'private'

                    cloneRoom.roomId = (await getMaxConversationRoomId(ctx)) + 1

                    cloneRoom.roomName = session.text(
                        'chatluna.room.room_name',
                        [
                            session.isDirect
                                ? `${session.username ?? session.userId}`
                                : `${
                                      session.event.guild.name ??
                                      session.username ??
                                      session.event.guild.id.toString()
                                  }`
                        ]
                    )

                    logger.success(
                        session.text('chatluna.room.auto_create', [
                            session.userId,
                            cloneRoom.roomName
                        ])
                    )
                } else {
                    // 如果是群聊的公共房间，那么就房主直接设置为聊天者，否则就是私聊
                    cloneRoom.roomMasterId = session.userId

                    cloneRoom.visibility = 'template_clone'

                    cloneRoom.roomId = (await getMaxConversationRoomId(ctx)) + 1

                    cloneRoom.roomName = session.text(
                        'chatluna.room.template_clone_room_name',
                        [
                            session.isDirect
                                ? `${session.username ?? session.userId}`
                                : `${
                                      session.event.guild.name ??
                                      session.username ??
                                      session.event.guild.id.toString()
                                  }`
                        ]
                    )

                    logger.success(
                        session.text('chatluna.room.auto_create_template', [
                            session.userId,
                            cloneRoom.roomName
                        ])
                    )
                }

                await createConversationRoom(ctx, session, cloneRoom)

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
                    joinRoom.chatMode !== config.defaultChatMode ||
                    joinRoom.model !== config.defaultModel
                ) {
                    needUpdate = true
                }

                joinRoom.model = config.defaultModel
                joinRoom.preset = config.defaultPreset
                joinRoom.chatMode = config.defaultChatMode

                if (joinRoom.preset !== config.defaultPreset) {
                    // 需要提前清空聊天记录
                    await ctx.chatluna.clearChatHistory(joinRoom)
                }

                if (needUpdate) {
                    await ctx.chatluna.clearCache(joinRoom)

                    await ctx.database.upsert('chathub_room', [joinRoom])

                    logger.debug(
                        session.text('chatluna.room.config_changed', [
                            joinRoom.roomName
                        ])
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
