import { Context, h } from 'koishi'
import { Config } from '../config'
import { ChainMiddlewareRunStatus, ChatChain } from '../chains/chain'
import { getAllJoinedConversationRoom } from '../chains/rooms'
import { ChatLunaMessage } from '../llm-core/memory/message/database_history'
import { logger } from '..'

export function apply(ctx: Context, config: Config, chain: ChatChain) {
    chain
        .middleware('rollback_chat', async (session, context) => {
            const { command } = context

            if (command !== 'rollback') return ChainMiddlewareRunStatus.SKIPPED

            let room = context.options.room

            const rollbackRound = context.options.rollback_round ?? 1

            if (room == null && context.options.room_resolve != null) {
                // 尝试完整搜索一次

                const rooms = await getAllJoinedConversationRoom(
                    ctx,
                    session,
                    true
                )

                const roomId = parseInt(context.options.room_resolve?.name)

                room = rooms.find(
                    (room) =>
                        room.roomName === context.options.room_resolve?.name ||
                        room.roomId === roomId
                )
            }

            if (room == null) {
                context.message = session.text('.room_not_found')
                return ChainMiddlewareRunStatus.STOP
            }

            // clear cache

            await ctx.chatluna.clearCache(room)

            // get messages
            const conversation = (
                await ctx.database.get('chathub_conversation', {
                    id: room.conversationId
                })
            )?.[0]

            if (conversation === null) {
                context.message = session.text('.conversation_not_exist')
                return ChainMiddlewareRunStatus.STOP
            }

            let parentId = conversation.latestId
            const messages: ChatLunaMessage[] = []

            // 获取 (轮数*2) 条消息，一轮对话 两条消息
            while (messages.length < rollbackRound * 2) {
                const message = await ctx.database.get('chathub_message', {
                    conversation: room.conversationId,
                    id: parentId
                })

                parentId = message[0]?.parent

                messages.unshift(...message)

                if (parentId == null) {
                    break
                }
            }

            // 小于目标轮次，就是没有
            if (messages.length < rollbackRound * 2) {
                context.message = session.text('.no_chat_history')
                return ChainMiddlewareRunStatus.STOP
            }

            // 最后一条消息

            const lastMessage =
                parentId == null
                    ? undefined
                    : await ctx.database
                          .get('chathub_message', {
                              conversation: room.conversationId,
                              id: parentId
                          })
                          .then((message) => message?.[0])

            const humanMessage = messages[messages.length - 2]
            await ctx.database.upsert('chathub_conversation', [
                {
                    id: room.conversationId,
                    latestId: parentId == null ? null : lastMessage.id
                }
            ])

            if ((context.options.message?.length ?? 0) < 1) {
                context.options.inputMessage =
                    await ctx.chatluna.messageTransformer.transform(session, [
                        h.text(humanMessage.text)
                    ])
            }

            await ctx.database.remove('chathub_message', {
                id: messages.map((message) => message.id)
            })

            await session.send(
                session.text('.rollback_success', [rollbackRound])
            )

            logger.debug(
                `rollback chat ${room.roomName} ${context.options.inputMessage}`
            )

            return ChainMiddlewareRunStatus.CONTINUE
        })
        .after('lifecycle-handle_command')
}

declare module '../chains/chain' {
    interface ChainMiddlewareName {
        rollback_chat: never
    }
    interface ChainMiddlewareContextOptions {
        rollback_round?: number
    }
}
