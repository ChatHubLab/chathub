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

            for (let i = 0; i < 3; i++) {
                const message = await ctx.database.get('chathub_message', {
                    conversation: room.conversationId,
                    id: parentId
                })

                if (message == null) {
                    break
                }

                parentId = message[0]?.parent

                messages.push(...message)
            }

            if (messages.length < 2) {
                context.message = session.text('.no_chat_history')
                return ChainMiddlewareRunStatus.STOP
            }

            const parentMessage = messages[2]

            if (parentMessage == null) {
                await ctx.database.upsert('chathub_conversation', [
                    {
                        id: room.conversationId,
                        latestId: null
                    }
                ])
            } else {
                await ctx.database.upsert('chathub_conversation', [
                    {
                        id: room.conversationId,
                        latestId: parentMessage.id
                    }
                ])

                messages.pop()
            }

            const humanMessage = messages[1]

            if (humanMessage.role !== 'human') {
                context.message = session.text('.invalid_chat_history')
                return ChainMiddlewareRunStatus.STOP
            }

            if ((context.options.message?.length ?? 0) < 1) {
                context.options.inputMessage =
                    await ctx.chatluna.messageTransformer.transform(session, [
                        h.text(humanMessage.text)
                    ])
            }

            while (messages.length > 0) {
                await ctx.database.remove('chathub_message', {
                    id: messages.pop()?.id
                })
            }

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
}
