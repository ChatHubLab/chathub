import { Context, Session } from 'koishi'
import { Config } from '../config'
import { ChainMiddlewareRunStatus, ChatChain } from '../chains/chain'
import {
    checkConversationRoomAvailability,
    getAllJoinedConversationRoom,
    queryConversationRoom,
    queryPublicConversationRooms
} from '../chains/rooms'
import { ConversationRoom } from '../types'
import { Pagination } from 'koishi-plugin-chatluna/utils/pagination'

export function apply(ctx: Context, config: Config, chain: ChatChain) {
    const pagination = new Pagination<ConversationRoom>({
        formatItem: (value) => '',
        formatString: {
            top: '',
            bottom: '',
            pages: ''
        }
    })

    chain
        .middleware('list_room', async (session, context) => {
            const {
                command,
                options: { page, limit, all_room }
            } = context

            if (command !== 'list_room') return ChainMiddlewareRunStatus.SKIPPED

            pagination.updateFormatString({
                top: session.text('.header') + '\n',
                bottom: '\n' + session.text('.footer'),
                pages: '\n' + session.text('.pages')
            })

            pagination.updateFormatItem((value) =>
                formatRoomInfo(ctx, session, value)
            )

            const rooms = await getAllJoinedConversationRoom(ctx, session)

            if (all_room) {
                const publicRooms = await queryPublicConversationRooms(
                    ctx,
                    session
                )
                for (const room of publicRooms) {
                    if (!rooms.find((r) => r.roomId === room.roomId)) {
                        rooms.push(
                            await queryConversationRoom(
                                ctx,
                                session,
                                room.roomId
                            )
                        )
                    }
                }
            }

            const key = session.isDirect
                ? session.userId
                : session.guildId + '-' + session.userId

            await pagination.push(rooms, key)

            context.message = await pagination.getFormattedPage(
                page,
                limit,
                key
            )

            return ChainMiddlewareRunStatus.STOP
        })
        .after('lifecycle-handle_command')
}

async function formatRoomInfo(
    ctx: Context,
    session: Session,
    room: ConversationRoom
) {
    const buffer = []

    buffer.push(session.text('.room_name', [room.roomName]))
    buffer.push(session.text('.room_id', [room.roomId]))
    buffer.push(session.text('.room_preset', [room.preset]))
    buffer.push(session.text('.room_model', [room.model]))
    buffer.push(session.text('.room_visibility', [room.visibility]))
    buffer.push(session.text('.room_chat_mode', [room.chatMode]))
    buffer.push(session.text('.room_master_id', [room.roomMasterId]))
    buffer.push(
        session.text('.room_availability', [
            await checkConversationRoomAvailability(ctx, room)
        ])
    )

    buffer.push('\n')

    return buffer.join('\n')
}

declare module '../chains/chain' {
    interface ChainMiddlewareName {
        list_room: never
    }

    interface ChainMiddlewareContextOptions {
        all_room?: boolean
    }
}
