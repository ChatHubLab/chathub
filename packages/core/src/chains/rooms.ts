import { randomInt } from 'crypto'
import { $, Context, Session, User } from 'koishi'
import { ModelType } from 'koishi-plugin-chatluna/llm-core/platform/types'
import { parseRawModelName } from 'koishi-plugin-chatluna/llm-core/utils/count_tokens'
import {
    ChatLunaError,
    ChatLunaErrorCode
} from 'koishi-plugin-chatluna/utils/error'
import { Config } from '../config'
import { chunkArray } from '../llm-core/utils/chunk'
import { ConversationRoom, ConversationRoomGroupInfo } from '../types'

export async function queryJoinedConversationRoom(
    ctx: Context,
    session: Session,
    name?: string
) {
    if (name != null) {
        const joinedRooms = await getAllJoinedConversationRoom(ctx, session)

        return joinedRooms.find(
            (it) => it.roomName === name || it.roomId === parseInt(name)
        )
    }

    const userRoomInfoList = await ctx.database.get('chathub_user', {
        userId: session.userId,
        groupId: session.isDirect ? '0' : session.guildId
    })

    if (userRoomInfoList.length > 1) {
        throw new ChatLunaError(
            ChatLunaErrorCode.UNKNOWN_ERROR,
            new Error('User has multiple default rooms, this is impossible!')
        )
    } else if (userRoomInfoList.length === 0) {
        return undefined
    }
    const userRoomInfo = userRoomInfoList[0]
    return await resolveConversationRoom(ctx, userRoomInfo.defaultRoomId)
}

export async function queryPublicConversationRoom(
    ctx: Context,
    session: Session
) {
    // 如果是私聊，直接返回 null

    if (session.isDirect) {
        return undefined
    }

    // 如果是群聊，那么就查询群聊的公共房间

    const groupRoomInfoList = await ctx.database.get(
        'chathub_room_group_member',
        {
            groupId: session.guildId,
            roomVisibility: {
                // TODO: better type
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                $in: ['template_clone', 'public'] as unknown as any
                //    $in: ['template_clone', 'public']
            }
        }
    )

    // 优先加入模版克隆房间
    const templateCloneRoom = groupRoomInfoList.find(
        (it) => it.roomVisibility === 'template_clone'
    )

    let roomId: number

    if (templateCloneRoom != null) {
        roomId = templateCloneRoom.roomId
    } else if (groupRoomInfoList.length < 1) {
        return undefined
    } else if (groupRoomInfoList.length === 1) {
        roomId = groupRoomInfoList[0].roomId
    } else {
        const groupRoomInfo =
            groupRoomInfoList[randomInt(groupRoomInfoList.length)]
        roomId = groupRoomInfo.roomId
    }

    const room = await resolveConversationRoom(ctx, roomId)

    if (room == null && roomId !== 0) {
        // why?
        await deleteConversationRoomByRoomId(ctx, roomId)
        return undefined
    }

    await joinConversationRoom(ctx, session, room)
    return room
}

export async function checkConversationRoomAvailability(
    ctx: Context,
    room: ConversationRoom
): Promise<boolean> {
    const platformService = ctx.chatluna.platform
    const presetService = ctx.chatluna.preset

    // check model

    const [platformName, modelName] = parseRawModelName(room.model)

    const platformModels = platformService.getModels(
        platformName,
        ModelType.llm
    )

    if (platformModels.length < 1) {
        return false
    }

    if (!platformModels.some((it) => it.name === modelName)) {
        return false
    }

    if (!(await presetService.getPreset(room.preset))) {
        return false
    }

    return true
}

export async function fixConversationRoomAvailability(
    ctx: Context,
    config: Config,
    room: ConversationRoom
) {
    const platformService = ctx.chatluna.platform
    const presetService = ctx.chatluna.preset

    // check model

    const [platformName, modelName] = parseRawModelName(room.model)

    const platformModels = platformService.getModels(
        platformName,
        ModelType.llm
    )

    if (platformModels.length < 1) {
        // 直接使用模版的房间
        room.model = (await getTemplateConversationRoom(ctx, config)).model
    } else if (!platformModels.some((it) => it.name === modelName)) {
        // 随机模型
        room.model = platformName + '/' + platformModels[0].name
    }

    if (!(await presetService.getPreset(room.preset))) {
        room.preset = (await presetService.getDefaultPreset()).triggerKeyword[0]
    }

    await ctx.database.upsert('chathub_room', [room])
}

export async function getTemplateConversationRoom(
    ctx: Context,
    config: Config
): Promise<ConversationRoom> {
    if (
        config.defaultChatMode == null ||
        config.defaultModel === '无' ||
        config.defaultPreset == null
    ) {
        if (config.defaultChatMode == null) {
            throw new ChatLunaError(ChatLunaErrorCode.ROOM_TEMPLATE_INVALID)
        }

        if (config.defaultModel === '无' || config.defaultModel == null) {
            const models = ctx.chatluna.platform.getAllModels(ModelType.llm)

            const model =
                models.find((model) => model.includes('3.5-turbo')) ?? models[0]

            config.defaultModel = model
        }

        if (config.defaultPreset == null) {
            const preset = await ctx.chatluna.preset.getDefaultPreset()

            config.defaultPreset = preset.triggerKeyword[0]
        }

        ctx.runtime.parent.scope.update(config, true)

        // throw new ChatLunaError(ChatLunaErrorCode.INIT_ROOM)
    }

    const room: ConversationRoom = {
        roomId: 0,
        roomName: '模板房间',
        roomMasterId: '0',
        preset: config.defaultPreset,
        conversationId: '0',
        chatMode: config.defaultChatMode,
        password: '',
        model: config.defaultModel,
        visibility: 'public',
        autoUpdate: true,
        updatedTime: new Date()
    }

    if (!(await checkConversationRoomAvailability(ctx, room))) {
        throw new ChatLunaError(ChatLunaErrorCode.ROOM_TEMPLATE_INVALID)
    }

    return room
}

export async function getConversationRoomCount(ctx: Context) {
    const count = await ctx.database.eval(
        'chathub_room',
        (row) => $.max(row.roomId),
        {}
    )

    return count
}

export async function transferConversationRoom(
    ctx: Context,
    session: Session,
    room: ConversationRoom,
    userId: string
) {
    const memberList = await ctx.database.get('chathub_room_member', {
        roomId: room.roomId,
        userId
    })

    if (memberList.length === 0) {
        throw new ChatLunaError(ChatLunaErrorCode.ROOM_NOT_FOUND)
    }

    await ctx.database.upsert('chathub_room', [
        {
            roomId: room.roomId,
            roomMasterId: userId
        }
    ])

    // 搜索原来的房主，降级为成员

    const oldMaster = await ctx.database.get('chathub_room_member', {
        roomId: room.roomId,
        roomPermission: 'owner'
    })

    if (oldMaster.length === 1) {
        await ctx.database.upsert('chathub_room_member', [
            {
                userId: oldMaster[0].userId,
                roomId: room.roomId,
                roomPermission: 'member'
            }
        ])
    } else {
        throw new ChatLunaError(ChatLunaErrorCode.ROOM_NOT_FOUND_MASTER)
    }

    await ctx.database.upsert('chathub_room_member', [
        {
            userId,
            roomId: room.roomId,
            roomPermission: 'owner'
        }
    ])

    await ctx.database.upsert('chathub_user', [
        {
            userId,
            defaultRoomId: room.roomId,
            groupId: session.isDirect ? '0' : session.guildId
        }
    ])
}

export async function switchConversationRoom(
    ctx: Context,
    session: Session,
    id: string | number
) {
    let joinedRoom = await getAllJoinedConversationRoom(ctx, session)

    const parsedId = typeof id === 'number' ? id : parseInt(id)

    let room = joinedRoom.find((it) => it.roomId === parsedId)

    if (room != null) {
        await ctx.database.upsert('chathub_user', [
            {
                userId: session.userId,
                defaultRoomId: room.roomId,
                groupId: session.isDirect ? '0' : session.guildId
            }
        ])

        return room
    }

    joinedRoom = joinedRoom.filter((it) => it.roomName === id)

    if (joinedRoom.length > 1) {
        throw new ChatLunaError(
            ChatLunaErrorCode.THE_NAME_FIND_IN_MULTIPLE_ROOMS
        )
    } else if (joinedRoom.length === 0) {
        throw new ChatLunaError(ChatLunaErrorCode.ROOM_NOT_FOUND)
    } else {
        room = joinedRoom[0]
    }

    await ctx.database.upsert('chathub_user', [
        {
            userId: session.userId,
            defaultRoomId: room.roomId,
            groupId: session.isDirect ? '0' : session.guildId
        }
    ])

    return room
}

export async function getAllJoinedConversationRoom(
    ctx: Context,
    session: Session,
    queryAll: boolean = false
) {
    // 这里分片进行 chunk 然后用 in 查询，这么做的好处是可以减少很多的查询次数
    const conversationRoomList = chunkArray(
        await ctx.database.get('chathub_room_member', {
            userId: session.userId
        }),
        35
    )

    const rooms: ConversationRoom[] = []

    for (const conversationRoomChunk of conversationRoomList) {
        const roomIds = conversationRoomChunk.map((it) => it.roomId)
        const roomList = await ctx.database.get('chathub_room', {
            roomId: {
                $in: roomIds
            }
        })

        let memberList: ConversationRoomGroupInfo[] = []

        if (queryAll === false) {
            memberList = await ctx.database.get('chathub_room_group_member', {
                roomId: {
                    $in: roomIds
                },
                // 设置 undefined 来全量搜索
                groupId: session.guildId ?? undefined
            })
        }

        for (const room of roomList) {
            const memberOfTheRoom = memberList.some(
                (it) => it.roomId === room.roomId
            )

            if (
                // 模版克隆房间或者公共房间需要指定房间的范围不能干预到私聊的
                (!session.isDirect && memberOfTheRoom) ||
                // 同上
                (session.isDirect && room.visibility !== 'template_clone') ||
                // 私有房间跨群
                room.visibility === 'private' ||
                (room.visibility === 'template_clone' &&
                    session.isDirect &&
                    !memberOfTheRoom) ||
                queryAll === true
            ) {
                rooms.push(room)
            }
        }
    }

    return rooms
}

export async function leaveConversationRoom(
    ctx: Context,
    session: Session,
    room: ConversationRoom
) {
    await ctx.database.remove('chathub_room_member', {
        userId: session.userId,
        roomId: room.roomId
    })

    await ctx.database.remove('chathub_user', {
        userId: session.userId,
        defaultRoomId: room.roomId
    })
}

export async function queryConversationRoom(
    ctx: Context,
    session: Session,
    name: string
) {
    const roomId = parseInt(name)

    const roomList = Number.isNaN(roomId)
        ? await ctx.database.get('chathub_room', {
              roomName: name
          })
        : await ctx.database.get('chathub_room', {
              roomId: parseInt(name)
          })

    if (roomList.length === 1) {
        return roomList[0] as ConversationRoom
    } else if (roomList.length > 1) {
        // 在限定搜索到群里一次。

        if (session.isDirect || Number.isNaN(roomId)) {
            throw new ChatLunaError(
                ChatLunaErrorCode.THE_NAME_FIND_IN_MULTIPLE_ROOMS
            )
        }

        const groupRoomList = await ctx.database.get(
            'chathub_room_group_member',
            {
                groupId: session.guildId,
                roomId: {
                    $in: roomList.map((it) => it.roomId)
                }
            }
        )

        if (groupRoomList.length === 1) {
            return roomList.find((it) => it.roomId === groupRoomList[0].roomId)
        } else if (groupRoomList.length > 1) {
            throw new ChatLunaError(
                ChatLunaErrorCode.THE_NAME_FIND_IN_MULTIPLE_ROOMS
            )
        }
    } else if (roomList.length === 0) {
        return undefined
    }
}

export async function resolveConversationRoom(ctx: Context, roomId: number) {
    const roomList = await ctx.database.get('chathub_room', {
        roomId
    })

    if (roomList.length > 1) {
        throw new ChatLunaError(
            ChatLunaErrorCode.THE_NAME_FIND_IN_MULTIPLE_ROOMS
        )
    } else if (roomList.length === 0) {
        return undefined
    }

    return roomList[0] as ConversationRoom
}

export async function deleteConversationRoom(
    ctx: Context,
    room: ConversationRoom
) {
    const chatBridger = ctx.chatluna.queryInterfaceWrapper(room, false)

    await chatBridger?.clearChatHistory(room)

    await deleteConversationRoomByRoomId(ctx, room.roomId)

    await ctx.database.remove('chathub_message', {
        conversation: room.conversationId
    })

    await ctx.database.remove('chathub_conversation', {
        id: room.conversationId
    })
}

export async function deleteConversationRoomByRoomId(
    ctx: Context,
    roomId: number
) {
    await ctx.database.remove('chathub_room', {
        roomId
    })

    await ctx.database.remove('chathub_room_member', {
        roomId
    })

    await ctx.database.remove('chathub_room_group_member', {
        roomId
    })

    await ctx.database.remove('chathub_user', {
        defaultRoomId: roomId
    })
}

export async function joinConversationRoom(
    ctx: Context,
    session: Session,
    roomId: number | ConversationRoom,
    isDirect: boolean = session.isDirect,
    userId: string = session.userId
) {
    // 接下来检查房间的权限和当前所处的环境

    const room =
        typeof roomId === 'number'
            ? await resolveConversationRoom(ctx, roomId)
            : roomId

    await ctx.database.upsert('chathub_user', [
        {
            userId,
            defaultRoomId: room.roomId,
            groupId: session.isDirect ? '0' : session.guildId
        }
    ])

    if (isDirect === false) {
        // 如果是群聊，那么就需要检查群聊的权限

        const groupMemberList = await ctx.database.get(
            'chathub_room_group_member',
            {
                groupId: session.guildId,
                roomId: room.roomId
            }
        )

        if (groupMemberList.length === 0) {
            await ctx.database.create('chathub_room_group_member', {
                groupId: session.guildId,
                roomId: room.roomId,
                roomVisibility: room.visibility
            })
        }
    }

    const memberList = await ctx.database.get('chathub_room_member', {
        userId,
        roomId: room.roomId
    })

    if (memberList.length === 0) {
        await ctx.database.create('chathub_room_member', {
            userId,
            roomId: room.roomId,
            roomPermission: userId === room.roomMasterId ? 'owner' : 'member'
        })
    }
}

export async function getConversationRoomUser(
    ctx: Context,
    session: Session,
    roomId: number | ConversationRoom,
    userId: string = session.userId
) {
    const room =
        typeof roomId === 'number'
            ? await resolveConversationRoom(ctx, roomId)
            : roomId

    const memberList = await ctx.database.get('chathub_room_member', {
        roomId: room.roomId,
        userId
    })

    return memberList?.[0]
}

export async function setUserPermission(
    ctx: Context,
    session: Session,
    roomId: number | ConversationRoom,
    permission: 'member' | 'admin',
    userId: string = session.userId
) {
    const room =
        typeof roomId === 'number'
            ? await resolveConversationRoom(ctx, roomId)
            : roomId

    const memberList = await ctx.database.get('chathub_room_member', {
        roomId: room.roomId,
        userId
    })

    if (memberList.length === 0) {
        throw new ChatLunaError(ChatLunaErrorCode.ROOM_NOT_FOUND)
    }

    await ctx.database.upsert('chathub_room_member', [
        {
            userId,
            roomId: room.roomId,
            roomPermission: permission
        }
    ])
}

export async function addConversationRoomToGroup(
    ctx: Context,
    session: Session,
    roomId: number | ConversationRoom,
    groupId: string = session.guildId
) {
    const room =
        typeof roomId === 'number'
            ? await resolveConversationRoom(ctx, roomId)
            : roomId

    const memberList = await ctx.database.get('chathub_room_group_member', {
        roomId: room.roomId,
        groupId
    })

    if (memberList.length === 0) {
        await ctx.database.create('chathub_room_group_member', {
            roomId: room.roomId,
            groupId,
            roomVisibility: room.visibility
        })
    }
}

export async function muteUserFromConversationRoom(
    ctx: Context,
    session: Session,
    roomId: number | ConversationRoom,
    userId: string
) {
    const room =
        typeof roomId === 'number'
            ? await resolveConversationRoom(ctx, roomId)
            : roomId

    const memberList = await ctx.database.get('chathub_room_member', {
        roomId: room.roomId,
        userId
    })

    if (memberList.length === 0) {
        throw new ChatLunaError(ChatLunaErrorCode.ROOM_NOT_JOINED)
    }

    await ctx.database.upsert('chathub_room_member', [
        {
            userId,
            roomId: room.roomId,
            mute: memberList[0].mute !== true
        }
    ])
}

export async function kickUserFromConversationRoom(
    ctx: Context,
    session: Session,
    roomId: number | ConversationRoom,
    userId: string
) {
    const room =
        typeof roomId === 'number'
            ? await resolveConversationRoom(ctx, roomId)
            : roomId

    const memberList = await ctx.database.get('chathub_room_member', {
        roomId: room.roomId,
        userId
    })

    if (memberList.length === 0) {
        throw new ChatLunaError(ChatLunaErrorCode.ROOM_NOT_JOINED)
    }

    await ctx.database.remove('chathub_room_member', {
        roomId: room.roomId,
        userId
    })

    await ctx.database.remove('chathub_user', {
        userId,
        defaultRoomId: room.roomId
    })
}

export async function checkAdmin(session: Session) {
    const tested = await session.app.permissions.test('chatluna:admin', session)

    if (tested) {
        return true
    }

    const user = await session.getUser<User.Field>(session.userId, [
        'authority'
    ])

    return user?.authority >= 3
}

export async function updateChatTime(ctx: Context, room: ConversationRoom) {
    await ctx.database.upsert('chathub_room', [
        {
            roomId: room.roomId,
            updatedTime: new Date()
        }
    ])
}

export async function createConversationRoom(
    ctx: Context,
    session: Session,
    room: ConversationRoom
) {
    // 先向 room 里面插入表

    await ctx.database.create('chathub_room', room)

    // 将创建者加入到房间成员里

    await ctx.database.create('chathub_room_member', {
        userId: session.userId,
        roomId: room.roomId,
        roomPermission:
            session.userId === room.roomMasterId ? 'owner' : 'member'
    })

    await joinConversationRoom(ctx, session, room)
}
