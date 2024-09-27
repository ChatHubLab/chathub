import { randomUUID } from 'crypto'
import { Context, Logger, Session } from 'koishi'
import { ModelType } from 'koishi-plugin-chatluna/llm-core/platform/types'
import { createLogger } from 'koishi-plugin-chatluna/utils/logger'
import {
    ChainMiddlewareContext,
    ChainMiddlewareContextOptions,
    ChainMiddlewareRunStatus,
    ChatChain
} from '../chains/chain'
import {
    createConversationRoom,
    getConversationRoomCount
} from '../chains/rooms'
import { Config } from '../config'
import { ConversationRoom } from '../types'

let logger: Logger

export function apply(ctx: Context, config: Config, chain: ChatChain) {
    logger = createLogger(ctx)
    const service = ctx.chatluna.platform

    chain
        .middleware('create_room', async (session, context) => {
            const {
                command,
                options: { room_resolve }
            } = context

            if (command !== 'create_room')
                return ChainMiddlewareRunStatus.SKIPPED

            if (!room_resolve) return ChainMiddlewareRunStatus.SKIPPED

            let { model, preset, name, chatMode, password, visibility } =
                room_resolve

            logger.debug(
                `[create_room] model: ${model}, length: ${
                    Object.values(room_resolve).filter((value) => value != null)
                        .length
                }, visibility: ${visibility}`
            )

            if (
                Object.values(room_resolve).filter((value) => value != null)
                    .length > 0 &&
                visibility !== 'template'
            ) {
                await context.send(session.text('.confirm_create'))

                const result = await session.prompt(1000 * 30)

                if (result == null) {
                    context.message = session.text('.timeout')
                    return ChainMiddlewareRunStatus.STOP
                }

                if (result === 'Y') {
                    room_resolve.preset =
                        room_resolve.preset ?? config.defaultPreset
                    room_resolve.name = room_resolve.name ?? '未命名房间'
                    room_resolve.chatMode =
                        room_resolve.chatMode ?? config.defaultChatMode
                    room_resolve.password = room_resolve.password ?? null
                    room_resolve.visibility =
                        room_resolve.visibility ?? 'private'
                    room_resolve.model =
                        room_resolve.model ?? config.defaultModel

                    await createRoom(ctx, context, session, context.options)

                    return ChainMiddlewareRunStatus.STOP
                } else if (result !== 'N') {
                    context.message = session.text('.cancelled')
                    return ChainMiddlewareRunStatus.STOP
                }
            }

            // 交互式创建

            // 1. 输入房间名

            if (name == null) {
                await context.send(session.text('.enter_name'))

                const result = await session.prompt(1000 * 30)

                if (result == null) {
                    context.message = session.text('.timeout')
                    return ChainMiddlewareRunStatus.STOP
                } else if (result === 'Q') {
                    context.message = session.text('.cancelled')
                    return ChainMiddlewareRunStatus.STOP
                }

                name = result.trim()
                room_resolve.name = name
            } else {
                await context.send(
                    session.text('.change_or_keep', [
                        session.text('.action.input'),
                        session.text('.field.name'),
                        name
                    ])
                )

                const result = await session.prompt(1000 * 30)

                if (result == null) {
                    context.message = session.text('.timeout')
                    return ChainMiddlewareRunStatus.STOP
                } else if (result === 'Q') {
                    context.message = session.text('.cancelled')
                    return ChainMiddlewareRunStatus.STOP
                } else if (result !== 'N') {
                    name = result.trim()
                    room_resolve.name = name
                }
            }

            // 2. 选择模型

            while (true) {
                let preModel = model
                if (preModel == null) {
                    await context.send(session.text('.enter_model'))

                    const result = await session.prompt(1000 * 30)

                    if (result == null) {
                        context.message = session.text('.timeout')
                        return ChainMiddlewareRunStatus.STOP
                    } else if (result === 'Q') {
                        context.message = session.text('.cancelled')
                        return ChainMiddlewareRunStatus.STOP
                    }

                    preModel = result.trim()
                } else {
                    await context.send(
                        session.text('.change_or_keep', [
                            session.text('.action.select'),
                            session.text('.field.model'),
                            preModel
                        ])
                    )

                    const result = await session.prompt(1000 * 30)

                    if (result == null) {
                        context.message = session.text('.timeout')
                        return ChainMiddlewareRunStatus.STOP
                    } else if (result === 'Q') {
                        context.message = session.text('.cancelled')
                        return ChainMiddlewareRunStatus.STOP
                    } else if (result !== 'N') {
                        preModel = result.trim()
                    }
                }

                const findModel = service
                    .getAllModels(ModelType.llm)
                    .find((searchModel) => searchModel === preModel)

                if (findModel == null) {
                    await context.send(
                        session.text('.model_not_found', [preModel])
                    )
                    preModel = null
                    room_resolve.model = null
                    continue
                } else {
                    model = preModel
                    room_resolve.model = model
                    break
                }
            }

            // 3. 选择预设

            const presetInstance = ctx.chatluna.preset
            while (true) {
                let prePreset = preset
                if (preset == null) {
                    await context.send(session.text('.enter_preset'))

                    const result = await session.prompt(1000 * 30)

                    if (result == null) {
                        context.message = session.text('.timeout')
                        return ChainMiddlewareRunStatus.STOP
                    } else if (result === 'Q') {
                        context.message = session.text('.cancelled')
                        return ChainMiddlewareRunStatus.STOP
                    } else if (result === 'N') {
                        prePreset = 'chatgpt'
                    } else {
                        prePreset = result.trim()
                    }
                } else {
                    await context.send(
                        session.text('.change_or_keep', [
                            session.text('.action.select'),
                            session.text('.field.preset'),
                            prePreset
                        ])
                    )

                    const result = await session.prompt(1000 * 30)

                    if (result == null) {
                        context.message = session.text('.timeout')
                        return ChainMiddlewareRunStatus.STOP
                    } else if (result === 'Q') {
                        context.message = session.text('.cancelled')
                        return ChainMiddlewareRunStatus.STOP
                    } else if (result !== 'N') {
                        prePreset = result.trim()
                    }
                }

                try {
                    await presetInstance.getPreset(prePreset)
                    preset = prePreset
                    room_resolve.preset = preset
                    break
                } catch {
                    await context.send(
                        session.text('.preset_not_found', [prePreset])
                    )
                    room_resolve.preset = null
                    continue
                }
            }

            // 4. 可见性
            while (true) {
                if (visibility == null) {
                    await context.send(session.text('.enter_visibility'))

                    const result = await session.prompt(1000 * 30)

                    if (result == null) {
                        context.message = session.text('.timeout')
                        return ChainMiddlewareRunStatus.STOP
                    } else if (result === 'Q') {
                        context.message = session.text('.cancelled')
                        return ChainMiddlewareRunStatus.STOP
                    } else if (result === 'N') {
                        room_resolve.visibility = 'private'
                    } else {
                        room_resolve.visibility = result.trim()
                    }
                } else {
                    await context.send(
                        session.text('.change_or_keep', [
                            session.text('.action.select'),
                            session.text('.field.visibility'),
                            visibility
                        ])
                    )

                    const result = await session.prompt(1000 * 30)

                    if (result == null) {
                        context.message = session.text('.timeout')
                        return ChainMiddlewareRunStatus.STOP
                    } else if (result === 'Q') {
                        context.message = session.text('.cancelled')
                        return ChainMiddlewareRunStatus.STOP
                    } else if (result !== 'N') {
                        room_resolve.visibility = result.trim()
                    }
                }

                visibility = room_resolve.visibility

                if (visibility === 'private' || visibility === 'public') {
                    break
                }

                await context.send(
                    session.text('.visibility_not_recognized', [visibility])
                )
            }

            // 5. 聊天模式

            if (chatMode == null) {
                await context.send(session.text('.enter_chat_mode'))

                const result = await session.prompt(1000 * 30)

                if (result == null) {
                    context.message = session.text('.timeout')
                    return ChainMiddlewareRunStatus.STOP
                } else if (result === 'Q') {
                    context.message = session.text('.cancelled')
                    return ChainMiddlewareRunStatus.STOP
                } else if (result === 'N') {
                    room_resolve.chatMode = 'chat'
                } else {
                    room_resolve.chatMode = result.trim()
                }
            } else {
                await context.send(
                    session.text('.change_or_keep', [
                        session.text('.action.select'),
                        session.text('.field.chat_mode'),
                        chatMode
                    ])
                )

                const result = await session.prompt(1000 * 30)

                if (result == null) {
                    context.message = session.text('.timeout')
                    return ChainMiddlewareRunStatus.STOP
                } else if (result === 'Q') {
                    context.message = session.text('.cancelled')
                    return ChainMiddlewareRunStatus.STOP
                } else if (result !== 'N') {
                    room_resolve.chatMode = result.trim()
                }
            }

            chatMode = room_resolve.chatMode

            // 6. 密码
            if (
                session.isDirect &&
                visibility === 'private' &&
                password == null
            ) {
                await context.send(session.text('.enter_password'))

                const result = await session.prompt(1000 * 30)

                if (result == null) {
                    context.message = session.text('.timeout')
                    return ChainMiddlewareRunStatus.STOP
                } else if (result === 'Q') {
                    context.message = session.text('.cancelled')
                    return ChainMiddlewareRunStatus.STOP
                } else if (result === 'N') {
                    room_resolve.password = null
                } else {
                    room_resolve.password = result.trim()
                }
            }

            // 7. 创建房间
            await createRoom(ctx, context, session, context.options)

            return ChainMiddlewareRunStatus.STOP
        })
        .after('lifecycle-handle_command')
}

async function createRoom(
    ctx: Context,
    context: ChainMiddlewareContext,
    session: Session,
    options: ChainMiddlewareContextOptions
) {
    const { model, preset, name, chatMode, password, visibility } =
        options.room_resolve

    const createRoom: ConversationRoom = {
        conversationId: randomUUID(),
        model,
        preset,
        roomName: name ?? '未命名房间',
        roomMasterId: session.userId,
        roomId: (await getConversationRoomCount(ctx)) + 1,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        visibility: visibility as any,
        chatMode,
        password: password ?? null,
        updatedTime: new Date()
    }

    await createConversationRoom(ctx, session, createRoom)

    if (visibility === 'template') {
        context.message = session.text('.template_room_created')
    } else {
        context.message = session.text('.room_created', [
            createRoom.roomId,
            createRoom.roomName
        ])
    }
}

declare module '../chains/chain' {
    interface ChainMiddlewareName {
        create_room: never
    }

    interface ChainMiddlewareContextOptions {
        room_resolve?: {
            conversationId?: string
            model?: string
            preset?: string
            name?: string
            chatMode?: string
            id?: string
            password?: string
            visibility?: string
        }
    }
}
