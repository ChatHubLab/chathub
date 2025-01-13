import { Context, Logger, Session, sleep } from 'koishi'
import {
    formatPresetTemplateString,
    PresetTemplate
} from 'koishi-plugin-chatluna/llm-core/prompt'
import { parseRawModelName } from 'koishi-plugin-chatluna/llm-core/utils/count_tokens'
import {
    ChatLunaError,
    ChatLunaErrorCode
} from 'koishi-plugin-chatluna/utils/error'
import { createLogger } from 'koishi-plugin-chatluna/utils/logger'
import {
    ChainMiddlewareContext,
    ChainMiddlewareRunStatus,
    ChatChain
} from 'koishi-plugin-chatluna/chains'
import { Config } from '../config'
import { ConversationRoom, Message } from '../types'
import { renderMessage } from './render_message'
import {
    getCurrentWeekday,
    getNotEmptyString,
    getTimeDiffFormat,
    PresetPostHandler
} from 'koishi-plugin-chatluna/utils/string'
import { updateChatTime } from '../chains/rooms'
import { BufferText } from '../utils/buffer_text'
import { v4 as uuidv4 } from 'uuid'

let logger: Logger

const requestIdCache = new Map<string, string>()

export function apply(ctx: Context, config: Config, chain: ChatChain) {
    logger = createLogger(ctx)
    chain
        .middleware('request_model', async (session, context) => {
            const { room, inputMessage } = context.options

            const presetTemplate = await ctx.chatluna.preset.getPreset(
                room.preset
            )

            if (presetTemplate.formatUserPromptString != null) {
                context.message = formatUserPromptString(
                    config,
                    presetTemplate,
                    session,
                    inputMessage.content,
                    room
                )

                inputMessage.content = context.message as string
            }

            const bufferText = new BufferText(
                room.chatMode === 'plugin',
                3,
                presetTemplate.config?.postHandler?.prefix,
                presetTemplate.config?.postHandler?.postfix
            )

            const postHandler = presetTemplate.config?.postHandler
                ? new PresetPostHandler(
                      ctx,
                      config,
                      presetTemplate.config?.postHandler
                  )
                : undefined

            let isFirstResponse = true

            if (config.streamResponse) {
                setTimeout(async () => {
                    await handleMessage(
                        context,
                        session,
                        config,
                        bufferText,
                        (message) => sendMessage(context, message, config)
                    )
                }, 0)
            }

            let responseMessage: Message

            inputMessage.conversationId = room.conversationId
            inputMessage.name =
                session.author?.name ?? session.author?.id ?? session.username

            const requestId = createRequestId(session, room)

            logger.debug(
                `create request id: ${requestId} for ${session.userId} in ${room.roomName}-${room.conversationId}`
            )

            try {
                responseMessage = await ctx.chatluna.chat(
                    session,
                    room,
                    inputMessage,
                    {
                        // eslint-disable-next-line @typescript-eslint/naming-convention
                        'llm-new-token': async (token) => {
                            if (token === '') {
                                return
                            }

                            if (isFirstResponse) {
                                isFirstResponse = false
                                await context?.recallThinkingMessage()
                            }

                            await bufferText.addText(token)
                        },
                        // eslint-disable-next-line @typescript-eslint/naming-convention
                        'llm-queue-waiting': async (count) => {
                            context.options.queueCount = count
                        },
                        // eslint-disable-next-line @typescript-eslint/naming-convention
                        'llm-call-tool': async (tool, arg) => {
                            if (!config.showThoughtMessage) {
                                return
                            }

                            context.send(formatToolCall(tool, arg))
                        },
                        // eslint-disable-next-line @typescript-eslint/naming-convention
                        'llm-used-token-count': async (tokens) => {
                            if (config.authSystem !== true) {
                                return
                            }
                            const balance =
                                await ctx.chatluna_auth.calculateBalance(
                                    session,
                                    parseRawModelName(room.model)[0],
                                    tokens
                                )

                            logger.debug(`current balance: ${balance}`)
                        }
                    },
                    config.streamResponse,
                    getSystemPromptVariables(session, config, room),
                    postHandler,
                    requestId
                )
            } catch (e) {
                if (e?.message?.includes('output values have 1 keys')) {
                    throw new ChatLunaError(
                        ChatLunaErrorCode.MODEL_RESPONSE_IS_EMPTY
                    )
                } else {
                    throw e
                }
            } finally {
                bufferText.end()
            }

            if (!config.streamResponse) {
                context.options.responseMessage = responseMessage
            } else {
                context.options.responseMessage = null
                context.message = null
            }

            await updateChatTime(ctx, room)

            return ChainMiddlewareRunStatus.CONTINUE
        })
        .after('lifecycle-request_model')
}

async function handleMessage(
    context: ChainMiddlewareContext,
    session: Session,
    config: Config,
    bufferText: BufferText,
    sendMessageFunc: (text: string) => Promise<void>
) {
    const isEditMessage = session.bot.editMessage != null
    if (isEditMessage) {
        try {
            await handleEditMessage(
                context,
                session,
                config,
                bufferText,
                sendMessageFunc
            )

            return
        } catch (error) {
            logger.error('Error handling edit message:', error)
        }
    }

    const getText = (() => {
        if (config.splitMessage) {
            return bufferText.splitByPunctuations.bind(bufferText)
        }
        return bufferText.splitByMarkdown.bind(bufferText)
    })() as () => AsyncGenerator<string, void, unknown>

    for await (const text of getText()) {
        try {
            await sendMessageFunc(text)
        } catch (error) {
            logger.error('Error sending message:', error)
        }
    }
}

async function handleEditMessage(
    context: ChainMiddlewareContext,
    session: Session,
    config: Config,
    bufferText: BufferText,
    sendMessage: (text: string) => Promise<void>
) {
    const { ctx } = context

    let messageId: string | null = null
    const queue: string[] = []
    let isFinished = false

    const editMessage = async (text: string) => {
        try {
            await session.bot.editMessage(
                session.channelId,
                messageId,
                text // await markdownRenderMessage(text)
            )
        } catch (error) {
            logger.error('Error editing message:', error)
        }
    }

    const processQueue = async () => {
        // eslint-disable-next-line no-unmodified-loop-condition
        while (!isFinished) {
            const firstQueue = queue.shift()
            if (firstQueue == null) {
                await sleep(2)
                continue
            }
            await editMessage(firstQueue)
        }

        if (queue.length > 0) {
            await editMessage(queue.shift())
        }
    }

    setTimeout(async () => {
        await processQueue()
    }, 0)

    for await (let text of bufferText.getCached()) {
        if (config.censor) {
            text = await ctx.censor.transform(text, session)
        }

        if (messageId == null) {
            try {
                messageId = await session.bot
                    .sendMessage(session.channelId, text)
                    .then((messageIds) => messageIds[0])
            } catch (error) {
                logger.error('Error sending message:', error)
            }
            continue
        }

        queue.unshift(text)
    }

    isFinished = true
}

function getSystemPromptVariables(
    session: Session,
    config: Config,
    room: ConversationRoom
) {
    return {
        name: config.botName,
        date: new Date().toLocaleString(),
        bot_id: session.bot.selfId,
        is_group: (!session.isDirect || session.guildId != null).toString(),
        is_private: session.isDirect?.toString(),
        user_id: session.author?.user?.id ?? session.event?.user?.id ?? '0',
        user: getNotEmptyString(
            session.author?.nick,
            session.author?.name,
            session.event.user?.name,
            session.username
        ),
        noop: '',
        time: new Date().toLocaleTimeString(),
        weekday: getCurrentWeekday(),
        idle_duration: getTimeDiffFormat(
            new Date().getTime(),
            room.updatedTime.getTime()
        )
    }
}

export function getRequestId(session: Session, room: ConversationRoom) {
    const userKey =
        session.userId +
        '-' +
        (session.guildId ?? '') +
        '-' +
        room.conversationId

    return requestIdCache.get(userKey)
}

export function createRequestId(session: Session, room: ConversationRoom) {
    const requestId = uuidv4()

    const userKey =
        session.userId +
        '-' +
        (session.guildId ?? '') +
        '-' +
        room.conversationId

    requestIdCache.set(userKey, requestId)

    return requestId
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function formatToolCall(tool: string, arg: string) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any

    return `{\n  tool: '${tool}',\n  log: '${arg}'\n}`
}

function formatUserPromptString(
    config: Config,
    presetTemplate: PresetTemplate,
    session: Session,
    prompt: string,
    room: ConversationRoom
) {
    return formatPresetTemplateString(presetTemplate.formatUserPromptString, {
        sender_id: session.author?.user?.id ?? session.event?.user?.id ?? '0',

        sender: getNotEmptyString(
            session.author?.nick,
            session.author?.name,
            session.event.user?.name,
            session.username
        ),
        prompt,
        ...getSystemPromptVariables(session, config, room)
    })
}

async function sendMessage(
    context: ChainMiddlewareContext,
    text: string,
    config: Config
) {
    if (text == null || text.trim() === '') {
        return
    }

    const renderedMessage = await renderMessage(
        context.ctx,
        {
            content: text
        },
        context.options.renderOptions
    )

    if (config.censor) {
        for (const key in renderedMessage) {
            renderedMessage[key] = await context.ctx.censor.transform(
                renderedMessage[key],
                context.session
            )
        }
    }

    await context.send(renderedMessage)
}

declare module '../chains/chain' {
    interface ChainMiddlewareName {
        request_model: never
    }

    interface ChainMiddlewareContextOptions {
        responseMessage?: Message
        inputMessage?: Message
        queueCount?: number
    }
}
