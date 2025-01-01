/* eslint-disable operator-linebreak */
import { Context, Logger } from 'koishi'
import { Config } from '../config'

import { Message } from '../types'
import { createLogger } from 'koishi-plugin-chatluna/utils/logger'
import { ChainMiddlewareRunStatus, ChatChain } from '../chains/chain'
import { withResolver } from 'koishi-plugin-chatluna/utils/promise'
import { ObjectLock } from 'koishi-plugin-chatluna/utils/lock'

let logger: Logger
const messages: Record<string, Message[]> = {}
const timeouts: Record<string, NodeJS.Timeout> = {}
const promises: Record<string, (messages: Message[]) => void> = {}
const queueLock: Record<string, ObjectLock> = {}

export function apply(ctx: Context, config: Config, chain: ChatChain) {
    logger = createLogger(ctx)
    chain
        .middleware('message_delay', async (session, context) => {
            if (
                config.messageDelay === 0 ||
                (context.command != null && context.command.length > 0)
            ) {
                return ChainMiddlewareRunStatus.CONTINUE
            }

            const { room, inputMessage } = context.options
            const lock = queueLock[room.conversationId] || new ObjectLock()
            queueLock[room.conversationId] = lock

            const unlock = await lock.lock()
            messages[room.conversationId] = messages[room.conversationId] || []
            messages[room.conversationId].push(inputMessage)

            const timeout = timeouts[room.conversationId]
            if (timeout) {
                logger.debug(`trigger message delay, stop the chain`)
                clearTimeout(timeout)
                resetTimeout(room.conversationId, config.messageDelay)
                unlock()
                return ChainMiddlewareRunStatus.STOP
            }
            unlock()

            const { promise, resolve } = withResolver<Message[]>()
            promises[room.conversationId] = resolve
            resetTimeout(room.conversationId, config.messageDelay)

            const delayMessages = await promise
            messages[room.conversationId] = []
            context.options.inputMessage = mergeMessages(delayMessages)

            return ChainMiddlewareRunStatus.CONTINUE
        })
        .after('resolve_room')
}

async function resetTimeout(conversationId: string, delay: number) {
    timeouts[conversationId] = setTimeout(() => {
        delete timeouts[conversationId]
        delete queueLock[conversationId]
        promises[conversationId](messages[conversationId])
    }, delay)
}

function mergeMessages(messages: Message[]) {
    const newMessage: Message = {
        content: messages.map((message) => message.content).join('\n\n'),
        name: messages[0].name,
        conversationId: messages[0].conversationId,
        additional_kwargs: messages[0].additional_kwargs
    }

    for (const message of messages) {
        if (message.additional_kwargs) {
            newMessage.additional_kwargs = {
                ...newMessage.additional_kwargs,
                ...message.additional_kwargs
            }
        }
    }

    return newMessage
}

declare module '../chains/chain' {
    interface ChainMiddlewareName {
        message_delay: never
    }
}
