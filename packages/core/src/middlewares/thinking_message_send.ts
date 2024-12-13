import { Context, Logger } from 'koishi'
import { Config } from '../config'
import {
    ChainMiddlewareContextOptions,
    ChainMiddlewareRunStatus,
    ChatChain
} from '../chains/chain'
import { createLogger } from 'koishi-plugin-chatluna/utils/logger'

let logger: Logger

export function apply(ctx: Context, config: Config, chain: ChatChain) {
    logger = createLogger(ctx)
    chain
        .middleware('thinking_message_send', async (session, context) => {
            if (!config.sendThinkingMessage || context.command?.length > 0) {
                return ChainMiddlewareRunStatus.SKIPPED
            }

            const thinkingTimeoutObject: ThinkingTimeoutObject = {}
            context.options.thinkingTimeoutObject = thinkingTimeoutObject

            thinkingTimeoutObject.timeout = setTimeout(async () => {
                const queueCount = await getQueueCount(
                    thinkingTimeoutObject,
                    context.options
                )

                if (thinkingTimeoutObject.timeout == null || queueCount < 1) {
                    return
                }

                const messageIds = await session.send(
                    session.text('chatluna.thinking_message', [
                        (queueCount ?? '0').toString()
                    ])
                )

                thinkingTimeoutObject.recallFunc = async () => {
                    try {
                        await session.bot.deleteMessage(
                            session.channelId,
                            messageIds[0]
                        )
                    } catch (e) {
                        logger.error(e)
                    }
                    thinkingTimeoutObject.autoRecallTimeout = undefined
                    thinkingTimeoutObject.timeout = undefined
                }

                thinkingTimeoutObject.autoRecallTimeout = setTimeout(
                    () => {
                        thinkingTimeoutObject.recallFunc?.()
                        thinkingTimeoutObject.autoRecallTimeout = undefined
                    },
                    1000 * 60 * 2 - 1000 * 3
                )
            }, config.sendThinkingMessageTimeout)

            return ChainMiddlewareRunStatus.CONTINUE
        })
        .before('lifecycle-prepare')
}

async function getQueueCount(
    obj: ThinkingTimeoutObject,
    options: ChainMiddlewareContextOptions
) {
    await new Promise((resolve, reject) => {
        const timer = setInterval(() => {
            if (obj.timeout != null && options.queueCount != null) {
                clearInterval(timer)
                resolve(undefined)
            }
        })
    })

    return options.queueCount
}

export interface ThinkingTimeoutObject {
    timeout?: NodeJS.Timeout
    recallFunc?: () => PromiseLike<void>
    autoRecallTimeout?: NodeJS.Timeout
}

declare module '../chains/chain' {
    interface ChainMiddlewareContextOptions {
        thinkingTimeoutObject?: ThinkingTimeoutObject
    }

    interface ChainMiddlewareName {
        thinking_message_send: never
    }
}
