import { Context, Session } from 'koishi'
import { parseRawModelName } from 'koishi-plugin-chatluna/llm-core/utils/count_tokens'
import {
    ChatLunaError,
    ChatLunaErrorCode
} from 'koishi-plugin-chatluna/utils/error'
import { ChatHubAuthGroup } from '../authorization/types'
import { Cache } from '../cache'
import {
    ChainMiddlewareContext,
    ChainMiddlewareRunStatus,
    ChatChain
} from '../chains/chain'
import crypto from 'crypto'
import { Config } from '../config'

export function apply(ctx: Context, config: Config, chain: ChatChain) {
    const chatLimitCache = new Cache(ctx, config, 'chathub/chat_limit')
    const platformService = ctx.chatluna.platform
    const authService = ctx.chatluna_auth

    chain
        .middleware('chat_time_limit_check', async (session, context) => {
            if (config.authSystem !== true) {
                return await oldChatLimitCheck(session, context)
            }

            const {
                room: { model }
            } = context.options

            // check account balance
            const authUser = await authService.getUser(session)

            if (
                authUser /* && context.command == null */ &&
                authUser.balance <= 0
            ) {
                context.message = session.text(
                    'chatluna.insufficient_balance',
                    [authUser.balance]
                )
                return ChainMiddlewareRunStatus.STOP
            }

            let authGroup = await authService.resolveAuthGroup(
                session,
                parseRawModelName(model)[0]
            )

            if (
                authGroup.supportModels != null &&
                authGroup.supportModels.find((m) => m === model) == null
            ) {
                context.message = session.text('chatluna.unsupported_model', [
                    authGroup.name,
                    model
                ])
                return ChainMiddlewareRunStatus.STOP
            }

            authGroup = await authService.resetAuthGroup(authGroup.id)

            context.options.authGroup = authGroup

            // check pre min

            if (
                (authGroup.currentLimitPerMin ?? 0) + 1 >
                authGroup.limitPerMin
            ) {
                context.message = session.text(
                    'chatluna.limit_per_minute_exceeded',
                    [
                        authGroup.name,
                        authGroup.limitPerMin,
                        authGroup.currentLimitPerMin
                    ]
                )

                return ChainMiddlewareRunStatus.STOP
            }

            if (
                (authGroup.currentLimitPerDay ?? 0) + 1 >
                authGroup.limitPerDay
            ) {
                context.message = session.text(
                    'chatluna.limit_per_day_exceeded',
                    [
                        authGroup.name,
                        authGroup.limitPerDay,
                        authGroup.currentLimitPerDay
                    ]
                )

                return ChainMiddlewareRunStatus.STOP
            }

            return ChainMiddlewareRunStatus.CONTINUE
        })
        .after('resolve_model')
        .before('request_model')

    async function oldChatLimitCheck(
        session: Session,
        context: ChainMiddlewareContext
    ) {
        const {
            room: { model, conversationId }
        } = context.options

        const config = platformService.getConfigs(
            parseRawModelName(model)[0]
        )?.[0]

        if (!config) {
            throw new ChatLunaError(
                ChatLunaErrorCode.MODEL_ADAPTER_NOT_FOUND,
                new Error(`Can't find model adapter for ${model}`)
            )
        }

        const chatLimitRaw = config.value.chatLimit

        const chatLimitComputed = await session.resolve(chatLimitRaw)

        let key = conversationId + '-' + session.userId

        // md5

        key = crypto.createHash('md5').update(key).digest('hex')

        let chatLimitOnDataBase = await chatLimitCache.get(key)

        if (chatLimitOnDataBase) {
            // 如果大于1小时的间隔，就重置
            if (Date.now() - chatLimitOnDataBase.time > 1000 * 60 * 60) {
                chatLimitOnDataBase = {
                    time: Date.now(),
                    count: 0
                }
            } else {
                // 用满了
                if (chatLimitOnDataBase.count >= chatLimitComputed) {
                    const time = Math.ceil(
                        (1000 * 60 * 60 -
                            (Date.now() - chatLimitOnDataBase.time)) /
                            1000 /
                            60
                    )

                    context.message = session.text(
                        'chatluna.chat_limit_exceeded',
                        [time]
                    )

                    return ChainMiddlewareRunStatus.STOP
                } else {
                    chatLimitOnDataBase.count++
                }
            }
        } else {
            chatLimitOnDataBase = {
                time: Date.now(),
                count: 0
            }
        }

        // 先保存一次
        await chatLimitCache.set(key, chatLimitOnDataBase)

        context.options.chatLimit = chatLimitOnDataBase
        context.options.chatLimitCache = chatLimitCache

        return ChainMiddlewareRunStatus.CONTINUE
    }
}

declare module '../chains/chain' {
    interface ChainMiddlewareName {
        chat_time_limit_check: never
    }

    interface ChainMiddlewareContextOptions {
        chatLimitCache?: Cache<'chathub/chat_limit', ChatLimit>
        chatLimit?: ChatLimit
        authGroup?: ChatHubAuthGroup
    }
}

declare module '@koishijs/cache' {
    interface Tables {
        'chathub/chat_limit': ChatLimit
    }
}

export interface ChatLimit {
    time: number
    count: number
}
