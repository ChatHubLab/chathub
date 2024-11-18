import { Context, Session } from 'koishi'
import { Config } from '../config'

import {
    ChainMiddlewareContext,
    ChainMiddlewareRunStatus,
    ChatChain
} from '../chains/chain'
import crypto from 'crypto'

export function apply(ctx: Context, config: Config, chain: ChatChain) {
    const authService = ctx.chatluna_auth

    chain
        .middleware('chat_time_limit_save', async (session, context) => {
            if (config.authSystem !== true) {
                return await oldChatLimitSave(session, context)
            }

            await authService.increaseAuthGroupCount(
                context.options.authGroup.id
            )

            return ChainMiddlewareRunStatus.CONTINUE
        })
        .after('render_message')

    //  .before("lifecycle-request_model")

    async function oldChatLimitSave(
        session: Session,
        context: ChainMiddlewareContext
    ) {
        const {
            chatLimit,
            chatLimitCache,
            room: { conversationId }
        } = context.options

        /*   console.log(
            await ctx.chatluna_auth._selectCurrentAuthGroup(
                session,
                parseRawModelName(model)[0]
            )
        ) */

        let key = conversationId + '-' + session.userId

        key = crypto.createHash('md5').update(key).digest('hex')

        chatLimit.count++

        // 先保存一次
        await chatLimitCache.set(key, chatLimit)

        return ChainMiddlewareRunStatus.CONTINUE
    }
}

declare module '../chains/chain' {
    interface ChainMiddlewareName {
        chat_time_limit_save: never
    }
}
