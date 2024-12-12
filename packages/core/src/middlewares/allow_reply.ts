/* eslint-disable operator-linebreak */
import { Context } from 'koishi'
import { Config } from '../config'
import { ChainMiddlewareRunStatus, ChatChain } from '../chains/chain'

export function apply(ctx: Context, config: Config, chain: ChatChain) {
    chain
        .middleware('allow_reply', async (session, context) => {
            // 禁止套娃
            if (ctx.bots[session.uid]) return ChainMiddlewareRunStatus.STOP

            // 私聊检查
            if (
                session.isDirect &&
                config.allowPrivate &&
                (context.command != null || config.privateChatWithoutCommand)
            ) {
                return await checkReplyPermission()
            }

            const botId = session.bot.userId

            // 艾特检查
            if (config.allowAtReply) {
                let appel = session.stripped.appel

                if (appel) {
                    return await checkReplyPermission()
                }

                // 从消息元素中检测是否有被艾特当前用户

                appel = session.elements.some(
                    (element) =>
                        element.type === 'at' && element.attrs?.['id'] === botId
                )

                if (appel) {
                    return await checkReplyPermission()
                }
            }

            // 引用检查
            // 检测回复的消息是否为 bot 本身

            if (config.allowQuoteReply && session.quote?.user?.id === botId) {
                return await checkReplyPermission()
            }

            // bot名字检查
            if (
                session.content.startsWith(config.botName) &&
                config.isNickname
            ) {
                return await checkReplyPermission()
            }

            // 随机回复检查
            if (Math.random() < config.randomReplyFrequency) {
                return await checkReplyPermission()
            }

            // 房间名称匹配检查
            if (config.allowChatWithRoomName) {
                return await checkReplyPermission()
            }

            // 命令检查
            if (context.command != null) {
                return await checkReplyPermission()
            }

            return ChainMiddlewareRunStatus.STOP

            // 辅助函数：检查回复权限
            async function checkReplyPermission() {
                const notReply = await ctx.serial(
                    'chatluna/before-check-sender',
                    session
                )
                return notReply
                    ? ChainMiddlewareRunStatus.STOP
                    : ChainMiddlewareRunStatus.CONTINUE
            }
        })
        .after('lifecycle-check')
}

declare module '../chains/chain' {
    interface ChainMiddlewareName {
        allow_reply: never
    }
}
