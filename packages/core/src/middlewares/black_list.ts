import { Context } from 'koishi';
import { Config } from '../config';
import { ChainMiddlewareRunStatus, ChatChain } from '../chains/chain';
import { createLogger } from '../llm-core/utils/logger';

const logger = createLogger("@dingyi222666/chathub/middlewares/black_list")

export function apply(ctx: Context, config: Config, chain: ChatChain) {

    chain.middleware("black_list", async (session, context) => {
        const resolved = await session.resolve(config.blackList)
        if (resolved === true) {
            logger.debug(`[黑名单] ${session.username}(${session.userId}): ${session.content}`)
            context.message = config.blockText
            return ChainMiddlewareRunStatus.STOP
        }
        return ChainMiddlewareRunStatus.CONTINUE
    }).after("allow_reply")
}

declare module '../chains/chain' {
    interface ChainMiddlewareName {
        "black_list": never
    }
}