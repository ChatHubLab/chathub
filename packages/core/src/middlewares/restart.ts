import { Context } from 'koishi'
import { Config } from '../config'
import { ChainMiddlewareRunStatus, ChatChain } from '../chains/chain'

export function apply(ctx: Context, config: Config, chain: ChatChain) {
    chain
        .middleware('restart', async (session, context) => {
            const { command } = context

            if (command !== 'restart') return ChainMiddlewareRunStatus.SKIPPED

            ctx.scope.parent.scope.update(config, true)

            context.message = session.text('.success')

            return ChainMiddlewareRunStatus.STOP
        })
        .before('black_list')
}

declare module '../chains/chain' {
    interface ChainMiddlewareName {
        restart: never
    }
}
