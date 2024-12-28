import { Context } from 'koishi'
import { apply as config } from './plugins/config'
import { apply as memory } from './plugins/memory'import { apply as config } from './plugins/config'
import { apply as memory } from './plugins/memory'import { Config } from '.'
// import start
// import end

export async function plugins(ctx: Context, parent: Config) {
    type Command = (ctx: Context, config: Config) => PromiseLike<void> | void

    const middlewares: Command[] =
        // middleware start
[
config,
memory,
]// middleware end

    for (const middleware of middlewares) {
        await middleware(ctx, parent)
    }
}
