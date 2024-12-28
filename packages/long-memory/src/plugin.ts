import { Context } from 'koishi'
import { Config } from 'koishi-plugin-chatluna-long-memory'
// import start
import { apply as config } from './plugins/config'
import { apply as memory } from './plugins/memory'
// import end

export async function plugins(ctx: Context, parent: Config) {
    type Plugin = (ctx: Context, config: Config) => PromiseLike<void> | void

    const middlewares: Plugin[] =
        // middleware start
        [config, memory] // middleware end

    for (const middleware of middlewares) {
        await middleware(ctx, parent)
    }
}
