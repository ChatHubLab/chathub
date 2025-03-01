import { Context } from 'koishi'
import { Config } from 'koishi-plugin-chatluna-long-memory'
// import start
import { apply as config } from './plugins/config'
import { apply as memory } from './plugins/memory'
import { apply as tool } from './plugins/tool' // import end
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'

export async function plugins(
    ctx: Context,
    parent: Config,
    plugin: ChatLunaPlugin
) {
    type Plugin = (
        ctx: Context,
        config: Config,
        plugin?: ChatLunaPlugin
    ) => PromiseLike<void> | void

    const middlewares: Plugin[] =
        // middleware start
        [config, memory, tool] // middleware end

    for (const middleware of middlewares) {
        await middleware(ctx, parent, plugin)
    }
}
