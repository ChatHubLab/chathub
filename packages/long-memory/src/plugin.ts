import { Context } from 'koishi'
// import start
import { apply as add_memory } from './plugins/add_memory'
import { apply as chat_middleware } from './plugins/chat_middleware'
import { apply as clear_memory } from './plugins/clear_memory'
import { apply as config } from './plugins/config'
import { apply as delete_memory } from './plugins/delete_memory'
import { apply as edit_memory } from './plugins/edit_memory'
import { apply as search_memory } from './plugins/search_memory'
import { apply as tool } from './plugins/tool' // import end
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'
import { Config } from '.'

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
        [
            add_memory,
            chat_middleware,
            clear_memory,
            config,
            delete_memory,
            edit_memory,
            search_memory,
            tool
        ] // middleware end

    for (const middleware of middlewares) {
        await middleware(ctx, parent, plugin)
    }
}
