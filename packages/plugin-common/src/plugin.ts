import { Context } from 'koishi'
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'
import { Config } from '.'
// import start
import { apply as code_sandbox } from './plugins/code_sandbox'
import { apply as command } from './plugins/command'
import { apply as cron } from './plugins/cron'
import { apply as draw } from './plugins/draw'
import { apply as fs } from './plugins/fs'
import { apply as group } from './plugins/group'
import { apply as knowledge } from './plugins/knowledge'
import { apply as music } from './plugins/music'
import { apply as openapi } from './plugins/openapi'
import { apply as request } from './plugins/request'
import { apply as think } from './plugins/think' // import end

export async function plugin(
    ctx: Context,
    config: Config,
    plugin: ChatLunaPlugin
) {
    type Plugin = (
        ctx: Context,
        config: Config,
        plugin: ChatLunaPlugin
    ) => PromiseLike<void> | void

    const middlewares: Plugin[] =
        // middleware start
        [
            code_sandbox,
            command,
            cron,
            draw,
            fs,
            group,
            knowledge,
            music,
            openapi,
            request,
            think
        ] // middleware end

    for (const middleware of middlewares) {
        await middleware(ctx, config, plugin)
    }
}
