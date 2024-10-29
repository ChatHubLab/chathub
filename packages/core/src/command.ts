import { Context } from 'koishi'
import { ChatChain } from 'koishi-plugin-chatluna/chains'
import { Config } from './config'
// import start
import { apply as auth } from './commands/auth'
import { apply as chat } from './commands/chat'
import { apply as memory } from './commands/memory'
import { apply as model } from './commands/model'
import { apply as preset } from './commands/preset'
import { apply as providers } from './commands/providers'
import { apply as room } from './commands/room' // import end

export async function command(ctx: Context, config: Config) {
    type Command = (
        ctx: Context,
        config: Config,
        chain: ChatChain
    ) => PromiseLike<void> | void

    const middlewares: Command[] =
        // middleware start
        [auth, chat, memory, model, preset, providers, room] // middleware end

    for (const middleware of middlewares) {
        await middleware(ctx, config, ctx.chatluna.chatChain)
    }
}
