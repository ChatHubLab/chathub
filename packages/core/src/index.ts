/* eslint-disable @typescript-eslint/no-var-requires */
import { Context, Logger, User } from 'koishi'
import { ChatLunaService } from 'koishi-plugin-chatluna/services/chat'
import { forkScopeToDisposable } from 'koishi-plugin-chatluna/utils/koishi'
import {
    clearLogger,
    createLogger,
    setLoggerLevel
} from 'koishi-plugin-chatluna/utils/logger'
import * as request from 'koishi-plugin-chatluna/utils/request'
import { PromiseLikeDisposable } from 'koishi-plugin-chatluna/utils/types'
import { ChatLunaAuthService } from './authorization/service'
import { command } from './command'
import { Config } from './config'
import { defaultFactory } from './llm-core/chat/default'
import { apply as longMemory } from './llm-core/memory/history'
import { apply as loreBook } from './llm-core/memory/lore_book'
import { middleware } from './middleware'
import { deleteConversationRoom } from 'koishi-plugin-chatluna/chains'
import { ConversationRoom } from './types'

export * from './config'
export const name = 'chatluna'
export const inject = {
    required: ['database'],
    optional: ['censor', 'vits', 'puppeteer']
}
export const inject2 = {
    database: { required: true },
    censor: { required: false },
    vits: { required: false },
    puppeteer: { required: false }
}

export let logger: Logger

export const usage = `
## chatluna v1.0 beta

### 目前插件还在 beta 阶段，可能会有很多 bug，可以去插件主页那边提 issue 或加群反馈。

ChatLuna 插件交流群：282381753 （有问题不知道怎么弄先加群问）

群里目前没有搭载该插件的 bot，加群的话最好是来询问问题或者提出意见的

[文档](https://chatluna.chat) 也在缓慢制作中，有问题可以在群里提出。
`

export function apply(ctx: Context, config: Config) {
    logger = createLogger(ctx)
    setupLogger(config)
    setupI18n(ctx)

    const disposables: PromiseLikeDisposable[] = []

    ctx.on('ready', async () => {
        setupProxy(ctx, config)
        await setupServices(ctx, config, disposables)
        await setupPermissions(ctx, disposables)
        await setupEntryPoint(ctx, config, disposables)
    })

    ctx.on('dispose', async () => {
        clearLogger()
        disposables.forEach((disposable) => disposable())
    })
}

async function setupEntryPoint(
    ctx: Context,
    config: Config,
    disposables: PromiseLikeDisposable[]
) {
    const entryPointPlugin = (ctx: Context, config: Config) => {
        ctx.on('ready', async () => {
            await initializeComponents(ctx, config)
        })

        setupMiddleware(ctx)
    }

    const entryPointDisposable = forkScopeToDisposable(
        ctx.plugin(
            {
                apply: entryPointPlugin,
                inject: {
                    ...inject2,
                    chatluna: { required: true },
                    chatluna_auth: { required: false },
                    database: { required: false }
                },
                name: 'chatluna_entry_point'
            },
            config
        )
    )
    disposables.push(entryPointDisposable)
}

async function initializeComponents(ctx: Context, config: Config) {
    await defaultFactory(ctx, ctx.chatluna.platform)
    await middleware(ctx, config)
    await command(ctx, config)
    await ctx.chatluna.preset.init()
    await setupAutoDelete(ctx, config)
    longMemory(ctx, config)
    loreBook(ctx, config)
}

function setupMiddleware(ctx: Context) {
    ctx.middleware(async (session, next) => {
        if (ctx.chatluna == null || ctx.chatluna.chatChain == null) {
            return next()
        }
        return next(async (_) => {
            await ctx.chatluna.chatChain.receiveMessage(session, ctx)
        })
    })
}

function setupLogger(config: Config) {
    if (config.isLog) {
        setLoggerLevel(Logger.DEBUG)
    }
}

function setupI18n(ctx: Context) {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    ctx.i18n.define('zh-CN', require('./locales/zh-CN'))
    ctx.i18n.define('en-US', require('./locales/en-US'))
}

function setupProxy(ctx: Context, config: Config) {
    if (config.isProxy) {
        request.setGlobalProxyAddress(
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            config.proxyAddress ?? (ctx.http.config as any)?.proxyAgent
        )
        logger.debug(
            'global proxy %c',
            config.proxyAddress,
            request.globalProxyAddress
        )
    }
}

async function setupServices(
    ctx: Context,
    config: Config,
    disposables: PromiseLikeDisposable[]
) {
    disposables.push(
        forkScopeToDisposable(ctx.plugin(ChatLunaService, config)),
        forkScopeToDisposable(ctx.plugin(ChatLunaAuthService, config))
    )
}

async function setupPermissions(
    ctx: Context,
    disposables: PromiseLikeDisposable[]
) {
    const adminPermissionDisposable = ctx.permissions.define('chatluna:admin', {
        inherits: ['authority.3']
    })
    disposables.push(() => {
        adminPermissionDisposable()
    })

    const adminProviderDisposable = ctx.permissions.provide(
        'chatluna:admin',
        async (name, session) => {
            return (
                (
                    await session.getUser<User.Field>(session.userId, [
                        'authority'
                    ])
                )?.authority >= 3
            )
        }
    )
    disposables.push(() => {
        adminProviderDisposable()
    })
}

async function setupAutoDelete(ctx: Context, config: Config) {
    if (!config.autoDelete) {
        return
    }

    async function execute() {
        const rooms = await ctx.database.get('chathub_room', {
            updatedTime: {
                $lt: new Date(Date.now() - config.autoDeleteTimeout * 1000)
            }
        })

        if (rooms.length === 0) {
            return
        }

        const success: ConversationRoom[] = []

        for (const room of rooms) {
            try {
                await deleteConversationRoom(ctx, room)
                success.push(room)
            } catch (e) {
                logger.error(e)
            }
        }

        logger.success(
            `auto delete %c rooms [%c]`,
            rooms.length,
            success.map((room) => room.roomName).join(',')
        )

        return ctx.setTimeout(execute, 30 * 60 * 1000)
    }

    await execute()
}
