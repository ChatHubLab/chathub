import { Context } from 'koishi'
import { ChatChain } from '../chains/chain'
import { Config } from '../config'

export function apply(ctx: Context, config: Config, chain: ChatChain) {
    if (config.authSystem !== true) return

    ctx.command('chatluna.auth', { authority: 1 })

    ctx.command('chatluna.auth.list')
        .option('page', '-p <page:number>')
        .option('limit', '-l <limit:number>')
        .option('platform', '-t <platform:string>')
        .action(async ({ options, session }) => {
            await chain.receiveCommand(session, 'list_auth_group', {
                authPlatform: options.platform,
                page: options.page ?? 1,
                limit: options.limit ?? 3
            })
        })

    ctx.command('chatluna.auth.add <name:string>')
        .option('user', '-u <user:user>')
        .action(async ({ session, options }, name) => {
            const userId = options.user?.split(':')?.[1] ?? session.userId
            await chain.receiveCommand(session, 'add_user_to_auth_group', {
                auth_group_resolve: { name },
                authUser: userId
            })
        })

    ctx.command('chatluna.auth.kick <name:string>')
        .option('user', '-u <user:user>')
        .action(async ({ session, options }, name) => {
            const userId = options.user?.split(':')?.[1] ?? session.userId
            await chain.receiveCommand(session, 'kick_user_form_auth_group', {
                auth_group_resolve: { name },
                authUser: userId
            })
        })

    ctx.command('chatluna.auth.create')
        .option('name', '-n <name:string>')
        .option('preMin', '-pm <min:number>')
        .option('preDay', '-pd <day:number>')
        .option('platform', '-pf <platform:string>')
        .option('supportModels', '-s [...model]')
        .option('priority', '-p <priority:number>')
        .option('cost', '-c <cost:number>')
        .action(async ({ session, options }) => {
            await chain.receiveCommand(session, 'create_auth_group', {
                auth_group_resolve: {
                    name: options.name ?? undefined,
                    requestPreDay: options.preDay ?? undefined,
                    requestPreMin: options.preMin ?? undefined,
                    platform: options.platform ?? undefined,
                    supportModels: options.supportModels ?? undefined,
                    priority: options.priority ?? undefined
                }
            })
        })

    ctx.command('chatluna.auth.set')
        .option('name', '-n <name:string>')
        .option('preMin', '-pm <min:number>')
        .option('preDay', '-pd <day:number>')
        .option('platform', '-pf <platform:string>')
        .option('supportModels', '-s [...model]')
        .option('priority', '-p <priority:number>')
        .option('cost', '-c <cost:number>')
        .action(async ({ session, options }) => {
            await chain.receiveCommand(session, 'set_auth_group', {
                auth_group_resolve: {
                    name: options.name ?? undefined,
                    requestPreDay: options.preDay ?? undefined,
                    requestPreMin: options.preMin ?? undefined,
                    platform: options.platform ?? undefined,
                    supportModels: options.supportModels ?? undefined,
                    priority: options.priority ?? undefined
                }
            })
        })

    ctx.command('chatluna.balance')

    ctx.command('chatluna.balance.clear <user:user>', { authority: 3 }).action(
        async ({ session }, user) => {
            const userId = user?.split(':')?.[1] ?? user

            await chain.receiveCommand(session, 'clear_balance', {
                authUser: userId
            })
        }
    )

    ctx.command('chatluna.balance.set <balance:number>', { authority: 3 })
        .option('user', '-u <user:user>')
        .action(async ({ options, session }, balance) => {
            const userId = options.user?.split(':')?.[1] ?? session.userId
            await chain.receiveCommand(session, 'set_balance', {
                authUser: userId,
                balance
            })
        })

    ctx.command('chatluna.balance.query [user:user]').action(
        async ({ session }, user) => {
            const userId = user?.split(':')?.[1] ?? session.userId

            await chain.receiveCommand(session, 'query_balance', {
                authUser: userId
            })
        }
    )
}
