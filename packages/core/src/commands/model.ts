import { Context } from 'koishi'
import { Config } from '../config'
import { ChatChain } from '../chains/chain'

export function apply(ctx: Context, config: Config, chain: ChatChain) {
    ctx.command('chatluna.model', {
        authority: 1
    })

    ctx.command('chatluna.model.list')
        .option('page', '-p <page:number>')
        .option('limit', '-l <limit:number>')
        .action(async ({ options, session }) => {
            await chain.receiveCommand(session, 'list_model', {
                page: options.page ?? 1,
                limit: options.limit ?? 5
            })
        })

    ctx.command('chatluna.model.search <query:string>')
        .option('page', '-p <page:number>')
        .option('limit', '-l <limit:number>')
        .action(async ({ options, session }, query) => {
            await chain.receiveCommand(session, 'search_model', {
                query,
                page: options.page ?? 1,
                limit: options.limit ?? 5
            })
        })
}
