import { Context } from 'koishi'
import { Config } from '../config'
import { ChainMiddlewareRunStatus, ChatChain } from '../chains/chain'
import { Pagination } from 'koishi-plugin-chatluna/utils/pagination'

export function apply(ctx: Context, config: Config, chain: ChatChain) {
    const service = ctx.chatluna.platform

    const pagination = new Pagination<string>({
        formatItem: (value) => value,
        formatString: {
            top: '',
            bottom: '',
            pages: ''
        }
    })

    chain
        .middleware('list_all_vectorstore', async (session, context) => {
            const {
                command,
                options: { page, limit }
            } = context

            if (command !== 'list_vector_store')
                return ChainMiddlewareRunStatus.SKIPPED

            pagination.updateFormatString({
                top: session.text('.header') + '\n',
                bottom: '\n' + session.text('.footer'),
                pages: '\n' + session.text('.pages')
            })

            const vectorStoreProviders = service.getVectorStores()

            await pagination.push(vectorStoreProviders)

            context.message = await pagination.getFormattedPage(page, limit)

            return ChainMiddlewareRunStatus.STOP
        })
        .after('lifecycle-handle_command')
}

declare module '../chains/chain' {
    interface ChainMiddlewareName {
        list_all_vectorstore: never
    }
}
