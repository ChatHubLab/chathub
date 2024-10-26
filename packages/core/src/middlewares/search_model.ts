import { Context } from 'koishi'

import { ModelType } from 'koishi-plugin-chatluna/llm-core/platform/types'
import { ChainMiddlewareRunStatus, ChatChain } from '../chains/chain'
import { Config } from '../config'
import { Pagination } from 'koishi-plugin-chatluna/utils/pagination'

export function apply(ctx: Context, config: Config, chain: ChatChain) {
    const services = ctx.chatluna.platform

    const pagination = new Pagination<string>({
        formatItem: (value) => value,
        formatString: {
            top: '',
            bottom: '',
            pages: ''
        }
    })

    chain
        .middleware('search_model', async (session, context) => {
            const {
                command,
                options: { page, limit, query }
            } = context

            if (command !== 'search_model')
                return ChainMiddlewareRunStatus.SKIPPED

            pagination.updateFormatString({
                top: session.text('.header') + '\n',
                bottom: '\n' + session.text('.footer'),
                pages: '\n' + session.text('.pages')
            })

            const models = services.getAllModels(ModelType.llm)

            await pagination.push(models)

            context.message = await pagination.searchPage(
                (value) => value.includes(query),
                page,
                limit
            )

            return ChainMiddlewareRunStatus.STOP
        })
        .after('lifecycle-handle_command')
}

declare module '../chains/chain' {
    interface ChainMiddlewareName {
        search_model: never
    }

    interface ChainMiddlewareContextOptions {
        query?: string
    }
}
