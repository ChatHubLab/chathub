import { Context } from 'koishi'
import { ModelType } from 'koishi-plugin-chatluna/llm-core/platform/types'
import { ChainMiddlewareRunStatus, ChatChain } from '../chains/chain'
import { Config } from '../config'
import { Pagination } from 'koishi-plugin-chatluna/utils/pagination'

export function apply(ctx: Context, config: Config, chain: ChatChain) {
    const service = ctx.chatluna.platform

    const pagination = new Pagination<string>({
        formatItem: (value) => value,
        formatString: {
            top: '以下是目前可用的嵌入模型列表：\n',
            bottom: '\n你可以使用 chatluna.embeddings.set <model> 来设置默认使用的嵌入模型'
        }
    })

    chain
        .middleware('list_all_embeddings', async (session, context) => {
            const {
                command,
                options: { page, limit }
            } = context

            if (command !== 'list_embeddings')
                return ChainMiddlewareRunStatus.SKIPPED

            const models = service.getAllModels(ModelType.embeddings)

            await pagination.push(models)

            context.message = await pagination.getFormattedPage(page, limit)

            return ChainMiddlewareRunStatus.STOP
        })
        .after('lifecycle-handle_command')
}

declare module '../chains/chain' {
    interface ChainMiddlewareName {
        list_all_embeddings: never
    }
}
