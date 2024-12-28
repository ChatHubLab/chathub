import { Context } from 'koishi'
import { ModelType } from 'koishi-plugin-chatluna/llm-core/platform/types'
import { parseRawModelName } from 'koishi-plugin-chatluna/llm-core/utils/count_tokens'
import { ChainMiddlewareRunStatus, ChatChain } from '../chains/chain'
import { Config } from '../config'

export function apply(ctx: Context, config: Config, chain: ChatChain) {
    const service = ctx.chatluna.platform

    chain
        .middleware('set_default_embeddings', async (session, context) => {
            const { command, options } = context

            if (command !== 'set_embeddings')
                return ChainMiddlewareRunStatus.SKIPPED

            const { setEmbeddings } = options

            if (!setEmbeddings) {
                context.message = session.text('.usage_hint')
                return ChainMiddlewareRunStatus.STOP
            }

            const embeddings = service.getAllModels(ModelType.embeddings)

            const [, modelName] = parseRawModelName(setEmbeddings)

            const targetEmbeddings = embeddings.filter((embeddingsName) => {
                return embeddingsName.includes(modelName)
            })

            if (targetEmbeddings.length > 1) {
                const buffer: string[] = []

                buffer.push(session.text('.multiple_models_found.header'))

                for (const embedding of targetEmbeddings) {
                    buffer.push(embedding)
                }

                buffer.push(session.text('.multiple_models_found.footer'))

                buffer.push(
                    session.text('.multiple_models_found.example', [
                        targetEmbeddings[0]
                    ])
                )

                context.message = buffer.join('\n')
            } else if (targetEmbeddings.length === 0) {
                context.message = session.text('.model_not_found')
            }

            const fullName = targetEmbeddings[0]

            await context.send(session.text('.success', [fullName]))

            config.defaultEmbeddings = fullName

            ctx.scope.parent.scope.update(config, true)

            return ChainMiddlewareRunStatus.STOP
        })
        .after('lifecycle-handle_command')
}

export interface EmbeddingsInfo {
    providerName: string
    model: string
}

declare module '../chains/chain' {
    interface ChainMiddlewareName {
        set_default_embeddings: never
    }

    interface ChainMiddlewareContextOptions {
        setEmbeddings?: string
    }
}
