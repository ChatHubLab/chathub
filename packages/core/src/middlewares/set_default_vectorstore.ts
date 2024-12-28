import { Context } from 'koishi'
import { Config } from '../config'
import { ChainMiddlewareRunStatus, ChatChain } from '../chains/chain'

export function apply(ctx: Context, config: Config, chain: ChatChain) {
    const service = ctx.chatluna.platform

    chain
        .middleware('set_default_vectorstore', async (session, context) => {
            const { command, options } = context

            if (command !== 'set_vector_store')
                return ChainMiddlewareRunStatus.SKIPPED

            const { setVectorStore } = options

            if (!setVectorStore) {
                context.message = session.text('.usage_hint')
                return ChainMiddlewareRunStatus.STOP
            }

            const targetVectorStoreProviders = service
                .getVectorStores()
                .filter((vectorStoreProviderName) =>
                    vectorStoreProviderName.includes(setVectorStore)
                )

            if (targetVectorStoreProviders.length > 1) {
                const buffer: string[] = []

                buffer.push(session.text('.multiple_stores_found.header'))

                for (const vectorStoreProvider of targetVectorStoreProviders) {
                    buffer.push(vectorStoreProvider)
                }

                buffer.push(session.text('.multiple_stores_found.footer'))

                buffer.push(
                    session.text('.multiple_stores_found.example', [
                        targetVectorStoreProviders[0]
                    ])
                )

                context.message = buffer.join('\n')
                return ChainMiddlewareRunStatus.STOP
            } else if (targetVectorStoreProviders.length === 0) {
                context.message = session.text('.store_not_found')
                return ChainMiddlewareRunStatus.STOP
            }

            const targetProviderName = targetVectorStoreProviders[0]

            const keysCache = ctx.chatluna.cache

            await keysCache.set('default-vector-store', targetProviderName)

            await context.send(session.text('.success', [targetProviderName]))

            config.defaultVectorStore = targetProviderName

            ctx.scope.parent.scope.update(config, true)

            return ChainMiddlewareRunStatus.STOP
        })
        .after('lifecycle-handle_command')
}

declare module '../chains/chain' {
    interface ChainMiddlewareName {
        set_default_vectorstore: never
    }

    interface ChainMiddlewareContextOptions {
        setVectorStore?: string
    }
}
