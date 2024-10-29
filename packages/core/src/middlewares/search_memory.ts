import { Context, Session } from 'koishi'

import { ChainMiddlewareRunStatus, ChatChain } from '../chains/chain'
import { Config } from '../config'
import { Pagination } from 'koishi-plugin-chatluna/utils/pagination'
import { Document } from '@langchain/core/documents'
import crypto from 'crypto'
import { parseRawModelName } from 'koishi-plugin-chatluna/llm-core/utils/count_tokens'

export function apply(ctx: Context, config: Config, chain: ChatChain) {
    const services = ctx.chatluna.platform

    const pagination = new Pagination<Document>({
        formatItem: (value) => '',
        formatString: {
            top: '',
            bottom: '',
            pages: ''
        }
    })

    chain
        .middleware('search_memory', async (session, context) => {
            let {
                command,
                options: { page, limit, query, type, room }
            } = context

            if (!type) {
                type = room.preset
            }

            if (command !== 'search_memory')
                return ChainMiddlewareRunStatus.SKIPPED

            pagination.updateFormatString({
                top: session.text('.header', [query, type]) + '\n',
                bottom: session.text('.footer'),
                pages: session.text('.pages')
            })

            pagination.updateFormatItem((value) =>
                formatDocumentInfo(session, value)
            )

            const [platform, modelName] = parseRawModelName(
                config.defaultEmbeddings
            )
            const embeddings = await ctx.chatluna.createEmbeddings(
                platform,
                modelName
            )

            const key = resolveLongMemoryId(type, session.userId)

            const vectorStore = await services.createVectorStore(
                config.defaultVectorStore,
                { embeddings, key }
            )

            const documents = await vectorStore
                .similaritySearch(query, 10000)
                .then((value) =>
                    value.sort((a, b) => {
                        const aRawId = (a.metadata?.raw_id as string) ?? ''
                        const bRawId = (b.metadata?.raw_id as string) ?? ''

                        return aRawId.localeCompare(bRawId)
                    })
                )

            await pagination.push(documents)

            context.message = await pagination.getFormattedPage(page, limit)

            return ChainMiddlewareRunStatus.STOP
        })
        .after('lifecycle-handle_command')
}

declare module '../chains/chain' {
    interface ChainMiddlewareName {
        search_memory: never
    }

    interface ChainMiddlewareContextOptions {
        query?: string
    }
}

async function formatDocumentInfo(session: Session, document: Document) {
    const buffer = []

    buffer.push(session.text('.document_id', [document.metadata?.raw_id]))
    buffer.push(session.text('.document_content', [document.pageContent]))

    buffer.push('\n')

    return buffer.join('\n')
}

export function resolveLongMemoryId(preset: string, userId: string) {
    const hash = crypto
        .createHash('sha256')
        .update(`${preset}-${userId}`)
        .digest('hex')

    return hash
}
