import { Tool } from '@langchain/core/tools'
import { Context } from 'koishi'
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'
import { Config } from '..'
import { CreateToolParams } from 'koishi-plugin-chatluna/llm-core/platform/types'
import crypto from 'crypto'
import { VectorStore } from '@langchain/core/vectorstores'

export async function apply(
    ctx: Context,
    config: Config,
    plugin: ChatLunaPlugin
) {
    if (config.memory !== true) {
        return
    }

    plugin.registerTool('memory_search', {
        selector(history) {
            return true
        },
        alwaysRecreate: true,
        async createTool(params, session) {
            return new MemorySearchTool(ctx, params)
        }
    })

    plugin.registerTool('memory_save', {
        selector(history) {
            return true
        },
        alwaysRecreate: true,
        async createTool(params, session) {
            return new MemorySaveTool(ctx, params)
        }
    })
}

export class MemorySearchTool extends Tool {
    name = 'memory_search'

    constructor(
        private ctx: Context,
        private params: CreateToolParams
    ) {
        super({})
    }

    /** @ignore */
    async _call(input: string) {
        const defaultVectorStoreName =
            this.ctx.chatluna.config.defaultVectorStore

        let vectorStore: VectorStore | null = null

        try {
            vectorStore = await this.ctx.chatluna.platform.createVectorStore(
                defaultVectorStoreName,
                {
                    embeddings: this.params.embeddings,
                    key: resolveMemoryKey(
                        this.params.userId,
                        this.params.preset
                    )
                }
            )
        } catch (error) {
            return 'An error occurred while searching for memories.'
        }

        if (!vectorStore) {
            return 'An error occurred while searching for memories.'
        }
        try {
            const result = await vectorStore.similaritySearch(input, 10)

            return result
                .map((item) => {
                    return item?.pageContent
                })
                .join('\n')
        } catch (error) {
            return 'An error occurred while searching for memories.'
        }
    }

    // eslint-disable-next-line max-len
    description = `Searches user-related memories. Input a brief keyword or phrase (e.g., "birthday"). Returns relevant memory snippets for conversation integration.`
}

export class MemorySaveTool extends Tool {
    name = 'memory_save'

    constructor(
        private ctx: Context,
        private params: CreateToolParams
    ) {
        super({})
    }

    /** @ignore */
    async _call(input: string) {
        const defaultVectorStoreName =
            this.ctx.chatluna.config.defaultVectorStore

        const vectorStore = await this.ctx.chatluna.platform.createVectorStore(
            defaultVectorStoreName,
            {
                embeddings: this.params.embeddings,
                key: resolveMemoryKey(this.params.userId, this.params.preset)
            }
        )

        if (!vectorStore) {
            return 'An error occurred while saving memories.'
        }

        await vectorStore.addDocuments([
            {
                pageContent: input,
                metadata: {}
            }
        ])

        await vectorStore.save()

        return 'Memory saved successfully.'
    }

    // eslint-disable-next-line max-len
    description = `Saves key user-related information. Use for important facts, preferences, or memories. Guidelines:

    - Be concise and informative
    - Start directly with the fact (e.g., "Likes pizza" not "The person likes pizza")
    - Focus on facts, preferences, and memories; exclude user/agent details
    - Match the input language (e.g., use Chinese for Chinese input)

    Input: Key points to remember about the user.`
}

function resolveMemoryKey(userId: string, preset: string) {
    const hash = crypto
        .createHash('sha256')
        .update(`${preset}-${userId}`)
        .digest('hex')

    return hash
}
