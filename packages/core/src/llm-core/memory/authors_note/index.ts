import { Context } from 'koishi'
import { Config } from 'koishi-plugin-chatluna'

export function apply(ctx: Context, config: Config): void {
    const cache = new Map<string, AuthorsNoteCache>()

    ctx.on(
        'chatluna/before-chat',
        async (
            conversationId,
            message,
            promptVariables,
            chatInterface,
            chain
        ) => {
            const preset = await chatInterface.preset

            const authorsNote = preset.authorsNote

            if (!authorsNote || authorsNote.insertFrequency === 0) {
                return
            }

            const authorsNoteCache = cache.get(conversationId) || {
                chatCount: 1
            }

            if (
                authorsNote.insertFrequency > 0 &&
                authorsNoteCache.chatCount % authorsNote.insertFrequency !== 0
            ) {
                return
            }

            cache.set(conversationId, authorsNoteCache)

            promptVariables['authors_note'] = authorsNote
        }
    )

    ctx.on('chatluna/after-chat', async (conversationId, chatInterface) => {
        let authorsNoteCache = cache.get(conversationId)
        if (!authorsNoteCache) {
            authorsNoteCache = {
                chatCount: 0
            }
            cache.set(conversationId, authorsNoteCache)
        }

        authorsNoteCache.chatCount++
    })

    ctx.on(
        'chatluna/clear-chat-history',
        async (conversationId, chatInterface) => {
            cache.delete(conversationId)
        }
    )
}

interface AuthorsNoteCache {
    chatCount: number
}
