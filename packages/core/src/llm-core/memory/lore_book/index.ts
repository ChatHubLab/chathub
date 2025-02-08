import { Context } from 'koishi'
import { Config, logger } from 'koishi-plugin-chatluna'
import { AIMessage, BaseMessage } from '@langchain/core/messages'
import {
    PresetTemplate,
    RoleBook
} from 'koishi-plugin-chatluna/llm-core/prompt'

export function apply(ctx: Context, config: Config): void {
    const cache = new Map<PresetTemplate, LoreBookMatcher>()

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

            if (!preset.loreBooks || preset.loreBooks.items.length === 0) {
                return
            }

            let matcher = cache.get(preset)
            if (!matcher) {
                const loreBooks = preset.loreBooks.items
                matcher = new LoreBookMatcher(loreBooks, {
                    scanDepth: preset.loreBooks?.scanDepth,
                    recursiveScan: preset.loreBooks?.recursiveScan,
                    maxRecursionDepth: preset.loreBooks?.maxRecursionDepth
                })
                cache.set(preset, matcher)
            }

            const messages = [
                ...(await chatInterface.chatHistory.getMessages())
            ]

            messages.push(message)

            const matchedLores = matcher.matchLoreBooks(messages)

            if (matchedLores.length > 0) {
                logger.debug(
                    `Found ${matchedLores.length} matched lore books: ${JSON.stringify(
                        matchedLores.map((lore) => lore.keywords)
                    )}`
                )
                promptVariables['lore_books'] = matchedLores
            }
        }
    )

    ctx.on(
        'chatluna/clear-chat-history',
        async (conversationId, chatInterface) => {
            cache.clear()
        }
    )
}

export class LoreBookMatcher {
    private loreBooks: RoleBook[]
    private defaultConfig: LoreBookConfig
    private regexCache: Map<string, RegExp>

    constructor(
        loreBooks: RoleBook[],
        defaultConfig: Partial<LoreBookConfig> = {}
    ) {
        this.loreBooks = loreBooks
        this.defaultConfig = {
            scanDepth: defaultConfig.scanDepth ?? 2,
            recursiveScan: defaultConfig.recursiveScan ?? true,
            maxRecursionDepth: defaultConfig.maxRecursionDepth ?? 3,
            matchWholeWord: defaultConfig.matchWholeWord ?? false,
            caseSensitive: defaultConfig.caseSensitive ?? true
        }
        this.regexCache = new Map()
    }

    matchLoreBooks(messages: BaseMessage[]): RoleBook[] {
        const matchedLores = new Set<RoleBook>()

        const recentMessages = messages.slice().reverse()

        this.stackMatch(recentMessages, matchedLores)

        return Array.from(matchedLores).sort(
            (a, b) => (a.order ?? 0) - (b.order ?? 0)
        )
    }

    private stackMatch(
        messages: BaseMessage[],
        matchedLores: Set<RoleBook>
    ): void {
        const stack: [BaseMessage[], number][] = [[messages, 0]]

        while (stack.length > 0) {
            const [currentMessages, depth] = stack.pop()!

            for (const loreBook of this.loreBooks) {
                if (loreBook.enabled === false || matchedLores.has(loreBook)) {
                    continue
                }

                const config = this.getConfig(loreBook)
                if (depth >= config.maxRecursionDepth) {
                    continue
                }

                // 根据 loreBook 的 scanDepth 裁剪消息
                const relevantMessages = currentMessages.slice(
                    0,
                    config.scanDepth
                )

                for (const message of relevantMessages) {
                    const content = message.content as string

                    const contentParts = this.splitContent(config, content)

                    for (const part of contentParts) {
                        if (
                            !this.matchKeywords(part, loreBook) &&
                            !loreBook.constant
                        ) {
                            continue
                        }

                        matchedLores.add(loreBook)

                        if (config.recursiveScan) {
                            stack.push([
                                this.splitContent(config, loreBook.content).map(
                                    (c) => new AIMessage(c)
                                ),
                                depth + 1
                            ])
                        }

                        break
                    }
                }
            }
        }
    }

    private matchKeywords(content: string, loreBook: RoleBook): boolean {
        return loreBook.keywords.some((keyword) => {
            const regex = this.getRegexFromKeyword(keyword, loreBook)
            return regex.test(content)
        })
    }

    private getRegexFromKeyword(
        keyword: string | RegExp,
        loreBook: RoleBook
    ): RegExp {
        if (keyword instanceof RegExp) {
            return keyword
        }

        const cacheKey = `${keyword}:${loreBook.caseSensitive}:${loreBook.matchWholeWord}`
        let regex = this.regexCache.get(cacheKey)

        if (!regex) {
            regex = this.createRegexFromKeyword(keyword, loreBook)
            this.regexCache.set(cacheKey, regex)
        }

        return regex
    }

    private splitContent(config: LoreBookConfig, content: string): string[] {
        if (config.matchWholeWord) {
            // 按照中英文标点符号和空格分割
            return content
                .split(/[。！？；；.!?;,，。！？、；：\s]+/g)
                .filter(Boolean)
        }

        return [content]
    }

    private createRegexFromKeyword(
        keyword: string,
        loreBook: RoleBook
    ): RegExp {
        let flags = 'g'
        if (!loreBook.caseSensitive) {
            flags += 'i'
        }

        const pattern = loreBook.matchWholeWord ? `\\b${keyword}\\b` : keyword
        return new RegExp(pattern, flags)
    }

    private getConfig(loreBook: RoleBook): LoreBookConfig {
        return {
            scanDepth: loreBook.scanDepth ?? this.defaultConfig.scanDepth,
            recursiveScan:
                loreBook.recursiveScan ?? this.defaultConfig.recursiveScan,
            maxRecursionDepth:
                loreBook.maxRecursionDepth ??
                this.defaultConfig.maxRecursionDepth,
            matchWholeWord:
                loreBook.matchWholeWord ?? this.defaultConfig.matchWholeWord,
            caseSensitive:
                loreBook.caseSensitive ?? this.defaultConfig.caseSensitive
        }
    }
}

interface LoreBookConfig {
    scanDepth: number
    recursiveScan: boolean
    maxRecursionDepth: number
    matchWholeWord: boolean
    caseSensitive: boolean
}
