/* eslint-disable max-len */
import { Document } from '@langchain/core/documents'
import {
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage
} from '@langchain/core/messages'
import { ChatPromptValueInterface } from '@langchain/core/prompt_values'
import {
    BaseChatPromptTemplate,
    BasePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate
} from '@langchain/core/prompts'
import { PartialValues } from '@langchain/core/utils/types'
import { messageTypeToOpenAIRole } from 'koishi-plugin-chatluna/llm-core/utils/count_tokens'

import { logger } from '../..'
import { SystemPrompts } from './base'

export interface ChatHubChatPromptInput {
    systemPrompts?: SystemPrompts
    conversationSummaryPrompt: HumanMessagePromptTemplate
    messagesPlaceholder?: MessagesPlaceholder
    tokenCounter: (text: string) => Promise<number>
    humanMessagePromptTemplate?: HumanMessagePromptTemplate
    sendTokenLimit?: number
}

export class ChatHubChatPrompt
    extends BaseChatPromptTemplate
    implements ChatHubChatPromptInput
{
    systemPrompts?: SystemPrompts

    tokenCounter: (text: string) => Promise<number>

    messagesPlaceholder?: MessagesPlaceholder

    humanMessagePromptTemplate: HumanMessagePromptTemplate

    conversationSummaryPrompt: HumanMessagePromptTemplate

    sendTokenLimit?: number

    constructor(fields: ChatHubChatPromptInput) {
        super({ inputVariables: ['chat_history', 'long_history', 'input'] })

        this.systemPrompts = fields.systemPrompts
        this.tokenCounter = fields.tokenCounter

        this.messagesPlaceholder = fields.messagesPlaceholder
        this.conversationSummaryPrompt = fields.conversationSummaryPrompt
        this.humanMessagePromptTemplate =
            fields.humanMessagePromptTemplate ??
            HumanMessagePromptTemplate.fromTemplate('{input}')
        this.sendTokenLimit = fields.sendTokenLimit ?? 4096
    }

    _getPromptType() {
        return 'chathub_chat' as const
    }

    private async _countMessageTokens(message: BaseMessage) {
        let result =
            (await this.tokenCounter(message.content as string)) +
            (await this.tokenCounter(
                messageTypeToOpenAIRole(message._getType())
            ))

        if (message.name) {
            result += await this.tokenCounter(message.name)
        }

        return result
    }

    async formatMessages({
        chat_history: chatHistory,
        long_history: longHistory,
        input
    }: {
        input: BaseMessage
        chat_history: BaseMessage[] | string
        long_history: Document[]
    }) {
        const result: BaseMessage[] = []
        let usedTokens = 0

        for (const message of this.systemPrompts || []) {
            const messageTokens = await this._countMessageTokens(message)

            // always add the system prompts
            result.push(message)
            usedTokens += messageTokens
        }

        const inputTokens = await this.tokenCounter(input.content as string)

        usedTokens += inputTokens

        let formatConversationSummary: HumanMessage | null
        if (!this.messagesPlaceholder) {
            const chatHistoryTokens = await this.tokenCounter(
                chatHistory as string
            )

            if (usedTokens + chatHistoryTokens > this.sendTokenLimit) {
                logger?.warn(
                    `Used tokens: ${
                        usedTokens + chatHistoryTokens
                    } exceed limit: ${
                        this.sendTokenLimit
                    }. Is too long history. Splitting the history.`
                )
            }

            // splice the chat history
            chatHistory = chatHistory.slice(-chatHistory.length * 0.6)

            if (longHistory.length > 0) {
                const formatDocuments: Document[] = []
                for (const document of longHistory) {
                    const documentTokens = await this.tokenCounter(
                        document.pageContent
                    )

                    // reserve 80 tokens for the format
                    if (
                        usedTokens + documentTokens >
                        this.sendTokenLimit - 80
                    ) {
                        break
                    }

                    usedTokens += documentTokens
                    formatDocuments.push(document)
                }

                formatConversationSummary =
                    await this.conversationSummaryPrompt.format({
                        long_history: formatDocuments
                            .map((document) => document.pageContent)
                            .join(' '),
                        chat_history: chatHistory
                    })
            }
        } else {
            const formatChatHistory: BaseMessage[] = []

            for (const message of (<BaseMessage[]>chatHistory).reverse()) {
                const messageTokens = await this._countMessageTokens(message)

                // reserve 400 tokens for the long history
                if (
                    usedTokens + messageTokens >
                    this.sendTokenLimit - (longHistory.length > 0 ? 480 : 80)
                ) {
                    break
                }

                usedTokens += messageTokens
                formatChatHistory.unshift(message)
            }

            if (longHistory.length > 0) {
                const formatDocuments: Document[] = []

                for (const document of longHistory) {
                    const documentTokens = await this.tokenCounter(
                        document.pageContent
                    )

                    // reserve 80 tokens for the format
                    if (
                        usedTokens + documentTokens >
                        this.sendTokenLimit - 80
                    ) {
                        break
                    }

                    usedTokens += documentTokens
                    formatDocuments.push(document)
                }

                formatConversationSummary =
                    await this.conversationSummaryPrompt.format({
                        long_history: formatDocuments
                            .map((document) => document.pageContent)
                            .join('\n')
                    })
            }

            const formatMessagesPlaceholder =
                await this.messagesPlaceholder.formatMessages({
                    chat_history: formatChatHistory
                })

            result.push(...formatMessagesPlaceholder)
        }

        if (formatConversationSummary) {
            result.push(formatConversationSummary)
            result.push(new AIMessage('Ok. I will remember.'))
        }

        result.push(input)

        logger?.debug(
            `Used tokens: ${usedTokens} exceed limit: ${this.sendTokenLimit}`
        )

        logger?.debug(`messages: ${JSON.stringify(result)}`)

        return result
    }

    partial(
        values: PartialValues
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ): Promise<BasePromptTemplate<any, ChatPromptValueInterface, any>> {
        throw new Error('Method not implemented.')
    }
}

export interface ChatHubBrowsingPromptInput {
    systemPrompts?: SystemPrompts
    conversationSummaryPrompt: SystemMessagePromptTemplate
    messagesPlaceholder?: MessagesPlaceholder
    tokenCounter: (text: string) => Promise<number>
    humanMessagePromptTemplate?: HumanMessagePromptTemplate
    sendTokenLimit?: number
}

export class ChatHubBrowsingPrompt
    extends BaseChatPromptTemplate
    implements ChatHubBrowsingPromptInput
{
    systemPrompts: SystemPrompts

    tokenCounter: (text: string) => Promise<number>

    messagesPlaceholder?: MessagesPlaceholder

    humanMessagePromptTemplate: HumanMessagePromptTemplate

    conversationSummaryPrompt: HumanMessagePromptTemplate

    sendTokenLimit?: number

    constructor(fields: ChatHubBrowsingPromptInput) {
        super({ inputVariables: ['chat_history', 'input'] })

        this.systemPrompts = fields.systemPrompts
        this.tokenCounter = fields.tokenCounter

        this.messagesPlaceholder = fields.messagesPlaceholder
        this.conversationSummaryPrompt = fields.conversationSummaryPrompt
        this.humanMessagePromptTemplate =
            fields.humanMessagePromptTemplate ??
            HumanMessagePromptTemplate.fromTemplate('{input}')
        this.sendTokenLimit = fields.sendTokenLimit ?? 4096
    }

    private async _countMessageTokens(message: BaseMessage) {
        let result = (
            await Promise.all([
                this.tokenCounter(message.content as string),
                this.tokenCounter(messageTypeToOpenAIRole(message._getType()))
            ])
        ).reduce((a, b) => a + b, 0)

        if (message.name) {
            result += await this.tokenCounter(message.name)
        }

        return result
    }

    _getPromptType(): string {
        return 'chatluna_browsing'
    }

    async formatMessages({
        chat_history: chatHistory,
        input,
        long_history: longHistory
    }: {
        input: string
        chat_history: BaseMessage[] | string
        long_history: Document[]
    }) {
        const result: BaseMessage[] = []

        let usedTokens = 0

        const systemMessages = this.systemPrompts

        for (const message of systemMessages) {
            const messageTokens = await this._countMessageTokens(message)

            usedTokens += messageTokens
            result.push(message)
        }

        let formatConversationSummary: SystemMessage
        if (!this.messagesPlaceholder) {
            chatHistory = (chatHistory as BaseMessage[])[0].content as string

            const chatHistoryTokens = await this.tokenCounter(
                chatHistory as string
            )

            if (usedTokens + chatHistoryTokens > this.sendTokenLimit) {
                logger.warn(
                    `Used tokens: ${
                        usedTokens + chatHistoryTokens
                    } exceed limit: ${
                        this.sendTokenLimit
                    }. Is too long history. Splitting the history.`
                )
            }

            // splice the chat history
            chatHistory = chatHistory.slice(-chatHistory.length * 0.6)

            if (longHistory.length > 0) {
                const formatDocuments: Document[] = []
                for (const document of longHistory) {
                    const documentTokens = await this.tokenCounter(
                        document.pageContent
                    )

                    // reserve 80 tokens for the format
                    if (
                        usedTokens + documentTokens >
                        this.sendTokenLimit - 80
                    ) {
                        break
                    }

                    usedTokens += documentTokens
                    formatDocuments.push(document)
                }

                formatConversationSummary =
                    await this.conversationSummaryPrompt.format({
                        long_history: formatDocuments
                            .map((document) => document.pageContent)
                            .join(' '),
                        chat_history: chatHistory
                    })
            }
        } else {
            const formatChatHistory: BaseMessage[] = []

            for (const message of (<BaseMessage[]>chatHistory)
                .slice(-100)
                .reverse()) {
                const messageTokens = await this._countMessageTokens(message)

                // reserve 100 tokens for the long history
                if (usedTokens + messageTokens > this.sendTokenLimit - 1000) {
                    break
                }

                usedTokens += messageTokens
                formatChatHistory.unshift(message)
            }

            if (longHistory.length > 0) {
                const formatDocuments: Document[] = []

                for (const document of longHistory) {
                    const documentTokens = await this.tokenCounter(
                        document.pageContent
                    )

                    // reserve 80 tokens for the format
                    if (
                        usedTokens + documentTokens >
                        this.sendTokenLimit - 80
                    ) {
                        break
                    }

                    usedTokens += documentTokens
                    formatDocuments.push(document)
                }

                formatConversationSummary =
                    await this.conversationSummaryPrompt.format({
                        long_history: formatDocuments
                            .map((document) => document.pageContent)
                            .join(' ')
                    })
            }

            const formatMessagesPlaceholder =
                await this.messagesPlaceholder.formatMessages({
                    chat_history: formatChatHistory
                })

            result.push(...formatMessagesPlaceholder)
        }

        if (formatConversationSummary) {
            // push after system message
            result.splice(1, 0, formatConversationSummary)
            result.splice(2, 0, new AIMessage('Ok.'))
        }

        if (input && input.length > 0) {
            result.push(new HumanMessage(input))
        }

        logger.debug(
            `Used tokens: ${usedTokens} exceed limit: ${this.sendTokenLimit}`
        )

        logger.debug(`messages: ${JSON.stringify(result)}`)

        return result
    }

    partial(
        values: PartialValues
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ): Promise<BasePromptTemplate<any, ChatPromptValueInterface, any>> {
        throw new Error('Method not implemented.')
    }
}
