import { Context } from 'koishi'
import {
    AIMessage,
    BaseMessage,
    BaseMessageFields,
    HumanMessage,
    MessageContent,
    MessageType,
    SystemMessage
} from '@langchain/core/messages'
import { v4 as uuidv4 } from 'uuid'
import { BaseChatMessageHistory } from '@langchain/core/chat_history'

export class KoishiChatMessageHistory extends BaseChatMessageHistory {
    // eslint-disable-next-line @typescript-eslint/naming-convention
    lc_namespace: string[] = ['llm-core', 'memory', 'message']

    conversationId: string

    private _ctx: Context
    private _latestId: string
    private _serializedChatHistory: ChatLunaMessage[]
    private _chatHistory: BaseMessage[]
    // eslint-disable-next-line @typescript-eslint/naming-convention
    private _additional_kwargs: Record<string, string>
    private _updatedAt: Date
    constructor(
        ctx: Context,
        conversationId: string,
        private _maxMessagesCount: number
    ) {
        super()

        this.conversationId = conversationId
        this._ctx = ctx
        this._chatHistory = []
        this._additional_kwargs = {}
        this._updatedAt = new Date(0)
    }

    // eslint-disable-next-line @typescript-eslint/naming-convention
    get additionalArgs() {
        return this._additional_kwargs
    }

    async getMessages(): Promise<BaseMessage[]> {
        const latestUpdateTime = await this.getLatestUpdateTime()

        if (
            latestUpdateTime > this._updatedAt ||
            this._chatHistory.length === 0
        ) {
            this._chatHistory = await this._loadMessages()
        }

        return this._chatHistory
    }

    async addUserMessage(message: string): Promise<void> {
        const humanMessage = new HumanMessage(message)
        await this._saveMessage(humanMessage)
    }

    async addAIChatMessage(message: string): Promise<void> {
        const aiMessage = new AIMessage(message)
        await this._saveMessage(aiMessage)
    }

    async addMessage(message: BaseMessage): Promise<void> {
        await this._saveMessage(message)
    }

    async clear(): Promise<void> {
        await this._ctx.database.remove('chathub_message', {
            conversation: this.conversationId
        })

        await this._ctx.database.upsert('chathub_conversation', [
            {
                id: this.conversationId,
                latestId: null
            }
        ])

        this._serializedChatHistory = []
        this._chatHistory = []
        this._latestId = null
    }

    async delete(): Promise<void> {
        await this._ctx.database.remove('chathub_conversation', {
            id: this.conversationId
        })
    }

    async updateAdditionalArg(key: string, value: string): Promise<void> {
        await this.loadConversation()
        this._additional_kwargs[key] = value
        await this._saveConversation()
    }

    async getAdditionalArg(key: string): Promise<string> {
        await this.loadConversation()

        return this._additional_kwargs[key]
    }

    async getAdditionalArgs(): Promise<{ [key: string]: string }> {
        await this.loadConversation()
        return this._additional_kwargs
    }

    async deleteAdditionalArg(key: string): Promise<void> {
        await this.loadConversation()
        delete this._additional_kwargs[key]
        await this._saveConversation()
    }

    async overrideAdditionalArgs(kwargs: {
        [key: string]: string
    }): Promise<void> {
        await this.loadConversation()
        this._additional_kwargs = Object.assign(this._additional_kwargs, kwargs)
        await this._saveConversation()
    }

    private async getLatestUpdateTime(): Promise<Date> {
        const conversation = (
            await this._ctx.database.get(
                'chathub_conversation',
                {
                    id: this.conversationId
                },
                ['updatedAt']
            )
        )?.[0]

        return conversation?.updatedAt ?? new Date(0)
    }

    private async _loadMessages(): Promise<BaseMessage[]> {
        const queried = await this._ctx.database.get('chathub_message', {
            conversation: this.conversationId
        })

        const sorted: ChatLunaMessage[] = []

        let currentMessageId = this._latestId

        let isBad = false

        if (currentMessageId == null && queried.length > 0) {
            isBad = true
        }

        while (currentMessageId != null && !isBad) {
            const currentMessage = queried.find(
                (item) => item.id === currentMessageId
            )

            if (!currentMessage) {
                isBad = true
                break
            }

            sorted.unshift(currentMessage)

            currentMessageId = currentMessage.parent
        }

        if (isBad) {
            this._ctx.logger.warn(
                `Bad conversation detected for %s`,
                this.conversationId
            )

            sorted.length = 0

            await this.clear()
        }

        this._serializedChatHistory = sorted

        return sorted.map((item) => {
            // eslint-disable-next-line @typescript-eslint/naming-convention
            const kw_args = JSON.parse(item.additional_kwargs ?? '{}')
            const content = JSON.parse(item.text as string) as MessageContent
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const fields: BaseMessageFields = {
                content,
                id: item.rawId ?? undefined,
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                additional_kwargs: kw_args as any
            }
            if (item.role === 'system') {
                return new SystemMessage(fields)
            } else if (item.role === 'human') {
                return new HumanMessage(fields)
            } else if (item.role === 'ai') {
                return new AIMessage(fields)
            } else {
                throw new Error('Unknown role')
            }
        })
    }

    private async _loadConversation() {
        const conversation = (
            await this._ctx.database.get('chathub_conversation', {
                id: this.conversationId
            })
        )?.[0]

        if (conversation) {
            this._latestId = conversation.latestId
            this._additional_kwargs =
                conversation.additional_kwargs != null
                    ? JSON.parse(conversation.additional_kwargs)
                    : {}
        } else {
            await this._ctx.database.create('chathub_conversation', {
                id: this.conversationId
            })
        }

        if (!this._serializedChatHistory) {
            await this._loadMessages()
            this._updatedAt = conversation?.updatedAt ?? new Date(0)
        }
    }

    async loadConversation() {
        if (!this._serializedChatHistory) {
            await this._loadConversation()
        }
    }

    private async _saveMessage(message: BaseMessage) {
        const lastedMessage = this._serializedChatHistory.find(
            (item) => item.id === this._latestId
        )

        let additionalArgs = Object.assign({}, message.additional_kwargs)

        delete additionalArgs['preset']
        delete additionalArgs['raw_content']
        delete additionalArgs['type']

        if (Object.keys(additionalArgs).length === 0) {
            additionalArgs = null
        }

        const serializedMessage: ChatLunaMessage = {
            id: uuidv4(),
            text: JSON.stringify(message.content),
            parent: lastedMessage?.id ?? null,
            role: message.getType(),
            additional_kwargs: additionalArgs
                ? JSON.stringify(additionalArgs)
                : null,
            rawId: message.id ?? null,
            conversation: this.conversationId
        }

        await this._ctx.database.upsert('chathub_message', [serializedMessage])

        this._serializedChatHistory.push(serializedMessage)
        this._chatHistory.push(message)
        this._latestId = serializedMessage.id

        const updatedAt = new Date()

        if (this._serializedChatHistory.length > this._maxMessagesCount) {
            const toDeleted = this._serializedChatHistory.splice(
                0,
                this._serializedChatHistory.length - this._maxMessagesCount
            )

            if (
                (toDeleted[toDeleted.length - 1].role === 'human' &&
                    this._serializedChatHistory[0]?.role === 'ai') ||
                this._serializedChatHistory[0]?.role === 'function'
            ) {
                toDeleted.push(this._serializedChatHistory.shift())
            }

            await this._ctx.database.remove('chathub_message', {
                id: toDeleted.map((item) => item.id)
            })

            // update latest message

            const firstMessage = this._serializedChatHistory[0]

            // first message
            firstMessage.parent = null

            await this._ctx.database.upsert('chathub_message', [firstMessage])

            // fetch latest message
            this._chatHistory = await this._loadMessages()
        }

        this._updatedAt = updatedAt

        await this._saveConversation(updatedAt)
    }

    private async _saveConversation(time: Date = new Date()) {
        await this._ctx.database.upsert('chathub_conversation', [
            {
                id: this.conversationId,
                latestId: this._latestId,
                additional_kwargs: JSON.stringify(this._additional_kwargs),
                updatedAt: time
            }
        ])
    }
}

declare module 'koishi' {
    interface Tables {
        chathub_conversation: ChatLunaConversation
        chathub_message: ChatLunaMessage
    }
}

export interface ChatLunaMessage {
    text: MessageContent
    id: string
    rawId?: string
    role: MessageType
    conversation: string
    additional_kwargs?: string
    parent?: string
}

export interface ChatLunaConversation {
    id: string
    latestId?: string
    additional_kwargs?: string
    updatedAt?: Date
}
