import { AIMessage, SystemMessage } from '@langchain/core/messages'
import {
    HumanMessagePromptTemplate,
    MessagesPlaceholder
} from '@langchain/core/prompts'
import { FakeEmbeddings } from '@langchain/core/utils/testing'
import { ChainValues } from '@langchain/core/utils/types'
import {
    callChatHubChain,
    ChatHubLLMCallArg,
    ChatHubLLMChain,
    ChatHubLLMChainWrapper,
    SystemPrompts
} from 'koishi-plugin-chatluna/llm-core/chain/base'
import { ChatLunaChatModel } from 'koishi-plugin-chatluna/llm-core/platform/model'
import {
    BufferMemory,
    ConversationSummaryMemory,
    VectorStoreRetrieverMemory
} from 'langchain/memory'
import { MemoryVectorStore } from 'langchain/vectorstores/memory'
import { ChatHubChatPrompt } from './prompt'

export interface ChatHubChatChainInput {
    botName: string
    systemPrompts?: SystemPrompts
    longMemory?: VectorStoreRetrieverMemory
    humanMessagePrompt?: string
    historyMemory: ConversationSummaryMemory | BufferMemory
}

export class ChatHubChatChain
    extends ChatHubLLMChainWrapper
    implements ChatHubChatChainInput
{
    botName: string

    longMemory: VectorStoreRetrieverMemory

    chain: ChatHubLLMChain

    historyMemory: ConversationSummaryMemory | BufferMemory

    systemPrompts?: SystemPrompts

    constructor({
        botName,
        longMemory,
        historyMemory,
        systemPrompts,
        chain
    }: ChatHubChatChainInput & {
        chain: ChatHubLLMChain
    }) {
        super()
        this.botName = botName

        // roll back to the empty memory if not set
        this.longMemory =
            longMemory ??
            new VectorStoreRetrieverMemory({
                vectorStoreRetriever: new MemoryVectorStore(
                    new FakeEmbeddings()
                ).asRetriever(6),
                memoryKey: 'long_history',
                inputKey: 'user',
                outputKey: 'ai',
                returnDocs: true
            })
        this.historyMemory = historyMemory
        this.systemPrompts = systemPrompts
        this.chain = chain
    }

    static fromLLM(
        llm: ChatLunaChatModel,
        {
            botName,
            longMemory,
            historyMemory,
            systemPrompts,
            humanMessagePrompt
        }: ChatHubChatChainInput
    ): ChatHubLLMChainWrapper {
        const humanMessagePromptTemplate =
            HumanMessagePromptTemplate.fromTemplate(
                humanMessagePrompt ?? '{input}'
            )

        let conversationSummaryPrompt: HumanMessagePromptTemplate
        let messagesPlaceholder: MessagesPlaceholder

        if (historyMemory instanceof ConversationSummaryMemory) {
            conversationSummaryPrompt = HumanMessagePromptTemplate.fromTemplate(
                // eslint-disable-next-line max-len
                `Here are some memories about the user. Please generate response based on the system prompt and content below. Relevant pieces of previous conversation: {long_history} (You do not need to use these pieces of information if not relevant, and based on these information, generate similar but non-repetitive responses. Pay attention, you need to think more and diverge your creativity) Current conversation: {chat_history}`
            )
        } else {
            conversationSummaryPrompt = HumanMessagePromptTemplate.fromTemplate(
                // eslint-disable-next-line max-len
                `Here are some memories about the user: {long_history} (You do not need to use these pieces of information if not relevant, and based on these information, generate similar but non-repetitive responses. Pay attention, you need to think more and diverge your creativity.)`
            )

            messagesPlaceholder = new MessagesPlaceholder('chat_history')
        }
        const prompt = new ChatHubChatPrompt({
            systemPrompts: systemPrompts ?? [
                new SystemMessage(
                    "You are ChatGPT, a large language model trained by OpenAI. Carefully heed the user's instructions."
                )
            ],
            conversationSummaryPrompt,
            messagesPlaceholder,
            tokenCounter: (text) => llm.getNumTokens(text),
            humanMessagePromptTemplate,
            sendTokenLimit:
                llm.invocationParams().maxTokenLimit ??
                llm.getModelMaxContextSize()
        })

        const chain = new ChatHubLLMChain({ llm, prompt })

        return new ChatHubChatChain({
            botName,
            longMemory,
            historyMemory,
            systemPrompts,
            chain
        })
    }

    async call({
        message,
        stream,
        events,
        conversationId,
        systemPrompts,
        signal
    }: ChatHubLLMCallArg): Promise<ChainValues> {
        if (systemPrompts != null) {
            const prompt = this.chain.prompt

            if (prompt instanceof ChatHubChatPrompt) {
                prompt.systemPrompts = systemPrompts
            }
        }

        const requests: ChainValues = {
            input: message
        }
        const chatHistory =
            await this.historyMemory.loadMemoryVariables(requests)

        const longHistory = await this.longMemory.loadMemoryVariables({
            user: message.content
        })

        requests['chat_history'] = chatHistory[this.historyMemory.memoryKey]
        requests['long_history'] = longHistory[this.longMemory.memoryKey]
        requests['id'] = conversationId

        const response = await callChatHubChain(
            this.chain,
            {
                ...requests,
                stream,
                signal
            },
            events
        )

        if (response.text == null) {
            throw new Error('response.text is null')
        }

        const responseString = response.text

        /* await this.longMemory.saveContext(
            { user: message.content },
            { ai: responseString }
        ) */

        await this.historyMemory.chatHistory.addMessage(message)
        await this.historyMemory.chatHistory.addAIChatMessage(responseString)

        const aiMessage = new AIMessage(responseString)
        response.message = aiMessage

        if (
            response.extra != null &&
            'additionalReplyMessages' in response.extra
        ) {
            response.additionalReplyMessages =
                response.extra.additionalReplyMessages
        }

        return response
    }

    get model() {
        return this.chain.llm
    }
}
