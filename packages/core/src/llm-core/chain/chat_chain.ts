import { AIMessage } from '@langchain/core/messages'
import { ChainValues } from '@langchain/core/utils/types'
import {
    callChatLunaChain,
    ChatLunaLLMCallArg,
    ChatLunaLLMChain,
    ChatLunaLLMChainWrapper
} from 'koishi-plugin-chatluna/llm-core/chain/base'
import { ChatLunaChatModel } from 'koishi-plugin-chatluna/llm-core/platform/model'
import { BufferMemory } from 'koishi-plugin-chatluna/llm-core/memory/langchain'
import { ChatLunaChatPrompt } from 'koishi-plugin-chatluna/llm-core/chain/prompt'
import { PresetTemplate } from 'koishi-plugin-chatluna/llm-core/prompt'

export interface ChatHubChatChainInput {
    botName: string
    preset: () => Promise<PresetTemplate>
    humanMessagePrompt?: string
    historyMemory: BufferMemory
}

export class ChatHubChatChain
    extends ChatLunaLLMChainWrapper
    implements ChatHubChatChainInput
{
    botName: string

    chain: ChatLunaLLMChain

    historyMemory: BufferMemory

    preset: () => Promise<PresetTemplate>

    constructor({
        botName,
        historyMemory,
        preset,
        chain
    }: ChatHubChatChainInput & {
        chain: ChatLunaLLMChain
    }) {
        super()
        this.botName = botName

        this.historyMemory = historyMemory
        this.preset = preset
        this.chain = chain
    }

    static fromLLM(
        llm: ChatLunaChatModel,
        { botName, historyMemory, preset }: ChatHubChatChainInput
    ): ChatLunaLLMChainWrapper {
        const prompt = new ChatLunaChatPrompt({
            preset,
            tokenCounter: (text) => llm.getNumTokens(text),
            sendTokenLimit:
                llm.invocationParams().maxTokenLimit ??
                llm.getModelMaxContextSize()
        })

        const chain = new ChatLunaLLMChain({ llm, prompt })

        return new ChatHubChatChain({
            botName,
            historyMemory,
            preset,
            chain
        })
    }

    async call({
        message,
        stream,
        events,
        conversationId,
        variables,
        signal
    }: ChatLunaLLMCallArg): Promise<ChainValues> {
        const requests: ChainValues = {
            input: message
        }
        const chatHistory =
            await this.historyMemory.loadMemoryVariables(requests)

        requests['chat_history'] = chatHistory[this.historyMemory.memoryKey]
        requests['variables'] = variables ?? {}
        requests['id'] = conversationId

        const response = await callChatLunaChain(
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

        const aiMessage = new AIMessage(responseString)

        response.message = aiMessage

        return response
    }

    get model() {
        return this.chain.llm
    }
}
