/* eslint-disable max-len */
import { Document } from '@langchain/core/documents'
import { Embeddings } from '@langchain/core/embeddings'
import {
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage
} from '@langchain/core/messages'
import { PromptTemplate } from '@langchain/core/prompts'
import { StructuredTool } from '@langchain/core/tools'
import { ChainValues } from '@langchain/core/utils/types'
import {
    callChatLunaChain,
    ChatLunaLLMCallArg,
    ChatLunaLLMChain,
    ChatLunaLLMChainWrapper
} from 'koishi-plugin-chatluna/llm-core/chain/base'
import { ChatLunaChatModel } from 'koishi-plugin-chatluna/llm-core/platform/model'
import {
    BufferMemory,
    VectorStoreRetrieverMemory
} from 'koishi-plugin-chatluna/llm-core/memory/langchain'
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters'
import { MemoryVectorStore } from 'koishi-plugin-chatluna/llm-core/vectorstores'
import { logger } from '..'
import { PresetTemplate } from 'koishi-plugin-chatluna/llm-core/prompt'
import { ChatLunaChatPrompt } from 'koishi-plugin-chatluna/llm-core/chain/prompt'
import { ChatLunaTool } from 'koishi-plugin-chatluna/llm-core/platform/types'
import { PuppeteerBrowserTool } from '../tools/puppeteerBrowserTool'
import { Session } from 'koishi'

// github.com/langchain-ai/weblangchain/blob/main/nextjs/app/api/chat/stream_log/route.ts#L81

export interface ChatLunaBrowsingChainInput {
    botName: string
    preset: () => Promise<PresetTemplate>
    embeddings: Embeddings

    historyMemory: BufferMemory
    enhancedSummary: boolean

    thoughtMessage: boolean

    summaryModel: ChatLunaChatModel

    searchPrompt: string
    newQuestionPrompt: string
    searchConfidenceThreshold: number
}

export class ChatLunaBrowsingChain
    extends ChatLunaLLMChainWrapper
    implements ChatLunaBrowsingChainInput
{
    botName: string

    embeddings: Embeddings

    searchMemory: VectorStoreRetrieverMemory

    chain: ChatLunaLLMChain

    historyMemory: BufferMemory

    preset: () => Promise<PresetTemplate>

    formatQuestionChain: ChatLunaLLMChain

    textSplitter: RecursiveCharacterTextSplitter

    tools: ChatLunaToolWrapper[]

    cacheUrls: string[]

    responsePrompt: PromptTemplate

    enhancedSummary: boolean

    summaryModel: ChatLunaChatModel

    thoughtMessage: boolean

    searchPrompt: string

    newQuestionPrompt: string

    searchConfidenceThreshold: number

    constructor({
        botName,
        embeddings,
        historyMemory,
        chain,
        tools,
        formatQuestionChain,
        enhancedSummary,
        thoughtMessage,
        searchPrompt,
        searchConfidenceThreshold
    }: ChatLunaBrowsingChainInput & {
        chain: ChatLunaLLMChain
        formatQuestionChain: ChatLunaLLMChain
        tools: ChatLunaToolWrapper[]
        searchPrompt: string
    }) {
        super()
        this.botName = botName

        this.embeddings = embeddings
        this.enhancedSummary = enhancedSummary

        this.textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1400,
            chunkOverlap: 200
        })

        // use memory
        this.searchMemory = new VectorStoreRetrieverMemory({
            vectorStoreRetriever: new MemoryVectorStore(embeddings).asRetriever(
                10
            ),
            memoryKey: 'long_history',
            inputKey: 'input',
            outputKey: 'result',
            returnDocs: true
        })
        this.formatQuestionChain = formatQuestionChain

        this.historyMemory = historyMemory
        this.thoughtMessage = thoughtMessage

        this.cacheUrls = []

        this.responsePrompt = PromptTemplate.fromTemplate(searchPrompt)
        this.chain = chain
        this.tools = tools
        this.searchConfidenceThreshold = searchConfidenceThreshold
    }

    static fromLLMAndTools(
        llm: ChatLunaChatModel,
        tools: ChatLunaToolWrapper[],
        {
            botName,
            embeddings,
            summaryModel,
            historyMemory,
            preset,
            thoughtMessage,
            searchPrompt,
            newQuestionPrompt,
            enhancedSummary,
            searchConfidenceThreshold
        }: ChatLunaBrowsingChainInput
    ): ChatLunaBrowsingChain {
        const prompt = new ChatLunaChatPrompt({
            preset,
            tokenCounter: (text) => llm.getNumTokens(text),
            sendTokenLimit:
                llm.invocationParams().maxTokenLimit ??
                llm.getModelMaxContextSize()
        })

        const chain = new ChatLunaLLMChain({ llm, prompt })
        const formatQuestionChain = new ChatLunaLLMChain({
            llm,
            prompt: PromptTemplate.fromTemplate(newQuestionPrompt)
        })

        return new ChatLunaBrowsingChain({
            botName,
            formatQuestionChain,
            embeddings,
            summaryModel,
            historyMemory,
            preset,
            thoughtMessage,
            searchPrompt,
            newQuestionPrompt,
            chain,
            tools,
            enhancedSummary,
            searchConfidenceThreshold
        })
    }

    private async _selectTool(name: string): Promise<StructuredTool> {
        const chatLunaTool = this.tools.find((tool) => tool.name === name)

        return chatLunaTool.tool.createTool({
            embeddings: this.embeddings,
            model: this.summaryModel
        })
    }

    async fetchUrlContent(url: string, task: string) {
        const webTool = await this._selectTool('web-browser').then(
            (tool) => tool as PuppeteerBrowserTool
        )

        // open first
        const text = await webTool.invoke({
            action: 'summarize',
            params: task,
            url
        })

        logger?.debug('fetch url content:', text)

        await this.putContentToMemory(text, url)
    }

    async putContentToMemory(content: string, url: string) {
        if (this.cacheUrls.includes(url)) {
            return
        }

        await this.searchMemory.vectorStoreRetriever.vectorStore.addDocuments(
            await this.textSplitter.splitText(content).then((texts) =>
                texts.map(
                    (text) =>
                        new Document({
                            pageContent: text,
                            metadata: {
                                source: url
                            }
                        })
                )
            )
        )

        this.cacheUrls.push(url)
    }

    async call({
        message,
        stream,
        events,
        conversationId,
        session,
        variables
    }: ChatLunaLLMCallArg): Promise<ChainValues> {
        const requests: ChainValues = {
            input: message
        }

        const chatHistory = (
            await this.historyMemory.loadMemoryVariables(requests)
        )[this.historyMemory.memoryKey] as BaseMessage[]

        requests['chat_history'] = chatHistory
        requests['id'] = conversationId
        requests['variables'] = variables ?? {}

        // recreate questions

        let needSearch = true
        const newQuestion = (
            await callChatLunaChain(
                this.formatQuestionChain,
                {
                    chat_history: formatChatHistoryAsString(
                        chatHistory.slice(-6)
                    ),
                    question: message.content
                },
                {
                    'llm-used-token-count': events['llm-used-token-count']
                }
            )
        )['text'] as string

        if (newQuestion.includes('[skip]')) {
            needSearch = false
        }

        logger?.debug(
            `need search: ${needSearch}, new question: ${newQuestion}`
        )

        // search questions

        if (needSearch) {
            await this._search(newQuestion, message, chatHistory, session)
        }

        // format and call

        const { text: finalResponse } = await callChatLunaChain(
            this.chain,
            {
                ...requests,
                stream
            },
            events
        )

        logger?.debug(`final response %c`, finalResponse)

        // remove to reduce context length
        /* if (responsePrompt.length > 0) {
            await this.historyMemory.chatHistory.addMessage(new SystemMessage(responsePrompt))
            await this.historyMemory.chatHistory.addAIChatMessage(
                "OK. I understand. I will respond to the user's question using the same language as their input. What's the user's question?"
            )
        } */

        const aiMessage = new AIMessage(finalResponse)

        return {
            message: aiMessage
        }
    }

    private async _search(
        newQuestion: string,
        message: HumanMessage,
        chatHistory: BaseMessage[],
        session: Session
    ) {
        const searchTool = await this._selectTool('web-search')

        // Use the rephrased question for search
        const rawSearchResults = await searchTool.invoke(newQuestion)

        const searchResults =
            (JSON.parse(rawSearchResults as string) as unknown as {
                title: string
                description: string
                url: string
            }[]) ?? []

        if (this.thoughtMessage) {
            await session.send(
                `Find ${searchResults.length} search results about ${newQuestion}.`
            )
        }

        // format questions

        const formattedSearchResults = searchResults.map((result) => {
            // sort like json style
            // title: xx, xx: xx like
            let resultString = ''

            for (const key in result) {
                resultString += `${key}: ${result[key]}, `
            }

            resultString = resultString.slice(0, -2)

            return resultString
        })

        const relatedContents: string[] = []

        let vectorSearchResults: Document[] = []

        if (this.enhancedSummary && searchResults.length > 0) {
            const fetchPromises = searchResults
                .filter((result) => result.url?.startsWith('http'))
                .map(async (result) => {
                    if (result.description.length > 500) {
                        // 不对大内容作二次解读
                        await this.putContentToMemory(
                            result.description,
                            result.url
                        )

                        return
                    }

                    if (this.thoughtMessage) {
                        await session.send(`Reading ${result.url}...`)
                    }

                    try {
                        return await this.fetchUrlContent(
                            result.url,
                            newQuestion
                        )
                    } catch (e) {
                        logger.warn(`Error fetching ${result.url}:`, e)
                    }
                })

            // 切片并发
            const chunkedFetchPromises = chunkArray(fetchPromises, 5)

            for (const chunk of chunkedFetchPromises) {
                await Promise.all(chunk)
            }

            vectorSearchResults =
                await this.searchMemory.vectorStoreRetriever.invoke(newQuestion)

            for (const result of vectorSearchResults) {
                relatedContents.push(
                    `content: ${result.pageContent}, source: ${result.metadata.source}`
                )
            }
        }

        let responsePrompt = ''
        if (formattedSearchResults?.length > 0) {
            responsePrompt = await this.responsePrompt.format({
                question: message.content,
                context:
                    relatedContents.join('\n\n') +
                    '\n\n' +
                    formattedSearchResults.join('\n\n')
            })

            chatHistory.push(new SystemMessage(responsePrompt))

            chatHistory.push(
                new AIMessage(
                    "OK. I understand. I will respond to the your's question using the same language as your input. What's the your's question?"
                )
            )

            logger?.debug('formatted search results', searchResults)
        }

        return responsePrompt
    }

    get model() {
        return this.chain.llm
    }
}

const formatChatHistoryAsString = (history: BaseMessage[]) => {
    return history
        .map((message) => `${message.getType()}: ${message.content}`)
        .join('\n')
}

interface ChatLunaToolWrapper {
    name: string
    tool: ChatLunaTool
}

export function chunkArray<T>(array: T[], size: number): T[][] {
    return Array.from({ length: Math.ceil(array.length / size) }, (_, i) =>
        array.slice(i * size, i * size + size)
    )
}
