/* eslint-disable max-len */
import { Document } from '@langchain/core/documents'
import { Embeddings } from '@langchain/core/embeddings'
import { AIMessage, BaseMessage, SystemMessage } from '@langchain/core/messages'
import {
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate
} from '@langchain/core/prompts'
import { StructuredTool, Tool } from '@langchain/core/tools'
import { ChainValues } from '@langchain/core/utils/types'
import {
    callChatHubChain,
    ChatHubLLMCallArg,
    ChatHubLLMChain,
    ChatHubLLMChainWrapper,
    SystemPrompts
} from 'koishi-plugin-chatluna/llm-core/chain/base'
import { ChatLunaSaveableVectorStore } from 'koishi-plugin-chatluna/llm-core/model/base'
import { ChatLunaChatModel } from 'koishi-plugin-chatluna/llm-core/platform/model'
import {
    BufferMemory,
    ConversationSummaryMemory,
    VectorStoreRetrieverMemory
} from 'langchain/memory'
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter'
import { MemoryVectorStore } from 'langchain/vectorstores/memory'
import { logger } from '..'
import { ChatHubBrowsingPrompt } from './prompt'

// github.com/langchain-ai/weblangchain/blob/main/nextjs/app/api/chat/stream_log/route.ts#L81

export interface ChatLunaBrowsingChainInput {
    botName: string
    systemPrompts?: SystemPrompts
    embeddings: Embeddings
    longMemory: VectorStoreRetrieverMemory
    historyMemory: ConversationSummaryMemory | BufferMemory
    enhancedSummary: boolean
}

export class ChatLunaBrowsingChain
    extends ChatHubLLMChainWrapper
    implements ChatLunaBrowsingChainInput
{
    botName: string

    embeddings: Embeddings

    searchMemory: VectorStoreRetrieverMemory

    chain: ChatHubLLMChain

    historyMemory: ConversationSummaryMemory | BufferMemory

    systemPrompts?: SystemPrompts

    longMemory: VectorStoreRetrieverMemory

    formatQuestionChain: ChatHubLLMChain

    textSplitter: RecursiveCharacterTextSplitter

    tools: StructuredTool[]

    responsePrompt: PromptTemplate

    enhancedSummary: boolean

    constructor({
        botName,
        embeddings,
        historyMemory,
        systemPrompts,
        chain,
        tools,
        longMemory,
        formatQuestionChain,
        enhancedSummary
    }: ChatLunaBrowsingChainInput & {
        chain: ChatHubLLMChain
        formatQuestionChain: ChatHubLLMChain
        tools: StructuredTool[]
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
                6
            ),
            memoryKey: 'long_history',
            inputKey: 'input',
            outputKey: 'result',
            returnDocs: true
        })
        this.formatQuestionChain = formatQuestionChain
        this.longMemory = longMemory
        this.historyMemory = historyMemory
        this.systemPrompts = systemPrompts
        this.responsePrompt = PromptTemplate.fromTemplate(RESPONSE_TEMPLATE)
        this.chain = chain
        this.tools = tools
    }

    static fromLLMAndTools(
        llm: ChatLunaChatModel,
        tools: Tool[],
        {
            botName,
            embeddings,
            historyMemory,
            systemPrompts,
            longMemory,
            enhancedSummary
        }: ChatLunaBrowsingChainInput
    ): ChatLunaBrowsingChain {
        const humanMessagePromptTemplate =
            HumanMessagePromptTemplate.fromTemplate('{input}')

        let conversationSummaryPrompt: HumanMessagePromptTemplate
        let messagesPlaceholder: MessagesPlaceholder

        if (historyMemory instanceof ConversationSummaryMemory) {
            conversationSummaryPrompt = HumanMessagePromptTemplate.fromTemplate(
                // eslint-disable-next-line max-len
                `This is some conversation between me and you. Please generate an response based on the system prompt and content below. Relevant pieces of previous conversation: {long_history} (You do not need to use these pieces of information if not relevant, and based on these information, generate similar but non-repetitive responses. Pay attention, you need to think more and diverge your creativity) Current conversation: {chat_history}`
            )
        } else {
            conversationSummaryPrompt = HumanMessagePromptTemplate.fromTemplate(
                // eslint-disable-next-line max-len
                `Relevant pieces of previous conversation: {long_history} (You do not need to use these pieces of information if not relevant, and based on these information, generate similar but non-repetitive responses. Pay attention, you need to think more and diverge your creativity.)`
            )

            messagesPlaceholder = new MessagesPlaceholder('chat_history')
        }
        const prompt = new ChatHubBrowsingPrompt({
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
                llm.invocationParams().maxTokens ?? llm.getModelMaxContextSize()
        })

        const chain = new ChatHubLLMChain({ llm, prompt })
        const formatQuestionChain = new ChatHubLLMChain({
            llm,
            prompt: PromptTemplate.fromTemplate(REPHRASE_TEMPLATE)
        })

        return new ChatLunaBrowsingChain({
            botName,
            formatQuestionChain,
            embeddings,
            historyMemory,
            systemPrompts,
            chain,
            tools,
            longMemory,
            enhancedSummary
        })
    }

    private _selectTool(name: string): StructuredTool {
        return this.tools.find((tool) => tool.name === name)
    }

    async fetchUrlContent(url: string) {
        const webTool = this._selectTool('web-browser')

        const text = (await webTool.invoke(
            JSON.stringify({
                url,
                raw_content: true
            })
        )) as unknown as string

        logger.debug(`url content for ${url} is ${text}`)

        await this.searchMemory.vectorStoreRetriever.vectorStore.addDocuments(
            await this.textSplitter.splitText(text).then((texts) =>
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
    }

    async call({
        message,
        stream,
        events,
        conversationId
    }: ChatHubLLMCallArg): Promise<ChainValues> {
        const requests: ChainValues = {
            input: message
        }

        const chatHistory = (
            await this.historyMemory.loadMemoryVariables(requests)
        )[this.historyMemory.memoryKey] as BaseMessage[]

        const longHistory = (
            await this.longMemory.loadMemoryVariables({
                user: message.content
            })
        )[this.longMemory.memoryKey]

        requests['long_history'] = longHistory
        requests['chat_history'] = chatHistory
        requests['id'] = conversationId

        // recreate questions

        const newQuestion = (
            await callChatHubChain(
                this.formatQuestionChain,
                {
                    chat_history: formatChatHistoryAsString(chatHistory),
                    question: message.content
                },
                {
                    'llm-used-token-count': events['llm-used-token-count']
                }
            )
        )['text'] as string

        logger?.debug(`new questions %c`, newQuestion)

        // search questions

        const searchTool = this._selectTool('search')

        const searchResults =
            (JSON.parse(
                (await searchTool.invoke(newQuestion)) as string
            ) as unknown as {
                title: string
                description: string
                url: string
            }[]) ?? []

        // format questions

        const formattedSearchResults = searchResults.map(
            (result) =>
                `title: ${result.title}\ndesc: ${result.description}` +
                (result.url ? `\nsource: ${result.url}` : '')
        )

        const relatedContents: string[] = []

        let vectorSearchResults =
            await this.searchMemory.vectorStoreRetriever.invoke(newQuestion)

        if (this.enhancedSummary && vectorSearchResults.length < 1) {
            for (const result of searchResults) {
                await this.fetchUrlContent(result.url)
            }
            vectorSearchResults =
                await this.searchMemory.vectorStoreRetriever.invoke(newQuestion)
        }

        for (const result of vectorSearchResults) {
            relatedContents.push(result.pageContent)
        }

        let responsePrompt = ''
        if (searchResults?.length > 0) {
            responsePrompt = await this.responsePrompt.format({
                question: message.content,
                context: formattedSearchResults.join('\n\n')
            })
            chatHistory.push(new SystemMessage(responsePrompt))

            if (relatedContents.length > 0) {
                chatHistory.push(
                    new SystemMessage(
                        "There's some related contents to helper you answer"
                    )
                )
                for (const content of relatedContents) {
                    chatHistory.push(new SystemMessage(content))
                }
            }

            chatHistory.push(new AIMessage("OK. What's your question?"))

            logger?.debug('formatted search results', searchResults)
        }
        // format and call

        requests['input'] = message.content

        const { text: finalResponse } = await callChatHubChain(
            this.chain,
            {
                ...requests,
                stream
            },
            events
        )

        logger?.debug(`final response %c`, finalResponse)
        if (responsePrompt.length > 0) {
            await this.historyMemory.chatHistory.addUserMessage(responsePrompt)
            await this.historyMemory.chatHistory.addAIChatMessage(
                "OK. What's your question?"
            )
        }

        await this.historyMemory.chatHistory.addMessage(message)
        await this.historyMemory.chatHistory.addAIChatMessage(finalResponse)

        await this.longMemory.saveContext(
            { user: message.content },
            { your: finalResponse }
        )

        const vectorStore = this.longMemory.vectorStoreRetriever.vectorStore

        if (vectorStore instanceof ChatLunaSaveableVectorStore) {
            logger?.debug('saving vector store')
            await vectorStore.save()
        }

        const aiMessage = new AIMessage(finalResponse)

        return {
            message: aiMessage
        }
    }

    get model() {
        return this.chain.llm
    }
}

const RESPONSE_TEMPLATE = `
GOAL: Now you need answering any question and output with question language.

Generate a comprehensive and informative, yet concise answer of 250 words or less for the given question based solely on the provided search results (URL and content).
You must only use information from the provided search results. Use an unbiased and journalistic tone. Combine search results together into a coherent answer.
Do not repeat text. Cite search results using [\${{number}}] notation. Only cite the most
relevant results that answer the question accurately. Place these citations at the end
of the sentence or paragraph that reference them - do not put them all at the end. If
different results refer to different entities within the same name, write separate
answers for each entity. If you want to cite multiple results for the same sentence,
format it as \`[\${{number1}}] [\${{number2}}]\`. However, you should NEVER do this with the
same number - if you want to cite \`number1\` multiple times for a sentence, only do
\`[\${{number1}}]\` not \`[\${{number1}}] [\${{number1}}]\`

Your text style should be the same as the system message set to.

You should use bullet points in your answer for readability. Put citations where they apply rather than putting them all at the end.

At the end, list the source of the referenced search results in markdown format.

If there is nothing in the context relevant to the question at hand, just say "Hmm,
I'm not sure." Don't try to make up an answer.


Anything between the following \`context\` html blocks is retrieved from a knowledge
bank, not part of the conversation with the user.

<context>
    {context}
<context/>

REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm not sure." Don't try to make up an answer. Anything between the preceding 'context' html blocks is retrieved from a knowledge bank, not part of the conversation with the user. The output need format with question's language. The output need format with question's language.`

// eslint-disable-next-line max-len
const REPHRASE_TEMPLATE = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question and use origin question language.

The standalone question should be search engine friendly. DO NOT ADD ANYTHING ELSE.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:`

const formatChatHistoryAsString = (history: BaseMessage[]) => {
    return history
        .map((message) => `${message._getType()}: ${message.content}`)
        .join('\n')
}
