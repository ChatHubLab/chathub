import { AIMessage, BaseMessage } from '@langchain/core/messages'
import { StructuredTool } from '@langchain/core/tools'
import { ChainValues } from '@langchain/core/utils/types'
import { Session } from 'koishi'
import {
    ChatLunaLLMCallArg,
    ChatLunaLLMChainWrapper,
    SystemPrompts
} from 'koishi-plugin-chatluna/llm-core/chain/base'
import {
    ChatHubBaseEmbeddings,
    ChatLunaChatModel
} from 'koishi-plugin-chatluna/llm-core/platform/model'
import { ChatLunaTool } from 'koishi-plugin-chatluna/llm-core/platform/types'
import {
    AgentExecutor,
    createOpenAIAgent,
    createReactAgent
} from 'koishi-plugin-chatluna/llm-core/agent'
import { BufferMemory } from 'koishi-plugin-chatluna/llm-core/memory/langchain'
import { logger } from '../..'
import {
    ChatLunaError,
    ChatLunaErrorCode
} from 'koishi-plugin-chatluna/utils/error'
import { PresetTemplate } from 'koishi-plugin-chatluna/llm-core/prompt'
import { ChatLunaChatPrompt } from 'koishi-plugin-chatluna/llm-core/chain/prompt'

export interface ChatLunaPluginChainInput {
    prompt: ChatLunaChatPrompt
    historyMemory: BufferMemory
    embeddings: ChatHubBaseEmbeddings
    agentMode?: 'tool-calling' | 'react'
    preset: () => Promise<PresetTemplate>
}

export class ChatLunaPluginChain
    extends ChatLunaLLMChainWrapper
    implements ChatLunaPluginChainInput
{
    executor: AgentExecutor

    historyMemory: BufferMemory

    systemPrompts?: SystemPrompts

    llm: ChatLunaChatModel

    embeddings: ChatHubBaseEmbeddings

    activeTools: ChatLunaTool[] = []

    tools: ChatLunaTool[]

    baseMessages: BaseMessage[] = undefined

    prompt: ChatLunaChatPrompt

    preset: () => Promise<PresetTemplate>

    agentMode?: 'tool-calling' | 'react'

    constructor({
        historyMemory,
        prompt,
        llm,
        tools,
        preset,
        embeddings,
        agentMode
    }: ChatLunaPluginChainInput & {
        tools: ChatLunaTool[]
        llm: ChatLunaChatModel
    }) {
        super()

        this.historyMemory = historyMemory
        this.prompt = prompt
        this.tools = tools
        this.embeddings = embeddings
        this.llm = llm
        this.agentMode = agentMode ?? 'react'

        this.preset = preset
    }

    static async fromLLMAndTools(
        llm: ChatLunaChatModel,
        tools: ChatLunaTool[],
        {
            historyMemory,
            preset,
            embeddings,
            agentMode
        }: Omit<ChatLunaPluginChainInput, 'prompt'>
    ): Promise<ChatLunaPluginChain> {
        const prompt = new ChatLunaChatPrompt({
            preset,
            tokenCounter: (text) => llm.getNumTokens(text),
            sendTokenLimit:
                llm.invocationParams().maxTokenLimit ??
                llm.getModelMaxContextSize()
        })

        return new ChatLunaPluginChain({
            historyMemory,
            prompt,
            llm,
            agentMode,
            embeddings,
            tools,
            preset
        })
    }

    private async _createExecutor(
        llm: ChatLunaChatModel,
        tools: StructuredTool[]
    ) {
        if (this.agentMode === 'react') {
            return AgentExecutor.fromAgentAndTools({
                tags: ['react'],
                agent: await createReactAgent({
                    llm,
                    tools,
                    prompt: this.prompt
                }),
                tools,
                memory: undefined,
                verbose: false
            })
        }

        return AgentExecutor.fromAgentAndTools({
            tags: ['tool-calling'],
            agent: createOpenAIAgent({
                llm,
                tools,
                prompt: this.prompt
            }),
            tools,
            memory: undefined,
            verbose: false
        })
    }

    private _getActiveTools(
        session: Session,
        messages: BaseMessage[]
    ): [ChatLunaTool[], boolean] {
        const tools: ChatLunaTool[] = this.activeTools

        const newActiveTools: [ChatLunaTool, boolean][] = this.tools.map(
            (tool) => {
                const base = tool.selector(messages)

                if (tool.authorization) {
                    return [tool, tool.authorization(session) && base]
                }

                return [tool, base]
            }
        )

        const differenceTools = newActiveTools.filter((tool) => {
            const include = tools.includes(tool[0])

            return !include || (include && tool[1] === false)
        })

        if (differenceTools.length > 0) {
            for (const differenceTool of differenceTools) {
                if (differenceTool[1] === false) {
                    const index = tools.findIndex(
                        (tool) => tool === differenceTool[0]
                    )
                    if (index > -1) {
                        tools.splice(index, 1)
                    }
                } else {
                    tools.push(differenceTool[0])
                }
            }
            return [this.activeTools, true]
        }

        return [
            this.tools,
            this.tools.some((tool) => tool?.alwaysRecreate === true)
        ]
    }

    async call({
        message,
        signal,
        session,
        events,
        conversationId,
        variables
    }: ChatLunaLLMCallArg): Promise<ChainValues> {
        const requests: ChainValues & {
            chat_history?: BaseMessage[]
            id?: string
        } = {
            input: message
        }

        this.baseMessages = await this.historyMemory.chatHistory.getMessages()

        requests['chat_history'] = this.baseMessages

        requests['id'] = conversationId
        requests['variables'] = variables ?? {}

        const [activeTools, recreate] = this._getActiveTools(
            session,
            this.baseMessages.concat(message)
        )

        if (recreate || this.executor == null) {
            const preset = await this.preset()
            const tools = activeTools.map((tool) =>
                tool.createTool(
                    {
                        model: this.llm,
                        embeddings: this.embeddings,
                        conversationId,
                        preset: preset.triggerKeyword[0],
                        userId: session.userId
                    },
                    session
                )
            )

            this.executor = await this._createExecutor(
                this.llm,
                await Promise.all(tools)
            )

            this.baseMessages =
                await this.historyMemory.chatHistory.getMessages()

            requests['chat_history'] = this.baseMessages
        }

        let usedToken = 0

        let response: ChainValues

        const request = () => {
            return this.executor.invoke(
                {
                    ...requests
                },
                {
                    signal,
                    callbacks: [
                        {
                            handleLLMEnd(output) {
                                usedToken +=
                                    output.llmOutput?.tokenUsage?.totalTokens ??
                                    0
                            },

                            handleAgentAction(action) {
                                events?.['llm-call-tool'](
                                    action.tool,
                                    action.toolInput,
                                    action.log
                                )
                            },

                            handleLLMNewToken(token) {
                                events?.['llm-new-token'](token)
                            }
                        }
                    ]
                }
            )
        }

        for (let i = 0; i < 3; i++) {
            if (signal.aborted) {
                throw new ChatLunaError(ChatLunaErrorCode.ABORTED)
            }
            try {
                response = await request()
                break
            } catch (e) {
                if (e.message.includes('Aborted')) {
                    throw new ChatLunaError(ChatLunaErrorCode.ABORTED)
                }
                logger.error(e)
            }
        }

        await events?.['llm-used-token-count']?.(usedToken)

        const responseString = response.output

        response.message = new AIMessage(responseString)

        return response
    }

    get model() {
        return this.llm
    }
}
