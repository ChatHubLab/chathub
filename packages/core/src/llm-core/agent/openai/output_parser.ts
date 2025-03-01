import {
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    isBaseMessage
} from '@langchain/core/messages'
import { ChatGeneration } from '@langchain/core/outputs'
import { AgentAction, AgentFinish, AgentStep } from '@langchain/core/agents'
import { OutputParserException } from '@langchain/core/output_parsers'
import {
    AgentActionOutputParser,
    AgentMultiActionOutputParser,
    ChatCompletionMessageFunctionCall,
    ChatCompletionMessageToolCall
} from '../types'

/**
 * Type that represents an agent action with an optional message log.
 */
export type FunctionsAgentAction = AgentAction & {
    messageLog?: BaseMessage[]
}

export class OpenAIFunctionsAgentOutputParser extends AgentActionOutputParser {
    // eslint-disable-next-line @typescript-eslint/naming-convention
    lc_namespace = ['@langchain/core/messages', 'agents', 'openai']

    // eslint-disable-next-line @typescript-eslint/naming-convention
    static lc_name() {
        return 'OpenAIFunctionsAgentOutputParser'
    }

    async parse(text: string): Promise<AgentAction | AgentFinish> {
        throw new Error(
            `OpenAIFunctionsAgentOutputParser can only parse messages.\nPassed input: ${text}`
        )
    }

    async parseResult(generations: ChatGeneration[]) {
        if (
            'message' in generations[0] &&
            isBaseMessage(generations[0].message)
        ) {
            return this.parseAIMessage(generations[0].message)
        }
        throw new Error(
            'parseResult on OpenAIFunctionsAgentOutputParser only works on ChatGeneration output'
        )
    }

    /**
     * Parses the output message into a FunctionsAgentAction or AgentFinish
     * object.
     * @param message The BaseMessage to parse.
     * @returns A FunctionsAgentAction or AgentFinish object.
     */
    parseAIMessage(message: BaseMessage): FunctionsAgentAction | AgentFinish {
        if (message.content && typeof message.content !== 'string') {
            throw new Error(
                'This agent cannot parse non-string model responses.'
            )
        }
        if (message.additional_kwargs.function_call) {
            // eslint-disable-next-line prefer-destructuring, @typescript-eslint/naming-convention
            const function_call: ChatCompletionMessageFunctionCall =
                message.additional_kwargs.function_call
            try {
                const toolInput = function_call.arguments
                    ? JSON.parse(function_call.arguments)
                    : {}
                return {
                    tool: function_call.name as string,
                    toolInput,
                    log:
                        message.content?.length > 0
                            ? (message.content as string)
                            : `Invoking "${function_call.name}" with ${
                                  function_call.arguments ?? '{}'
                              }`,
                    messageLog: [message]
                }
            } catch (error) {
                throw new OutputParserException(
                    `Failed to parse function arguments from chat model response. Text: "${function_call.arguments}". ${error}`
                )
            }
        } else {
            return {
                returnValues: { output: message.content, message },
                log: message.content as string
            }
        }
    }

    getFormatInstructions(): string {
        throw new Error(
            'getFormatInstructions not implemented inside OpenAIFunctionsAgentOutputParser.'
        )
    }
}

/**
 * Type that represents an agent action with an optional message log.
 */
export type ToolsAgentAction = AgentAction & {
    toolCallId: string
    messageLog?: BaseMessage[]
}

export type ToolsAgentStep = AgentStep & {
    action: ToolsAgentAction
}

export class OpenAIToolsAgentOutputParser extends AgentMultiActionOutputParser {
    // eslint-disable-next-line @typescript-eslint/naming-convention
    lc_namespace = ['@langchain/core/messages', 'agents', 'openai']

    // eslint-disable-next-line @typescript-eslint/naming-convention
    static lc_name() {
        return 'OpenAIToolsAgentOutputParser'
    }

    async parse(text: string): Promise<AgentAction[] | AgentFinish> {
        throw new Error(
            `OpenAIFunctionsAgentOutputParser can only parse messages.\nPassed input: ${text}`
        )
    }

    async parseResult(generations: ChatGeneration[]) {
        if (
            'message' in generations[0] &&
            isBaseMessage(generations[0].message)
        ) {
            return this.parseAIMessage(generations[0].message)
        }
        throw new Error(
            'parseResult on OpenAIFunctionsAgentOutputParser only works on ChatGeneration output'
        )
    }

    /**
     * Parses the output message into a ToolsAgentAction[] or AgentFinish
     * object.
     * @param message The BaseMessage to parse.
     * @returns A ToolsAgentAction[] or AgentFinish object.
     */
    parseAIMessage(message: BaseMessage): ToolsAgentAction[] | AgentFinish {
        if (message.content && typeof message.content !== 'string') {
            throw new Error(
                'This agent cannot parse non-string model responses.'
            )
        }

        if (
            (message instanceof AIMessageChunk ||
                message instanceof AIMessage) &&
            message.tool_calls != null
        ) {
            const toolCalls = message.tool_calls

            if (toolCalls.length < 1) {
                return {
                    returnValues: { output: message.content },
                    log: message.content as string
                }
            }

            try {
                return toolCalls.map((toolCall, i) => {
                    const toolInput = toolCall.args
                    const messageLog = i === 0 ? [message] : []
                    return {
                        tool: toolCall.name as string,
                        toolInput,
                        toolCallId: toolCall.id,
                        log:
                            message.content?.length > 0
                                ? (message.content as string)
                                : `Invoking "${toolCall.name}" with ${
                                      JSON.stringify(toolCall.args) ?? '{}'
                                  }`,
                        messageLog
                    }
                })
            } catch (error) {
                throw new OutputParserException(
                    `Failed to parse tool arguments from chat model response. Text: "${JSON.stringify(
                        toolCalls
                    )}". ${error}`
                )
            }
        } else if (message.additional_kwargs.tool_calls) {
            const toolCalls: ChatCompletionMessageToolCall[] = message
                .additional_kwargs.tool_calls as ChatCompletionMessageToolCall[]
            try {
                return toolCalls.map((toolCall, i) => {
                    const toolInput = toolCall.function.arguments
                        ? JSON.parse(toolCall.function.arguments)
                        : {}
                    const messageLog = i === 0 ? [message] : []
                    return {
                        tool: toolCall.function.name as string,
                        toolInput,
                        toolCallId: toolCall.id,
                        log:
                            message.content?.length > 0
                                ? (message.content as string)
                                : `Invoking "${toolCall.function.name}" with ${
                                      toolCall.function.arguments ?? '{}'
                                  }`,
                        messageLog
                    }
                })
            } catch (error) {
                throw new OutputParserException(
                    `Failed to parse tool arguments from chat model response. Text: "${JSON.stringify(
                        toolCalls
                    )}". ${error}`
                )
            }
        } else {
            return {
                returnValues: { output: message.content },
                log: message.content as string
            }
        }
    }

    getFormatInstructions(): string {
        throw new Error(
            'getFormatInstructions not implemented inside OpenAIToolsAgentOutputParser.'
        )
    }
}
