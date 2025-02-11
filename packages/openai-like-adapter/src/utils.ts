import {
    AIMessageChunk,
    BaseMessage,
    ChatMessageChunk,
    FunctionMessageChunk,
    HumanMessageChunk,
    MessageType,
    SystemMessageChunk,
    ToolMessage,
    ToolMessageChunk
} from '@langchain/core/messages'
import { StructuredTool } from '@langchain/core/tools'
import { JsonSchema7Type, zodToJsonSchema } from 'zod-to-json-schema'
import {
    ChatCompletionResponseMessage,
    ChatCompletionResponseMessageRoleEnum,
    ChatCompletionTool
} from './types'

export function langchainMessageToOpenAIMessage(
    messages: BaseMessage[],
    model?: string
): ChatCompletionResponseMessage[] {
    const result: ChatCompletionResponseMessage[] = []

    for (const rawMessage of messages) {
        const role = messageTypeToOpenAIRole(rawMessage.getType())

        const msg = {
            content: (rawMessage.content as string) || null,
            name:
                role === 'assistant' || role === 'tool'
                    ? rawMessage.name
                    : undefined,
            role,
            //  function_call: rawMessage.additional_kwargs.function_call,
            tool_calls: rawMessage.additional_kwargs.tool_calls,
            tool_call_id: (rawMessage as ToolMessage).tool_call_id
        } as ChatCompletionResponseMessage

        if (msg.tool_calls == null) {
            delete msg.tool_calls
        }

        if (msg.tool_call_id == null) {
            delete msg.tool_call_id
        }

        if (msg.tool_calls) {
            for (const toolCall of msg.tool_calls) {
                const tool = toolCall.function

                if (!tool.arguments) {
                    continue
                }
                // Remove spaces, new line characters etc.
                tool.arguments = JSON.stringify(JSON.parse(tool.arguments))
            }
        }

        const images = rawMessage.additional_kwargs.images as string[] | null

        const lowerModel = model?.toLowerCase()
        if (
            (lowerModel?.includes('vision') ||
                lowerModel?.includes('gpt-4o') ||
                lowerModel?.includes('claude') ||
                lowerModel?.includes('gemini')) &&
            images != null
        ) {
            msg.content = [
                {
                    type: 'text',
                    text: rawMessage.content as string
                }
            ]

            for (const image of images) {
                msg.content.push({
                    type: 'image_url',
                    image_url: {
                        url: image,
                        detail: 'low'
                    }
                })
            }
        }

        result.push(msg)
    }

    return result
}

export function messageTypeToOpenAIRole(
    type: MessageType
): ChatCompletionResponseMessageRoleEnum {
    switch (type) {
        case 'system':
            return 'system'
        case 'ai':
            return 'assistant'
        case 'human':
            return 'user'
        case 'function':
            return 'function'
        case 'tool':
            return 'tool'
        default:
            throw new Error(`Unknown message type: ${type}`)
    }
}

export function formatToolsToOpenAITools(
    tools: StructuredTool[]
): ChatCompletionTool[] {
    if (tools.length < 1) {
        return undefined
    }
    return tools.map(formatToolToOpenAITool)
}

export function formatToolToOpenAITool(
    tool: StructuredTool
): ChatCompletionTool {
    const parameters = removeAdditionalProperties(
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        zodToJsonSchema(tool.schema as any)
    )

    return {
        type: 'function',
        function: {
            name: tool.name,
            description: tool.description,
            // any?
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            parameters
        }
    }
}

function removeAdditionalProperties(schema: JsonSchema7Type): JsonSchema7Type {
    const updatedSchema = { ...schema }
    if (Object.hasOwn(updatedSchema, 'additionalProperties')) {
        delete updatedSchema['additionalProperties']
    }

    if (Object.hasOwn(updatedSchema, '$schema')) {
        delete updatedSchema['$schema']
    }

    if (updatedSchema['properties']) {
        const keys = Object.keys(updatedSchema['properties'])
        removeProperties(updatedSchema['properties'], keys, 0)
    }
    return updatedSchema
}
function removeProperties(
    properties: JsonSchema7Type,
    keys: string[],
    index: number
): void {
    if (index >= keys.length) {
        return
    }
    const key = keys[index]
    // eslint-disable-next-line no-param-reassign
    properties[key] = removeAdditionalProperties(properties[key])
    removeProperties(properties, keys, index + 1)
}

export function convertDeltaToMessageChunk(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    delta: Record<string, any>,
    defaultRole?: ChatCompletionResponseMessageRoleEnum
) {
    const role = (
        (delta.role?.length ?? 0) > 0 ? delta.role : defaultRole
    ).toLowerCase()
    const content = delta.content ?? ''
    const reasoningContent = delta.reasoning_content ?? ''

    let additionalKwargs: {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/naming-convention
        function_call?: any
        // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/naming-convention
        tool_calls?: any
        reasoning_content?: string
    }
    if (delta.function_call) {
        additionalKwargs = {
            function_call: delta.function_call
        }
    } else if (delta.tool_calls) {
        additionalKwargs = {
            tool_calls: delta.tool_calls
        }
    } else {
        additionalKwargs = {}
    }

    if (reasoningContent.length > 0) {
        additionalKwargs.reasoning_content = reasoningContent
    }

    if (role === 'user') {
        return new HumanMessageChunk({ content })
    } else if (role === 'assistant') {
        const toolCallChunks = []
        if (Array.isArray(delta.tool_calls)) {
            for (const rawToolCall of delta.tool_calls) {
                toolCallChunks.push({
                    name: rawToolCall.function?.name,
                    args: rawToolCall.function?.arguments,
                    id: rawToolCall.id,
                    index: rawToolCall.index
                })
            }
        }
        return new AIMessageChunk({
            content,
            tool_call_chunks: toolCallChunks,
            additional_kwargs: additionalKwargs
        })
    } else if (role === 'system') {
        return new SystemMessageChunk({ content })
    } else if (role === 'function') {
        return new FunctionMessageChunk({
            content,
            additional_kwargs: additionalKwargs,
            name: delta.name
        })
    } else if (role === 'tool') {
        return new ToolMessageChunk({
            content,
            additional_kwargs: additionalKwargs,
            tool_call_id: delta.tool_call_id
        })
    } else {
        return new ChatMessageChunk({ content, role })
    }
}
