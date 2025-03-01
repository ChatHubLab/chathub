/* eslint-disable @typescript-eslint/naming-convention */
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
import {
    ChatCompletionFunction,
    WenxinMessage,
    WenxinMessageRole
} from './types'
import { StructuredTool } from '@langchain/core/tools'
import { zodToJsonSchema } from 'zod-to-json-schema'

export function langchainMessageToWenXinMessage(
    messages: BaseMessage[]
): WenxinMessage[] {
    const mappedMessage = messages.map((rawMessage) => {
        const role = messageTypeToWenXinRole(rawMessage.getType())

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
        }

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

        return msg
    })

    const result: WenxinMessage[] = []

    for (let i = 0; i < mappedMessage.length; i++) {
        const message = mappedMessage[i]

        if (i === 0 && message.role === 'assistant') {
            result.push({
                role: 'user',
                content:
                    'Continue what I said to you last time. Follow these instructions.'
            })
        }

        result.push({
            role: message.role,
            content: message.content as string,
            name: message.name,
            tool_calls: message.tool_calls,
            tool_call_id: message.tool_call_id
        })

        if (
            mappedMessage?.[i + 1]?.role === 'assistant' &&
            mappedMessage?.[i].role === 'assistant'
        ) {
            result.push({
                role: 'user',
                content:
                    'Continue what I said to you last time. Follow these instructions.'
            })
        }
    }

    if (result[result.length - 1].role === 'assistant') {
        result.push({
            role: 'user',
            content:
                'Continue what I said to you last time. Follow these instructions.'
        })
    }

    if (result[0].role === 'assistant') {
        result.unshift({
            role: 'user',
            content:
                'Continue what I said to you last time. Follow these instructions.'
        })
    }

    return result
}

export function messageTypeToWenXinRole(type: MessageType): WenxinMessageRole {
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

export function formatToolsToWenxinTools(
    tools: StructuredTool[]
): ChatCompletionFunction[] {
    if (tools.length < 1) {
        return undefined
    }
    return tools.map(formatToolToWenxinTool)
}

export function formatToolToWenxinTool(
    tool: StructuredTool
): ChatCompletionFunction {
    return {
        name: tool.name,
        description: tool.description,
        // any?
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        parameters: zodToJsonSchema(tool.schema as any)
    }
}

export function convertDeltaToMessageChunk(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    delta: Record<string, any>,
    defaultRole?: WenxinMessageRole
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
