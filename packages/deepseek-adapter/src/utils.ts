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
import { zodToJsonSchema } from 'zod-to-json-schema'
import {
    ChatCompletionResponseMessage,
    ChatCompletionResponseMessageRoleEnum,
    ChatCompletionTool
} from './types'

export function langchainMessageToDeepseekMessage(
    messages: BaseMessage[],
    model?: string
): ChatCompletionResponseMessage[] {
    const mappedMessage: ChatCompletionResponseMessage[] = []

    for (const rawMessage of messages) {
        const role = messageTypeToDeepseekRole(rawMessage.getType())

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

        if (
            (model?.includes('vision') || model?.includes('chat')) &&
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

        mappedMessage.push(msg)
    }

    if (!model.includes('reasoner')) {
        return mappedMessage
    }

    if (!model.includes('reasoner')) {
        return mappedMessage
    }

    if (!model.includes('reasoner')) {
        return mappedMessage
    }

    const result: ChatCompletionResponseMessage[] = []

    for (let i = 0; i < mappedMessage.length; i++) {
        const message = mappedMessage[i]

        // 检查 result 的最后一个消息的角色
        if (result.length > 0) {
            const lastMessage = result[result.length - 1]

            // 如果最后一个消息是 user，且当前消息也是 user，插入一个 assistant 消息
            if (lastMessage.role === 'user' && message.role === 'user') {
                result.push({
                    role: 'assistant',
                    content: 'Okay, what do I need to do?'
                })
            }
            // 如果最后一个消息是 assistant，且当前消息也是 assistant，插入一个 user 消息
            else if (
                lastMessage.role === 'assistant' &&
                message.role === 'assistant'
            ) {
                result.push({
                    role: 'user',
                    content: 'Ok'
                })
            }
        }

        // 处理 system 消息
        if (message.role === 'system') {
            result.push({
                role: 'user',
                content: message.content
            })

            result.push({
                role: 'assistant',
                content: 'Okay, what do I need to do?'
            })

            result.push({
                role: 'user',
                content:
                    'Continue what I said to you last message. Follow these instructions.'
            })
        } else {
            // 直接添加非 system 消息
            result.push(message)
        }
    }

    const lastMessage = result[result.length - 1]

    if (lastMessage.role === 'assistant') {
        result.push({
            role: 'user',
            content: 'Continue output'
        })
    }

    return result
}

export function messageTypeToDeepseekRole(
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
    return {
        type: 'function',
        function: {
            name: tool.name,
            description: tool.description,
            // any?
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            parameters: zodToJsonSchema(tool.schema as any)
        }
    }
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
    // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/naming-convention
    let additional_kwargs: {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        function_call?: any
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        tool_calls?: any
        reasoning_content?: string
    }
    if (delta.function_call) {
        additional_kwargs = {
            function_call: delta.function_call
        }
    } else if (delta.tool_calls) {
        additional_kwargs = {
            tool_calls: delta.tool_calls
        }
    } else if (reasoningContent.length > 0) {
        additional_kwargs = {
            reasoning_content: reasoningContent
        }
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
            additional_kwargs
        })
    } else if (role === 'system') {
        return new SystemMessageChunk({ content })
    } else if (role === 'function') {
        return new FunctionMessageChunk({
            content,
            additional_kwargs,
            name: delta.name
        })
    } else if (role === 'tool') {
        return new ToolMessageChunk({
            content,
            additional_kwargs,
            tool_call_id: delta.tool_call_id
        })
    } else {
        return new ChatMessageChunk({ content, role })
    }
}
