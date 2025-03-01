import {
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    MessageType,
    ToolMessage
} from '@langchain/core/messages'
import {
    ChatCompletionResponseMessageRoleEnum,
    ClaudeDeltaResponse,
    ClaudeMessage,
    CluadeTool
} from './types'
import { StructuredTool } from '@langchain/core/tools'
import { zodToJsonSchema } from 'zod-to-json-schema'

export function langchainMessageToClaudeMessage(
    messages: BaseMessage[],
    model?: string
): ClaudeMessage[] {
    const result: ClaudeMessage[] = []

    const mappedMessages = messages.map((rawMessage) => {
        const images = rawMessage.additional_kwargs.images as string[] | null

        const result: ClaudeMessage = {
            role: messageTypeToClaudeRole(rawMessage.getType()),
            content: rawMessage.content as string
        }

        if (model.includes('claude-3') && images != null) {
            result.content = []
            for (const image of images) {
                result.content.push({
                    type: 'image',
                    source: {
                        type: 'base64',
                        media_type: 'image/jpeg',
                        // remove base64 header
                        data: image.replace(/^data:image\/\w+;base64,/, '')
                    }
                })
            }
            result.content.push({
                type: 'text',
                text: rawMessage.content as string
            })
        }

        if (
            (rawMessage instanceof AIMessageChunk ||
                rawMessage instanceof AIMessage) &&
            (rawMessage.tool_calls?.length ?? 0) > 0
        ) {
            result.content = []

            const thinkContent = rawMessage.content as string

            if ((thinkContent?.length ?? 0) > 0) {
                result.content.push({
                    type: 'text',
                    text: thinkContent
                })
            }

            const mapToolCalls = rawMessage.tool_calls.map((toolCall) => ({
                type: 'tool_use' as const,
                id: toolCall.id,
                name: toolCall.name,
                input: toolCall.args
            }))

            result.content.push(...mapToolCalls)
        } else if (rawMessage instanceof ToolMessage) {
            result.content = []

            result.content.push({
                type: 'tool_result',
                content: rawMessage.content as string,
                tool_use_id: rawMessage.tool_call_id
            })
        }

        return result
    })

    for (let i = 0; i < mappedMessages.length; i++) {
        const message = mappedMessages[i]

        if (message.role !== 'system') {
            result.push(message)
            continue
        }

        /*   if (removeSystemMessage) {
            continue
        } */

        result.push({
            role: 'user',
            content: message.content
        })

        if (mappedMessages?.[i + 1]?.role === 'assistant') {
            continue
        }

        if (mappedMessages?.[i + 1]?.role === 'user') {
            result.push({
                role: 'assistant',
                content: 'Okay, what do I need to do?'
            })
        }
    }

    if (result[result.length - 1].role === 'assistant') {
        result.push({
            role: 'user',
            content:
                'Continue what I said to you last message. Follow these instructions.'
        })
    }

    return result
}

export function messageTypeToClaudeRole(
    type: MessageType
): ChatCompletionResponseMessageRoleEnum {
    switch (type) {
        case 'system':
            return 'system'
        case 'ai':
            return 'assistant'
        case 'human':
        case 'function':
        case 'tool':
            return 'user'
        default:
            throw new Error(`Unknown message type: ${type}`)
    }
}

export function formatToolsToClaudeTools(
    tools: StructuredTool[]
): CluadeTool[] {
    if (tools.length < 1) {
        return undefined
    }
    return tools.map(formatToolToClaudeTool)
}

export function formatToolToClaudeTool(tool: StructuredTool): CluadeTool {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/naming-convention
    const input_schema = zodToJsonSchema(tool.schema as any) as any

    delete input_schema['$schema']
    delete input_schema['additionalProperties']

    return {
        name: tool.name,
        description: tool.description,
        // any?
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        input_schema
    }
}

export function convertDeltaToMessageChunk(delta: ClaudeDeltaResponse) {
    if (delta.type === 'message_start') {
        return new AIMessageChunk({
            content: '',
            id: delta.message.id
        })
    } else if (
        delta.type === 'content_block_start' &&
        delta.content_block.type === 'tool_use'
    ) {
        const toolCallContentBlock = delta.content_block
        return new AIMessageChunk({
            content: '',
            tool_call_chunks: [
                {
                    id: toolCallContentBlock.id,
                    index: delta.index,
                    name: toolCallContentBlock.name,
                    args: ''
                }
            ],
            additional_kwargs: {}
        })
    } else if (
        delta.type === 'content_block_delta' &&
        delta.delta.type === 'text_delta'
    ) {
        const content = delta.delta?.text
        if (content !== undefined) {
            return new AIMessageChunk({
                content
            })
        }
    } else if (
        delta.type === 'content_block_delta' &&
        delta.delta.type === 'input_json_delta'
    ) {
        return new AIMessageChunk({
            content: '',
            tool_call_chunks: [
                {
                    index: delta.index,
                    args: delta.delta.partial_json
                }
            ],
            additional_kwargs: {}
        })
    } else if (
        delta.type === 'content_block_start' &&
        delta.content_block.type === 'text'
    ) {
        const content = delta.content_block?.text
        if (content !== undefined) {
            return new AIMessageChunk({
                content,
                additional_kwargs: {}
            })
        }
    } else if (
        delta.type === 'content_block_delta' &&
        delta.delta.type === 'thinking_delta'
    ) {
        const thinkResult = delta.delta.thinking
        return new AIMessageChunk({
            content: '',
            additional_kwargs: {
                reasoning_content: thinkResult
            }
        })
    }
}
