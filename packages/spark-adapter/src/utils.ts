import { BaseMessage, MessageType } from '@langchain/core/messages'
import { StructuredTool } from '@langchain/core/tools'
import { zodToJsonSchema } from 'zod-to-json-schema'
import {
    ChatCompletionMessage,
    ChatCompletionMessageRoleEnum,
    ChatCompletionTool
} from './types'

export function langchainMessageToSparkMessage(
    messages: BaseMessage[],
    removeSystemMessage?: boolean
): ChatCompletionMessage[] {
    const mappedMessage = messages.map((it) => {
        const role = messageTypeSparkAIRole(it._getType())

        return {
            role,
            function_call: it.additional_kwargs.tool_calls?.[0]?.function,
            content: it.content as string,
            name: it.name
        } satisfies ChatCompletionMessage
    })

    const result: ChatCompletionMessage[] = []

    for (let i = 0; i < mappedMessage.length; i++) {
        const message = mappedMessage[i]

        if (message.role !== 'system') {
            result.push(message)
            continue
        }

        if (removeSystemMessage) {
            continue
        }

        result.push({
            role: 'user',
            content: message.content
        })

        result.push({
            role: 'assistant',
            content: 'Okay, what do I need to do?'
        })

        if (mappedMessage?.[i + 1]?.role === 'assistant') {
            result.push({
                role: 'user',
                content:
                    'Continue what I said to you last message. Follow these instructions.'
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

    if (result[0].role === 'assistant') {
        result.unshift({
            role: 'user',
            content:
                'Continue what I said to you last time. Follow these instructions.'
        })
    }

    return result
}

export function messageTypeSparkAIRole(
    type: MessageType
): ChatCompletionMessageRoleEnum {
    switch (type) {
        case 'system':
            return 'system'
        case 'ai':
            return 'assistant'
        case 'human':
            return 'user'
        case 'function':
            return 'user'
        case 'tool':
            return 'user'
        default:
            throw new Error(`Unknown message type: ${type}`)
    }
}

export function formatToolsToSparkTools(
    tools: StructuredTool[]
): ChatCompletionTool[] {
    if (tools.length < 1) {
        return undefined
    }
    return tools.map(formatToolToSparkTool)
}

export function formatToolToSparkTool(
    tool: StructuredTool
): ChatCompletionTool {
    return {
        name: tool.name,
        description: tool.description,
        // any?
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        parameters: zodToJsonSchema(tool.schema as any)
    }
}

export const modelMapping = {
    'spark-lite': {
        wsUrl: 'v1.1/chat',
        model: 'general'
    },
    'spark-pro': {
        wsUrl: 'v3.1/chat',
        model: 'generalv3'
    },
    'spark-pro-128k': {
        wsUrl: 'chat/pro-128k',
        model: 'pro-128k'
    },
    'spark-max': {
        wsUrl: 'v3.5/chat',
        model: 'generalv3.5'
    },
    'spark-max-32k': {
        wsUrl: 'chat/max-32k',
        model: 'max-32k'
    },
    'spark-4.0-ultra': {
        wsUrl: 'v4.0/chat',
        model: '4.0Ultra'
    }
}
