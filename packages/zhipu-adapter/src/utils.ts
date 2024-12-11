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
    ChatCompletionResponseMessage,
    ChatCompletionResponseMessageRoleEnum,
    ChatCompletionTool,
    ZhipuClientConfig
} from './types'
import { StructuredTool } from '@langchain/core/tools'
import { zodToJsonSchema } from 'zod-to-json-schema'

export function langchainMessageToZhipuMessage(
    messages: BaseMessage[],
    model?: string
): ChatCompletionResponseMessage[] {
    const result: ChatCompletionResponseMessage[] = []
    const mappedMessage: ChatCompletionResponseMessage[] = []

    for (const rawMessage of messages) {
        const role = messageTypeToZhipuRole(rawMessage._getType())

        const msg = {
            content: (rawMessage.content as string) || null,
            name: role === 'assistant' ? rawMessage.name : undefined,
            role,
            tool_call_id: (rawMessage as ToolMessage).tool_call_id,
            tool_calls: model.includes('4v')
                ? undefined
                : rawMessage.additional_kwargs.tool_calls
        } as ChatCompletionResponseMessage

        if (msg.tool_calls) {
            for (const toolCall of msg.tool_calls) {
                const tool = toolCall.function

                if (!tool?.arguments) {
                    continue
                }
                // Remove spaces, new line characters etc.
                tool.arguments = JSON.stringify(JSON.parse(tool.arguments))
            }
        }

        const images = rawMessage.additional_kwargs.images as string[] | null

        if (model.includes('4v') && images != null) {
            msg.content = [
                {
                    type: 'text',
                    text: rawMessage.content as string
                }
            ]

            // base 64??
            for (const image of images) {
                msg.content.push({
                    type: 'image_url',
                    image_url: {
                        url: image
                        // detail: 'low'
                    }
                })
            }
        } else if (model.includes('tools')) {
            msg.content = [
                {
                    type: 'text',
                    text: rawMessage.content as string
                }
            ]
        }

        mappedMessage.push(msg)
    }

    if (model === 'glm-4v-flash') {
        // The 4v-flash only supports one image
        let lastImageMessageIndex = mappedMessage
            .slice()
            .reverse()
            .findIndex((message) => Array.isArray(message.content))

        if (lastImageMessageIndex !== -1) {
            lastImageMessageIndex =
                mappedMessage.length - 1 - lastImageMessageIndex

            for (let index = lastImageMessageIndex - 1; index >= 0; index--) {
                const message = mappedMessage[index]
                const content = message.content

                if (!Array.isArray(content)) {
                    continue
                }

                message.content = content.find(
                    (value) => value.type === 'text'
                ).text
            }
        }
    }

    for (let i = 0; i < mappedMessage.length; i++) {
        const message = mappedMessage[i]

        if (message.role !== 'system') {
            result.push(message)
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

    return result
}

export function messageTypeToZhipuRole(
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

export function convertDeltaToMessageChunk(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    delta: Record<string, any>,
    defaultRole?: ChatCompletionResponseMessageRoleEnum
) {
    const role = delta.role ?? defaultRole
    const content = delta.content ?? ''
    // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/naming-convention
    let additional_kwargs: { function_call?: any; tool_calls?: any }
    if (delta.function_call) {
        additional_kwargs = {
            function_call: delta.function_call
        }
    } else if (delta.tool_calls) {
        additional_kwargs = {
            tool_calls: delta.tool_calls
        }
    } else {
        additional_kwargs = {}
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

export function formatToolsToZhipuTools(
    model: string,
    tools: StructuredTool[],
    clientConfig: ZhipuClientConfig
): ChatCompletionTool[] {
    let result: ChatCompletionTool[] = []

    if (clientConfig.webSearch) {
        result.push({
            type: 'web_search',
            web_search: {
                enable: true
            }
        })
    }

    if (clientConfig.retrieval?.length > 0) {
        const mappedTools = clientConfig.retrieval.map((item) => {
            return {
                type: 'retrieval',
                retrieval: {
                    knowledge_id: item,
                    prompt_template:
                        clientConfig.knowledgePromptTemplate?.length > 0
                            ? clientConfig.knowledgePromptTemplate
                            : undefined
                }
            } satisfies ChatCompletionTool
        })

        result.push(...mappedTools)
    }

    if (clientConfig.codeInterpreter) {
        result.push({
            type: 'code_interpreter'
        } satisfies ChatCompletionTool)
    }

    if (clientConfig.webSearch && model.includes('tools')) {
        result.push({
            type: 'web_browser',
            web_browser: {
                browser: 'auto'
            }
        } satisfies ChatCompletionTool)

        // remove web_search
        result = result.filter((tool) => tool.type !== 'web_search')
    }

    if (tools?.length > 0) {
        result.push(...tools.map(formatToolToZhipuTool))
    }

    if (result.length < 1) {
        return undefined
    }

    return result
}

export function formatToolToZhipuTool(
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
