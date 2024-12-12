/* eslint-disable @typescript-eslint/no-explicit-any */
import {
    AIMessageChunk,
    BaseMessage,
    ChatMessageChunk,
    HumanMessageChunk,
    MessageType,
    SystemMessageChunk
} from '@langchain/core/messages'
import { StructuredTool } from '@langchain/core/tools'
import { zodToJsonSchema } from 'zod-to-json-schema'
import {
    ChatCompletionFunction,
    ChatCompletionResponseMessage,
    ChatCompletionResponseMessageRoleEnum,
    ChatMessagePart,
    ChatPart,
    ChatUploadDataPart
} from './types'
import { Config, logger } from '.'

export async function langchainMessageToGeminiMessage(
    messages: BaseMessage[],
    model?: string
): Promise<ChatCompletionResponseMessage[]> {
    const mappedMessage = await Promise.all(
        messages.map(async (rawMessage) => {
            const role = messageTypeToGeminiRole(rawMessage._getType())

            if (
                role === 'function' ||
                rawMessage.additional_kwargs?.function_call != null
            ) {
                return {
                    role: 'function',
                    parts: [
                        {
                            functionResponse:
                                rawMessage.additional_kwargs?.function_call !=
                                null
                                    ? undefined
                                    : {
                                          name: rawMessage.name,
                                          response: {
                                              name: rawMessage.name,
                                              content: (() => {
                                                  try {
                                                      const result = JSON.parse(
                                                          rawMessage.content as string
                                                      )

                                                      if (
                                                          typeof result ===
                                                          'string'
                                                      ) {
                                                          return {
                                                              response: result
                                                          }
                                                      } else {
                                                          return result
                                                      }
                                                  } catch (e) {
                                                      return {
                                                          response:
                                                              rawMessage.content
                                                      }
                                                  }
                                              })()
                                          }
                                      },
                            functionCall:
                                rawMessage.additional_kwargs?.function_call !=
                                null
                                    ? {
                                          name: rawMessage.additional_kwargs
                                              .function_call.name,
                                          args: (() => {
                                              try {
                                                  const result = JSON.parse(
                                                      rawMessage
                                                          .additional_kwargs
                                                          .function_call
                                                          .arguments
                                                  )

                                                  if (
                                                      typeof result === 'string'
                                                  ) {
                                                      return {
                                                          input: result
                                                      }
                                                  } else {
                                                      return result
                                                  }
                                              } catch (e) {
                                                  return {
                                                      input: rawMessage
                                                          .additional_kwargs
                                                          .function_call
                                                          .arguments
                                                  }
                                              }
                                          })()
                                      }
                                    : undefined
                        }
                    ]
                }
            }

            const images = rawMessage.additional_kwargs.images as
                | string[]
                | null

            const result: ChatCompletionResponseMessage = {
                role,
                parts: [
                    {
                        text: rawMessage.content as string
                    }
                ]
            }

            if (
                (model.includes('vision') || model.includes('gemini')) &&
                images != null &&
                !model.includes('gemini-1.0')
            ) {
                for (const image of images) {
                    result.parts.push({
                        inline_data: {
                            // base64 image match type
                            data: image.replace(/^data:image\/\w+;base64,/, ''),
                            mime_type: 'image/jpeg'
                        }
                    })
                }
            }

            return result
        })
    )

    const result: ChatCompletionResponseMessage[] = []

    for (let i = 0; i < mappedMessage.length; i++) {
        const message = mappedMessage[i]

        if (message.role !== 'system') {
            result.push(message)
            continue
        }

        /*   if (removeSystemMessage) {
            continue
        } */

        result.push({
            role: 'user',
            parts: message.parts
        })

        const nextMessage = mappedMessage?.[i + 1]

        if (nextMessage?.role === 'model') {
            continue
        }

        if (nextMessage?.role === 'user' || nextMessage?.role === 'system') {
            result.push({
                role: 'model',
                parts: [{ text: 'Okay, what do I need to do?' }]
            })
        }

        if (nextMessage?.role === 'system') {
            result.push({
                role: 'user',
                parts: [
                    {
                        text: 'Continue what I said to you last message. Follow these instructions.'
                    }
                ]
            })
        }
    }

    if (result[result.length - 1].role === 'model') {
        result.push({
            role: 'user',
            parts: [
                {
                    text: 'Continue what I said to you last message. Follow these instructions.'
                }
            ]
        })
    }

    if (model.includes('vision')) {
        // format prompts

        const textBuffer: string[] = []

        const last = result.pop()

        for (let i = 0; i < result.length; i++) {
            const message = result[i]
            const text = (message.parts[0] as ChatMessagePart).text

            textBuffer.push(`${message.role}: ${text}`)
        }

        const lastParts = last.parts

        let lastImagesParts = lastParts.filter(
            (part) =>
                (part as ChatUploadDataPart).inline_data?.mime_type ===
                'image/jpeg'
        ) as ChatUploadDataPart[]

        if (lastImagesParts.length < 1) {
            for (let i = result.length - 1; i >= 0; i--) {
                const message = result[i]
                const images = message.parts.filter(
                    (part) =>
                        (part as ChatUploadDataPart).inline_data?.mime_type ===
                        'image/jpeg'
                ) as ChatUploadDataPart[]

                if (images.length > 0) {
                    lastImagesParts = images
                    break
                }
            }
        }

        ;(
            lastParts.filter(
                (part) =>
                    (part as ChatMessagePart).text !== undefined &&
                    (part as ChatMessagePart).text !== null
            ) as ChatMessagePart[]
        ).forEach((part) => {
            textBuffer.push(`${last.role}: ${part.text}`)
        })

        return [
            {
                role: 'user',
                parts: [
                    {
                        text: textBuffer.join('\n')
                    },
                    ...lastImagesParts
                ]
            }
        ]
    }

    return result
}

export function partAsType<T extends ChatPart>(part: ChatPart): T {
    return part as T
}

export function formatToolsToGeminiAITools(
    tools: StructuredTool[],
    config: Config,
    model: string
): Record<string, any> {
    if (tools.length < 1 && !config.googleSearch) {
        return undefined
    }
    const functions = tools.map(formatToolToGeminiAITool)

    const result = []

    if (functions.length > 0 && !config.googleSearch) {
        result.push({
            functionDeclarations: functions
        })
    } else if (functions.length > 0 && config.googleSearch) {
        logger.warn('Google search is enabled, function call will be disabled.')
    }

    if (config.googleSearch) {
        if (model.includes('gemini-2')) {
            result.push({
                google_search: {}
            })
        } else {
            result.push({
                google_search_retrieval: {
                    dynamic_retrieval_config: {
                        mode: 'MODE_DYNAMIC',
                        dynamic_threshold: config.searchThreshold
                    }
                }
            })
        }
    }

    return result
}

export function formatToolToGeminiAITool(
    tool: StructuredTool
): ChatCompletionFunction {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const parameters = zodToJsonSchema(tool.schema as any)

    // remove unsupported properties
    delete parameters['$schema']
    delete parameters['additionalProperties']
    return {
        name: tool.name,
        description: tool.description,
        // any?
        parameters
    }
}

export function messageTypeToGeminiRole(
    type: MessageType
): ChatCompletionResponseMessageRoleEnum {
    switch (type) {
        case 'system':
            return 'system'
        case 'ai':
            return 'model'
        case 'human':
            return 'user'
        case 'function':
            return 'function'
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
        return new AIMessageChunk({ content, additional_kwargs })
    } else if (role === 'system') {
        return new SystemMessageChunk({ content })
    } else {
        return new ChatMessageChunk({ content, role })
    }
}
