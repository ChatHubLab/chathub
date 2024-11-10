import { AIMessageChunk } from '@langchain/core/messages'
import { ChatGenerationChunk } from '@langchain/core/outputs'
import { JSONParser } from '@streamparser/json'
import {
    EmbeddingsRequester,
    EmbeddingsRequestParams,
    ModelRequester,
    ModelRequestParams
} from 'koishi-plugin-chatluna/llm-core/platform/api'
import { ClientConfig } from 'koishi-plugin-chatluna/llm-core/platform/config'
import {
    ChatLunaError,
    ChatLunaErrorCode
} from 'koishi-plugin-chatluna/utils/error'
import { sse } from 'koishi-plugin-chatluna/utils/sse'
import { readableStreamToAsyncIterable } from 'koishi-plugin-chatluna/utils/stream'
import * as fetchType from 'undici/types/fetch'
import { logger } from '.'
import {
    ChatCompletionMessageFunctionCall,
    ChatFunctionCallingPart,
    ChatMessagePart,
    ChatPart,
    ChatResponse,
    CreateEmbeddingResponse
} from './types'
import {
    formatToolsToGeminiAITools,
    langchainMessageToGeminiMessage,
    partAsType
} from './utils'
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'

export class GeminiRequester
    extends ModelRequester
    implements EmbeddingsRequester
{
    constructor(
        private _config: ClientConfig,
        private _plugin: ChatLunaPlugin
    ) {
        super()
    }

    async *completionStream(
        params: ModelRequestParams
    ): AsyncGenerator<ChatGenerationChunk> {
        try {
            const response = await this._post(
                `models/${params.model}:streamGenerateContent`,
                {
                    contents: await langchainMessageToGeminiMessage(
                        params.input,
                        params.model
                    ),
                    safetySettings: [
                        {
                            category: 'HARM_CATEGORY_HARASSMENT',
                            threshold: 'BLOCK_NONE'
                        },
                        {
                            category: 'HARM_CATEGORY_HATE_SPEECH',
                            threshold: 'BLOCK_NONE'
                        },
                        {
                            category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
                            threshold: 'BLOCK_NONE'
                        },
                        {
                            category: 'HARM_CATEGORY_DANGEROUS_CONTENT',
                            threshold: 'BLOCK_NONE'
                        }
                    ],
                    generationConfig: {
                        stopSequences: params.stop,
                        temperature: params.temperature,
                        maxOutputTokens: params.model.includes('vision')
                            ? undefined
                            : params.maxTokens,
                        topP: params.topP
                    },
                    tools:
                        !params.model.includes('vision') && params.tools != null
                            ? {
                                  functionDeclarations:
                                      formatToolsToGeminiAITools(params.tools)
                              }
                            : undefined
                },
                {
                    signal: params.signal
                }
            )

            let errorCount = 0

            const stream = new TransformStream<ChatPart, ChatPart>()

            const iterable = readableStreamToAsyncIterable<ChatPart>(
                stream.readable
            )

            const jsonParser = new JSONParser()

            const writable = stream.writable.getWriter()

            jsonParser.onEnd = async () => {
                await writable.close()
            }

            jsonParser.onValue = async ({ value }) => {
                const transformValue = value as unknown as ChatResponse

                if (transformValue.candidates && transformValue.candidates[0]) {
                    const parts = transformValue.candidates[0]?.content?.parts

                    if (parts == null || parts.length < 1) {
                        throw new Error(JSON.stringify(value))
                    }

                    for (const part of parts) {
                        await writable.write(part)
                    }
                }
            }

            await sse(
                response,
                async (rawData) => {
                    jsonParser.write(rawData)
                    return true
                },
                0
            )

            let content = ''

            let isOldVisionModel = params.model.includes('vision')

            const functionCall: ChatCompletionMessageFunctionCall & {
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                arguments: any
            } = {
                name: '',
                args: '',
                arguments: ''
            }

            for await (const chunk of iterable) {
                const messagePart = partAsType<ChatMessagePart>(chunk)
                const chatFunctionCallingPart =
                    partAsType<ChatFunctionCallingPart>(chunk)

                if (messagePart.text) {
                    if (params.tools != null) {
                        content = messagePart.text
                    } else {
                        content += messagePart.text
                    }

                    // match /w*model:
                    if (isOldVisionModel && /\s*model:\s*/.test(content)) {
                        isOldVisionModel = false
                        content = messagePart.text.replace(/\s*model:\s*/, '')
                    }
                }

                const deltaFunctionCall = chatFunctionCallingPart.functionCall

                if (deltaFunctionCall) {
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    let args: any =
                        deltaFunctionCall.args?.input ?? deltaFunctionCall.args

                    try {
                        let parsedArgs = JSON.parse(args)

                        if (typeof parsedArgs !== 'string') {
                            args = parsedArgs
                        }

                        parsedArgs = JSON.parse(args)

                        if (typeof parsedArgs !== 'string') {
                            args = parsedArgs
                        }
                        // eslint-disable-next-line @typescript-eslint/no-unused-vars
                    } catch (e) {}

                    functionCall.args = JSON.stringify(args)

                    functionCall.name = deltaFunctionCall.name

                    functionCall.arguments = deltaFunctionCall.args
                }

                try {
                    const messageChunk = new AIMessageChunk(content)

                    messageChunk.additional_kwargs = {
                        function_call:
                            functionCall.name.length > 0
                                ? {
                                      name: functionCall.name,
                                      arguments: functionCall.args,
                                      args: functionCall.arguments
                                  }
                                : undefined
                        // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    } as any

                    messageChunk.content = content

                    const generationChunk = new ChatGenerationChunk({
                        message: messageChunk,
                        text: messageChunk.content
                    })

                    yield generationChunk
                    content = messageChunk.content
                } catch (e) {
                    if (errorCount > 5) {
                        logger.error('error with chunk', chunk)
                        throw new ChatLunaError(
                            ChatLunaErrorCode.API_REQUEST_FAILED,
                            e
                        )
                    } else {
                        errorCount++
                        continue
                    }
                }
            }
        } catch (e) {
            if (e instanceof ChatLunaError) {
                throw e
            } else {
                throw new ChatLunaError(ChatLunaErrorCode.API_REQUEST_FAILED, e)
            }
        }
    }

    async embeddings(
        params: EmbeddingsRequestParams
    ): Promise<number[] | number[][]> {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        let data: CreateEmbeddingResponse | string

        if (typeof params.input === 'string') {
            params.input = [params.input]
        }

        try {
            const response = await this._post(
                `models/${params.model}:batchEmbedContents`,
                {
                    requests: params.input.map((input) => {
                        return {
                            model: `models/${params.model}`,
                            content: {
                                parts: [
                                    {
                                        text: input
                                    }
                                ]
                            }
                        }
                    })
                }
            )

            data = await response.text()

            data = JSON.parse(data) as CreateEmbeddingResponse

            if (data.embeddings && data.embeddings.length > 0) {
                return data.embeddings.map((embedding) => {
                    return embedding.values
                })
            }

            throw new Error(
                'error when calling gemini embeddings, Result: ' +
                    JSON.stringify(data)
            )
        } catch (e) {
            const error = new Error(
                'error when calling gemini embeddings, Result: ' +
                    JSON.stringify(data)
            )

            error.stack = e.stack
            error.cause = e.cause
            logger.debug(e)

            throw new ChatLunaError(ChatLunaErrorCode.API_REQUEST_FAILED, error)
        }
    }

    async getModels(): Promise<string[]> {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        let data: any
        try {
            const response = await this._get('models')
            data = await response.text()
            data = JSON.parse(data as string)

            if (!data.models || !data.models.length) {
                throw new Error(
                    'error when listing gemini models, Result:' +
                        JSON.stringify(data)
                )
            }

            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            return (<Record<string, any>[]>data.models)
                .map((model) => model.name as string)
                .filter(
                    (model) =>
                        model.includes('gemini') || model.includes('embedding')
                )
        } catch (e) {
            const error = new Error(
                'error when listing gemini models, Result: ' +
                    JSON.stringify(data)
            )

            error.stack = e.stack
            error.cause = e.cause

            throw error
        }
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    private _post(url: string, data: any, params: fetchType.RequestInit = {}) {
        const requestUrl = this._concatUrl(url)

        for (const key in data) {
            if (data[key] === undefined) {
                delete data[key]
            }
        }

        const body = JSON.stringify(data)

        return this._plugin.fetch(requestUrl, {
            body,
            headers: this._buildHeaders(),
            method: 'POST',
            ...params
        })
    }

    private _get(url: string) {
        const requestUrl = this._concatUrl(url)

        return this._plugin.fetch(requestUrl, {
            method: 'GET',
            headers: this._buildHeaders()
        })
    }

    private _concatUrl(url: string) {
        const apiEndPoint = this._config.apiEndpoint

        // match the apiEndPoint ends with '/v1' or '/v1/' using regex

        if (apiEndPoint.endsWith('/')) {
            return apiEndPoint + url + `?key=${this._config.apiKey}`
        }

        return apiEndPoint + '/' + url + `?key=${this._config.apiKey}`
    }

    private _buildHeaders() {
        return {
            /*  Authorization: `Bearer ${this._config.apiKey}`, */
            'Content-Type': 'application/json'
        }
    }

    async init(): Promise<void> {}

    async dispose(): Promise<void> {}
}
