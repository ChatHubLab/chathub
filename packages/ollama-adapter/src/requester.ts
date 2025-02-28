import { AIMessageChunk } from '@langchain/core/messages'
import { ChatGenerationChunk } from '@langchain/core/outputs'
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
import {
    OllamaDeltaResponse,
    OllamaEmbedResponse,
    OllamaRequest
} from './types'
import { langchainMessageToOllamaMessage } from './utils'
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'
import { Config } from '.'

export class OllamaRequester
    extends ModelRequester
    implements EmbeddingsRequester
{
    constructor(
        private _config: ClientConfig,
        private _plugin: ChatLunaPlugin<ClientConfig, Config>
    ) {
        super()
    }

    async *completionStream(
        params: ModelRequestParams
    ): AsyncGenerator<ChatGenerationChunk> {
        try {
            const response = await this._post(
                'api/chat',
                {
                    model: params.model,
                    messages: langchainMessageToOllamaMessage(
                        params.input,
                        this._plugin.config.supportImage
                    ),
                    keep_alive: this._plugin.config.keepAlive ? -1 : undefined,
                    options: {
                        temperature: params.temperature,
                        // top_k: params.n,
                        top_p: params.topP,
                        stop:
                            typeof params.stop === 'string'
                                ? params.stop
                                : params.stop?.[0]
                    },
                    stream: true
                } satisfies OllamaRequest,
                {
                    signal: params.signal
                }
            )

            const stream = new TransformStream<string, OllamaDeltaResponse>()

            const iterable = readableStreamToAsyncIterable<OllamaDeltaResponse>(
                stream.readable
            )

            const writable = stream.writable.getWriter()

            let buffer = ''

            sse(
                response,
                async (rawData) => {
                    buffer += rawData

                    const parts = buffer.split('\n')

                    buffer = parts.pop() ?? ''

                    for (const part of parts) {
                        try {
                            writable.write(JSON.parse(part))
                        } catch (error) {
                            console.warn('invalid json: ', part)
                        }
                    }
                },
                0
            )

            for await (const chunk of iterable) {
                try {
                    const content = chunk.message.content

                    const generationChunk = new ChatGenerationChunk({
                        message: new AIMessageChunk(content),
                        text: content
                    })
                    yield generationChunk

                    if (chunk.done) {
                        return
                    }
                } catch (e) {
                    throw new ChatLunaError(
                        ChatLunaErrorCode.API_REQUEST_FAILED,
                        new Error(
                            'error when calling ollama completion, Result: ' +
                                chunk
                        )
                    )
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
        let data: OllamaEmbedResponse | string

        try {
            const response = await this._post('api/embed', {
                input: params.input,
                model: params.model,
                keep_alive: this._plugin.config.keepAlive ? -1 : undefined
            })

            data = await response.text()

            data = JSON.parse(data as string) as OllamaEmbedResponse

            if (data.embeddings && data.embeddings.length > 0) {
                if (typeof params.input === 'string') {
                    return data.embeddings[0]
                }
                return data.embeddings
            }

            throw new Error(
                'error when calling ollama embeddings, Result: ' +
                    JSON.stringify(data)
            )
        } catch (e) {
            const error = new Error(
                'error when calling ollama embeddings, Result: ' +
                    JSON.stringify(data)
            )

            throw new ChatLunaError(ChatLunaErrorCode.API_REQUEST_FAILED, error)
        }
    }

    async getModels(): Promise<string[]> {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        let data: any
        try {
            const response = await this._get('api/tags')
            data = await response.text()
            data = JSON.parse(data as string)

            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            return (<Record<string, any>[]>data.models).map(
                (model) => model.name
            )
        } catch (e) {
            const error = new Error(
                'error when listing ollama models, Result: ' +
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

    private _buildHeaders() {
        return {
            Authorization: `Bearer ${this._config.apiKey}`,
            'Content-Type': 'application/json'
        }
    }

    private _concatUrl(url: string): string {
        const apiEndPoint = this._config.apiEndpoint

        if (apiEndPoint.endsWith('/')) {
            return apiEndPoint + url
        }

        return apiEndPoint + '/' + url
    }

    async init(): Promise<void> {}

    async dispose(): Promise<void> {}
}
