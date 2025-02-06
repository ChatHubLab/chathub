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
import { sseIterable } from 'koishi-plugin-chatluna/utils/sse'
import * as fetchType from 'undici/types/fetch'
import { Config } from '.'
import {
    ChatCompletionResponse,
    ChatCompletionResponseMessageRoleEnum,
    CreateEmbeddingResponse
} from './types'
import {
    convertDeltaToMessageChunk,
    formatToolsToHunyuanTools,
    langchainMessageToHunyuanMessage
} from './utils'
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'

export class HunyuanRequester
    extends ModelRequester
    implements EmbeddingsRequester
{
    constructor(
        private _config: ClientConfig,
        private _pluginConfig: Config,
        private _plugin: ChatLunaPlugin
    ) {
        super()
    }

    async *completionStream(
        params: ModelRequestParams
    ): AsyncGenerator<ChatGenerationChunk> {
        try {
            const response = await this._post(
                'chat/completions',
                {
                    model: params.model,
                    messages: langchainMessageToHunyuanMessage(
                        params.input,
                        params.model
                    ),
                    tools:
                        params.tools != null && !params.model.includes('vision')
                            ? formatToolsToHunyuanTools(params.tools)
                            : undefined,
                    stream: true,
                    top_p: params.topP,
                    temperature: params.temperature,
                    enable_enhancement: params.model.includes('vision')
                        ? undefined
                        : this._pluginConfig.enableSearch
                },
                {
                    signal: params.signal
                }
            )

            const iterator = sseIterable(response)

            const defaultRole: ChatCompletionResponseMessageRoleEnum =
                'assistant'

            for await (const event of iterator) {
                const chunk = event.data

                if (chunk === '[DONE]') {
                    return
                }

                let data: ChatCompletionResponse

                try {
                    data = JSON.parse(chunk)
                } catch (err) {
                    throw new ChatLunaError(
                        ChatLunaErrorCode.API_REQUEST_FAILED,
                        new Error(
                            'error when calling Hunyuan completion, Result: ' +
                                chunk
                        )
                    )
                }

                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                if ((data as any).Response) {
                    // check DataInspectionFailed

                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    if ((data as any).Error?.code.include('IllegalDetected')) {
                        throw new ChatLunaError(
                            ChatLunaErrorCode.API_UNSAFE_CONTENT,
                            new Error(
                                'Unsafe content detected, please try again.' +
                                    chunk
                            )
                        )
                    }

                    throw new ChatLunaError(
                        ChatLunaErrorCode.API_REQUEST_FAILED,
                        new Error(
                            'error when calling Hunyuan completion, Result: ' +
                                chunk
                        )
                    )
                }

                const choice = data.choices?.[0]

                if (!choice) {
                    continue
                }

                const messageChunk = convertDeltaToMessageChunk(
                    choice.delta,
                    defaultRole
                )

                const generationChunk = new ChatGenerationChunk({
                    message: messageChunk,
                    text: messageChunk.content as string
                })

                yield generationChunk

                if (choice.finish_reason === 'stop') {
                    break
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

        try {
            const response = await this._post('embeddings', {
                input: params.input,
                model: params.model
            })

            data = await response.text()

            data = JSON.parse(data as string) as CreateEmbeddingResponse

            if (data.data && data.data.length > 0) {
                return (data as CreateEmbeddingResponse).data.map(
                    (it) => it.embedding
                )
            }

            throw new Error(
                'error when calling Hunyuan embeddings, Result: ' +
                    JSON.stringify(data)
            )
        } catch (e) {
            const error = new Error(
                'error when calling Hunyuan embeddings, Result: ' +
                    JSON.stringify(data)
            )

            console.error(e)

            throw new ChatLunaError(ChatLunaErrorCode.API_REQUEST_FAILED, error)
        }
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    private _post(url: string, data: any, params: fetchType.RequestInit = {}) {
        const requestUrl = this._concatUrl(url)

        const body = JSON.stringify(data)

        return this._plugin.fetch(requestUrl, {
            body,
            headers: this._buildHeaders(!url.includes('text-embedding')),
            method: 'POST',
            ...params
        })
    }

    private _buildHeaders(stream: boolean = true) {
        return {
            Authorization: `Bearer ${this._config.apiKey}`,
            'Content-Type': 'application/json'
        }
    }

    private _concatUrl(url: string): string {
        return 'https://api.hunyuan.cloud.tencent.com/v1/' + url
    }

    async init(): Promise<void> {}

    async dispose(): Promise<void> {}
}
