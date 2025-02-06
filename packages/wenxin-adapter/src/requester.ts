import {
    EmbeddingsRequester,
    EmbeddingsRequestParams,
    ModelRequester,
    ModelRequestParams
} from 'koishi-plugin-chatluna/llm-core/platform/api'
import { ClientConfig } from 'koishi-plugin-chatluna/llm-core/platform/config'
import * as fetchType from 'undici/types/fetch'
import { ChatGenerationChunk } from '@langchain/core/outputs'
import {
    ChatCompletionResponse,
    CreateEmbeddingResponse,
    WenxinMessage,
    WenxinMessageRole
} from './types'
import {
    ChatLunaError,
    ChatLunaErrorCode
} from 'koishi-plugin-chatluna/utils/error'
import { sseIterable } from 'koishi-plugin-chatluna/utils/sse'
import {
    convertDeltaToMessageChunk,
    formatToolsToWenxinTools,
    langchainMessageToWenXinMessage,
    modelMappedUrl
} from './utils'
import { Config } from '.'
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'

export class WenxinRequester
    extends ModelRequester
    implements EmbeddingsRequester
{
    private _accessToken: string | undefined

    constructor(
        private _config: ClientConfig,
        private _pluginConfig: Config,
        private _plugin: ChatLunaPlugin<ClientConfig, Config>
    ) {
        super()
    }

    async *completionStream(
        params: ModelRequestParams
    ): AsyncGenerator<ChatGenerationChunk> {
        await this.init()

        let messages = params.input

        // Wenxin requires the system message to be put in the params, not messages array
        const systemMessage = messages.find(
            (message) => message._getType() === 'system'
        )
        if (systemMessage) {
            // eslint-disable-next-line no-param-reassign
            messages = messages.filter((message) => message !== systemMessage)
        }
        const messagesMapped: WenxinMessage[] =
            langchainMessageToWenXinMessage(messages)

        try {
            const response = await this._post(
                modelMappedUrl[params.model](this._accessToken),
                {
                    messages: messagesMapped,
                    stream: true,
                    system: systemMessage ? systemMessage.content : undefined,
                    temperature: params.temperature,
                    top_p: params.topP,
                    penalty_score: params.presencePenalty,
                    disable_search: !this._pluginConfig.enableSearch,
                    enable_citation: this._pluginConfig.enableSearch,
                    functions:
                        params.tools != null
                            ? formatToolsToWenxinTools(params.tools)
                            : undefined
                },
                {
                    signal: params.signal
                }
            )

            const iterator = sseIterable(response)

            const defaultRole: WenxinMessageRole = 'assistant'

            let errorCount = 0

            for await (const event of iterator) {
                const chunk = event.data
                if (chunk === '[DONE]') {
                    return
                }

                try {
                    const data = JSON.parse(chunk) as ChatCompletionResponse

                    const message = data as ChatCompletionResponse
                    if (!message) {
                        continue
                    }

                    if (message.need_clear_history) {
                        errorCount = 114514
                        throw new ChatLunaError(
                            ChatLunaErrorCode.API_UNSAFE_CONTENT,
                            new Error(
                                'error when calling wenxin completion, Result: ' +
                                    chunk
                            )
                        )
                    }

                    const messageChunk = convertDeltaToMessageChunk(
                        message,
                        defaultRole
                    )

                    const generationChunk = new ChatGenerationChunk({
                        message: messageChunk,
                        text: messageChunk.content as string
                    })

                    yield generationChunk
                } catch (e) {
                    console.error(e, chunk)
                    if (errorCount > 5) {
                        if (e instanceof ChatLunaError) {
                            throw e
                        }
                        throw new ChatLunaError(
                            ChatLunaErrorCode.API_REQUEST_FAILED,
                            new Error(
                                'error when calling wenxin completion, Result: ' +
                                    chunk
                            )
                        )
                    } else {
                        errorCount++
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
        await this.init()

        if (
            typeof params.input === 'string' &&
            params.input.trim().length < 1
        ) {
            return []
        }

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        let data: CreateEmbeddingResponse | string

        try {
            const response = await this._post(
                `https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/embedding-v1?access_token=${this._accessToken}`,
                {
                    input:
                        params.input instanceof Array
                            ? params.input
                            : [params.input]
                }
            )

            data = await response.text()

            data = JSON.parse(data) as CreateEmbeddingResponse

            if (data.data && data.data.length > 0) {
                const rawEmbeddings = (
                    data as CreateEmbeddingResponse
                ).data.map((it) => it.embedding)

                if (params.input instanceof Array) {
                    return rawEmbeddings
                }

                return rawEmbeddings[0]
            }

            throw new Error(
                'error when calling wenxin embeddings, Result: ' +
                    JSON.stringify(data)
            )
        } catch (e) {
            const error = new Error(
                'error when calling wenxin embeddings, Result: ' +
                    JSON.stringify(data)
            )

            error.stack = e.stack
            error.cause = e.cause

            throw new ChatLunaError(ChatLunaErrorCode.API_REQUEST_FAILED, error)
        }
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    private _post(url: string, data: any, params: fetchType.RequestInit = {}) {
        const body = JSON.stringify(data)

        return this._plugin.fetch(url, {
            body,
            headers: this._buildHeaders(),
            method: 'POST',
            ...params
        })
    }

    private _buildHeaders() {
        return {
            'Content-Type': 'application/json'
        }
    }

    private async _getAccessToken(): Promise<string> {
        // eslint-disable-next-line max-len
        const url = `https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=${this._config.apiKey}&client_secret=${this._config.apiEndpoint}`
        const response = await this._plugin.fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                Accept: 'application/json'
            }
        })
        if (!response.ok) {
            const text = await response.text()
            const error = new Error(
                `Baidu get access token failed with status code ${response.status}, response: ${text}`
            )
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            ;(error as any).response = response
            throw new ChatLunaError(ChatLunaErrorCode.API_REQUEST_FAILED, error)
        }
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const json = (await response.json()) as any
        return json.access_token
    }

    async init(): Promise<void> {
        if (this._accessToken == null) {
            this._accessToken = await this._getAccessToken()
        }
    }

    async dispose(): Promise<void> {}
}
