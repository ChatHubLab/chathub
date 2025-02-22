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
    langchainMessageToWenXinMessage
} from './utils'
import { Config, logger } from '.'
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'

export class WenxinRequester
    extends ModelRequester
    implements EmbeddingsRequester
{
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

        const messagesMapped: WenxinMessage[] = langchainMessageToWenXinMessage(
            params.input
        )

        try {
            const response = await this._post(
                'v2/chat/completions',
                {
                    messages: messagesMapped,
                    stream: true,
                    temperature: params.temperature,
                    top_p: params.topP,
                    penalty_score: params.presencePenalty,
                    disable_search: !this._pluginConfig.enableSearch,

                    model: params.model,
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

            let defaultRole: WenxinMessageRole = 'assistant'

            let errorCount = 0

            let reasoningContent = ''

            let reasoningTime = 0
            let isSetReasoingTime = false

            for await (const event of iterator) {
                const chunk = event.data
                if (chunk === '[DONE]') {
                    break
                }

                try {
                    const data = JSON.parse(chunk)

                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    if ((data as any).error) {
                        throw new ChatLunaError(
                            ChatLunaErrorCode.API_REQUEST_FAILED,
                            new Error(
                                'error when calling openai completion, Result: ' +
                                    chunk
                            )
                        )
                    }

                    const choice = data.choices?.[0]
                    if (!choice) {
                        continue
                    }

                    const { delta } = choice
                    const messageChunk = convertDeltaToMessageChunk(
                        delta,
                        defaultRole
                    )

                    if (delta.reasoning_content) {
                        reasoningContent = (reasoningContent +
                            delta.reasoning_content) as string

                        if (reasoningTime === 0) {
                            reasoningTime = Date.now()
                        }
                    }

                    if (
                        (delta.reasoning_content == null ||
                            delta.reasoning_content === '') &&
                        delta.content &&
                        delta.content.length > 0 &&
                        reasoningTime > 0 &&
                        !isSetReasoingTime
                    ) {
                        reasoningTime = Date.now() - reasoningTime
                        messageChunk.additional_kwargs.reasoning_time =
                            reasoningTime
                        isSetReasoingTime = true
                    }

                    defaultRole = (
                        (delta.role?.length ?? 0) > 0 ? delta.role : defaultRole
                    ) as WenxinMessageRole

                    const generationChunk = new ChatGenerationChunk({
                        message: messageChunk,
                        text: messageChunk.content as string
                    })

                    yield generationChunk
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

            if (reasoningContent.length > 0) {
                logger.debug(
                    `reasoning content: ${reasoningContent}. Use time: ${reasoningTime / 1000} s.`
                )
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
            const response = await this._post(`v2/embeddings`, {
                input:
                    params.input instanceof Array
                        ? params.input
                        : [params.input],
                model: params.model
            })

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

        return this._plugin.fetch(`https://qianfan.baidubce.com/${url}`, {
            body,
            headers: this._buildHeaders(),
            method: 'POST',
            ...params
        })
    }

    private _buildHeaders() {
        return {
            'Content-Type': 'application/json',
            appid: '',
            Authorization: `Bearer ${this._config.apiKey}`
        }
    }

    async dispose() {}
    async init() {}
}
