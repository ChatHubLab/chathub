import { ChatGenerationChunk } from '@langchain/core/outputs'
import {
    ModelRequester,
    ModelRequestParams
} from 'koishi-plugin-chatluna/llm-core/platform/api'
import * as fetchType from 'undici/types/fetch'
import { Config } from '.'
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'
import { Context } from 'koishi'
import { AssistantStreamResponse, DifyClientConfig } from './types'
import { sseIterable } from 'koishi-plugin-chatluna/utils/sse'
import {
    ChatLunaError,
    ChatLunaErrorCode
} from 'koishi-plugin-chatluna/utils/error'
import { AIMessageChunk } from '@langchain/core/messages'

export class DifyRequester extends ModelRequester {
    constructor(
        private _ctx: Context,
        private _config: DifyClientConfig,
        private _pluginConfig: Config,
        private _plugin: ChatLunaPlugin
    ) {
        super()
    }

    async *completionStream(
        params: ModelRequestParams
    ): AsyncGenerator<ChatGenerationChunk> {
        const config = this._config.additionalModel.get(params.model)

        if (!config) {
            throw new ChatLunaError(
                ChatLunaErrorCode.MODEL_NOT_FOUND,
                new Error(`Dify model not found: ${params.model}`)
            )
        }
        const conversationId = params.id

        if (!conversationId) {
            throw new ChatLunaError(
                ChatLunaErrorCode.MODEL_NOT_FOUND,
                new Error(`Dify model not found: ${params.model}`)
            )
        }

        const difyConversationId = await this.getDifyConversationId(
            conversationId,
            config
        )

        if (config.workflowType !== 'Workflow') {
            const iter = this._agentStream(
                difyConversationId,
                params.input[params.input.length - 1].content as string,
                conversationId,
                config
            )

            for await (const chunk of iter) {
                yield chunk
            }
        } else {
            const iter = this._workflowStream(params, config)

            for await (const chunk of iter) {
                yield chunk
            }
        }
    }

    private async *_agentStream(
        difyConversationId: string,
        input: string,
        conversationId: string,
        config: { apiKey: string; workflowName: string; workflowType: string }
    ): AsyncGenerator<ChatGenerationChunk> {
        const response = await this._post(
            '/chat-messages',
            {
                query: input,
                response_mode: 'streaming',
                inputs: {},
                user: 'chatluna',
                conversation_id:
                    difyConversationId == null ? '' : difyConversationId
            },
            config.apiKey
        )

        const iterator = sseIterable(response)
        let updatedDifyConversationId: string | undefined

        for await (const event of iterator) {
            const chunk = event.data

            console.log(chunk)
            if (chunk == null) {
                continue
            }

            let data: AssistantStreamResponse

            try {
                data = JSON.parse(chunk)
            } catch (err) {
                this._ctx.logger.error(
                    'error when parsing dify stream response, Result:' + chunk
                )
                throw new ChatLunaError(
                    ChatLunaErrorCode.API_REQUEST_FAILED,
                    new Error(
                        'error when calling qwen completion, Result: ' + chunk
                    )
                )
            }

            if (data.event === 'error') {
                throw new ChatLunaError(
                    ChatLunaErrorCode.API_REQUEST_FAILED,
                    new Error(
                        'error when calling dify completion, Result:' + chunk
                    )
                )
            }

            const content = data.answer

            if (content != null) {
                const messageChunk = new AIMessageChunk(content)
                const generationChunk = new ChatGenerationChunk({
                    message: messageChunk,
                    text: content
                })

                yield generationChunk
            }

            updatedDifyConversationId = data.conversation_id

            if (data.event === 'message_end') {
                await this.updateDifyConversationId(
                    conversationId,
                    config.workflowName,
                    updatedDifyConversationId
                )
                break
            }
        }
    }

    private async *_workflowStream(
        params: ModelRequestParams,
        config: { apiKey: string; workflowName: string; workflowType: string }
    ): AsyncGenerator<ChatGenerationChunk> {
        const response = await this._post(
            '/workflows/run',
            {
                response_mode: 'streaming',
                inputs: {
                    prompt: params.input[params.input.length - 1]
                        .content as string,
                    chatluna_history: params.input.map((it) => {
                        return {
                            role: it.getType(),
                            content: it.content
                        }
                    })
                },
                user: 'chatluna'
            },
            config.apiKey
        )

        const iterator = sseIterable(response)

        for await (const event of iterator) {
            const chunk = event.data

            console.log(chunk)
            if (chunk == null) {
                continue
            }

            let data: AssistantStreamResponse

            try {
                data = JSON.parse(chunk)
            } catch (err) {
                this._ctx.logger.error(
                    'error when parsing dify stream response, Result:' + chunk
                )
                throw new ChatLunaError(
                    ChatLunaErrorCode.API_REQUEST_FAILED,
                    new Error(
                        'error when calling qwen completion, Result: ' + chunk
                    )
                )
            }

            if (data.event === 'error') {
                throw new ChatLunaError(
                    ChatLunaErrorCode.API_REQUEST_FAILED,
                    new Error(
                        'error when calling dify completion, Result:' + chunk
                    )
                )
            }

            const content = data.answer

            if (content != null) {
                const messageChunk = new AIMessageChunk(content)
                const generationChunk = new ChatGenerationChunk({
                    message: messageChunk,
                    text: content
                })

                yield generationChunk
            }
        }
    }

    private async getDifyConversationId(
        conversationId: string,
        config: { apiKey: string; workflowName: string; workflowType: string }
    ) {
        return this._ctx.chatluna.cache.get(
            'chathub/keys',
            'dify/' + conversationId + '/' + config.workflowName
        )
    }

    private async updateDifyConversationId(
        conversationId: string,
        workflowName: string,
        difyConversationId: string
    ) {
        return this._ctx.chatluna.cache.set(
            'chathub/keys',
            'dify/' + conversationId + '/' + workflowName,
            difyConversationId
        )
    }

    private _post(
        url: string,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        data: any,
        apiKey: string,
        params: fetchType.RequestInit = {}
    ) {
        const requestUrl = this._concatUrl(url)

        const body = JSON.stringify(data)

        console.log(
            'Dify request url: ' + requestUrl + ', body: ' + body,
            params,
            this._buildHeaders(apiKey)
        )

        return this._plugin.fetch(requestUrl, {
            body,
            headers: this._buildHeaders(apiKey),
            method: 'POST',
            ...params
        })
    }

    private _buildHeaders(apiKey: string) {
        return {
            Authorization: `Bearer ${apiKey}`,
            'Content-Type': 'application/json'
        }
    }

    private _concatUrl(url: string): string {
        return this._pluginConfig.apiURL + url
    }

    async init(): Promise<void> {}

    async dispose(model?: string, id?: string): Promise<void> {
        if (id == null || model == null) {
            this._ctx.logger.warn('Dify clear: model or id is null')
            return
        }
        const conversationId = id
        const config = this._config.additionalModel.get(model)
        const difyConversationId = await this.getDifyConversationId(
            conversationId,
            config
        )

        if (difyConversationId) {
            await this._plugin
                .fetch(
                    this._concatUrl('/conversations/' + difyConversationId),
                    {
                        headers: this._buildHeaders(config.apiKey),
                        method: 'DELETE'
                    }
                )
                .then(async (res) => {
                    if (res.ok) {
                        this._ctx.logger.info('Dify clear: success')
                    } else {
                        this._ctx.logger.warn(
                            'Dify clear: failed: ' + (await res.text())
                        )
                    }
                })
        }

        // TODO: delete
    }
}
