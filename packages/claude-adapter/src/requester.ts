import * as fetchType from 'undici/types/fetch'
import {
    ModelRequester,
    ModelRequestParams
} from 'koishi-plugin-chatluna/llm-core/platform/api'
import { ChatGenerationChunk } from '@langchain/core/outputs'
import { ClientConfig } from 'koishi-plugin-chatluna/llm-core/platform/config'
import {
    ChatLunaError,
    ChatLunaErrorCode
} from 'koishi-plugin-chatluna/utils/error'
import { Context } from 'koishi'
import { sseIterable } from 'koishi-plugin-chatluna/utils/sse'
import { Config } from '.'
import { ClaudeDeltaResponse, ClaudeRequest } from './types'
import {
    convertDeltaToMessageChunk,
    formatToolsToClaudeTools,
    langchainMessageToClaudeMessage
} from './utils'
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'

export class ClaudeRequester extends ModelRequester {
    constructor(
        private ctx: Context,
        private _pluginConfig: Config,
        private _config: ClientConfig,
        private _plugin: ChatLunaPlugin
    ) {
        super()
    }

    async *completionStream(
        params: ModelRequestParams
    ): AsyncGenerator<ChatGenerationChunk> {
        const response = await this._post('messages', {
            model: params.model,
            max_tokens: params.maxTokens ?? 4096,
            temperature: params.temperature,
            top_p: params.topP,
            stop_sequences:
                typeof params.stop === 'string' ? [params.stop] : params.stop,
            stream: true,
            messages: langchainMessageToClaudeMessage(
                params.input,
                params.model
            ),
            tools:
                params.tools != null
                    ? formatToolsToClaudeTools(params.tools)
                    : undefined
        } satisfies ClaudeRequest)

        const iterator = sseIterable(response)

        for await (const event of iterator) {
            if (event.event === 'ping') continue

            if (event.event === 'error') {
                throw new ChatLunaError(
                    ChatLunaErrorCode.API_REQUEST_FAILED,
                    new Error(event.data)
                )
            }

            if (event.event === 'message_delta') return

            const chunk = event.data

            if (chunk === '[DONE]') {
                return
            }

            const parsedRawChunk = JSON.parse(chunk) as ClaudeDeltaResponse

            const parsedChunk = convertDeltaToMessageChunk(parsedRawChunk)

            // console.log(findTools, parsedRawChunk, parsedChunk)

            if (parsedChunk == null) continue

            yield new ChatGenerationChunk({
                message: parsedChunk,
                text: parsedChunk.content as string
            })
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

        // console.log('POST', requestUrl, body)

        return this._plugin.fetch(requestUrl, {
            body,
            headers: this._buildHeaders(),
            method: 'POST',
            ...params
        })
    }

    private _buildHeaders() {
        return {
            'x-api-key': this._config.apiKey,
            'anthropic-version': '2023-06-01',
            'Content-Type': 'application/json'
        }
    }

    private _concatUrl(url: string): string {
        const apiEndPoint = this._config.apiEndpoint

        // match the apiEndPoint ends with '/v1' or '/v1/' using regex
        if (!apiEndPoint.match(/\/v1\/?$/)) {
            if (apiEndPoint.endsWith('/')) {
                return apiEndPoint + 'v1/' + url
            }

            return apiEndPoint + '/v1/' + url
        }

        if (apiEndPoint.endsWith('/')) {
            return apiEndPoint + url
        }

        return apiEndPoint + '/' + url
    }

    dispose(): Promise<void> {
        return Promise.resolve(undefined)
    }

    init(): Promise<void> {
        return Promise.resolve(undefined)
    }
}
