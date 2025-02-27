import { Context } from 'koishi'
import { PlatformModelClient } from 'koishi-plugin-chatluna/llm-core/platform/client'
import { ClientConfig } from 'koishi-plugin-chatluna/llm-core/platform/config'
import { ChatLunaChatModel } from 'koishi-plugin-chatluna/llm-core/platform/model'
import {
    ModelInfo,
    ModelType
} from 'koishi-plugin-chatluna/llm-core/platform/types'
import { Config } from '.'
import { ClaudeRequester } from './requester'
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'

export class ClaudeClient extends PlatformModelClient {
    platform = 'claude'

    private _models: ModelInfo[]

    private _requester: ClaudeRequester

    constructor(
        ctx: Context,
        private _config: Config,
        private _clientConfig: ClientConfig,
        plugin: ChatLunaPlugin
    ) {
        super(ctx, _clientConfig)
        this._requester = new ClaudeRequester(
            ctx,
            _config,
            _clientConfig,
            plugin
        )
    }

    async init(): Promise<void> {
        if (this._models) {
            return
        }

        this._models = await this.getModels()
    }

    async getModels(): Promise<ModelInfo[]> {
        if (this._models) {
            return this._models
        }

        return await this.refreshModels()
    }

    async refreshModels(): Promise<ModelInfo[]> {
        return [
            'claude-3-5-sonnet-20240620',
            'claude-3-opus-20240229',
            'claude-3-sonnet-20240229',
            'claude-3-5-sonnet-20241022',
            'claude-3-7-sonnet-20250219',
            'claude-3-5-haiku-20241022',
            'claude-3-haiku-20240307',
            'claude-2.1',
            'claude-2.0',
            'claude-instant-1.2'
        ].map((model) => {
            return {
                name: model,
                maxTokens:
                    model.includes('2.0') || model.includes('1.2')
                        ? 1000000
                        : 2000000,
                supportMode: ['all'],
                functionCall: model.includes('claude-3'),
                type: ModelType.llm
            }
        })
    }

    protected _createModel(model: string): ChatLunaChatModel {
        const info = this._models.find((m) => m.name === model)
        return new ChatLunaChatModel({
            requester: this._requester,
            modelInfo: this._models[0],
            model,
            maxTokenLimit: this._config.maxTokens,
            modelMaxContextSize: info.maxTokens ?? 100000,
            timeout: this._config.timeout,
            maxRetries: this._config.maxRetries,
            llmType: model
        })
    }
}
