import { Context } from 'koishi'
import { PlatformModelAndEmbeddingsClient } from 'koishi-plugin-chatluna/llm-core/platform/client'
import { ClientConfig } from 'koishi-plugin-chatluna/llm-core/platform/config'
import {
    ChatLunaChatModel,
    ChatLunaEmbeddings
} from 'koishi-plugin-chatluna/llm-core/platform/model'
import {
    ModelInfo,
    ModelType
} from 'koishi-plugin-chatluna/llm-core/platform/types'
import {
    ChatLunaError,
    ChatLunaErrorCode
} from 'koishi-plugin-chatluna/utils/error'
import { Config } from '.'
import { OllamaRequester } from './requester'
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'

export class OllamaClient extends PlatformModelAndEmbeddingsClient<ClientConfig> {
    platform = 'ollama'

    private _requester: OllamaRequester

    private _models: Record<string, ModelInfo>

    constructor(
        ctx: Context,
        private _config: Config,
        clientConfig: ClientConfig,
        plugin: ChatLunaPlugin<ClientConfig, Config>
    ) {
        super(ctx, clientConfig)

        this._requester = new OllamaRequester(clientConfig, plugin)
    }

    async init(): Promise<void> {
        await this.getModels()
    }

    async refreshModels(): Promise<ModelInfo[]> {
        try {
            const rawModels = await this._requester.getModels()

            return rawModels.map((model) => {
                return {
                    name: model,
                    type:
                        model.includes('embed') ||
                        model.includes('all-minilm') ||
                        model.includes('bge') ||
                        model.includes('paraphrase-multilingual')
                            ? ModelType.embeddings
                            : ModelType.llm,
                    supportMode: ['all'],
                    maxTokens: ((model: string) => {
                        if (model.startsWith('llama3')) {
                            return 32000
                        }

                        return 128000
                    })(model)
                }
            })
        } catch (e) {
            throw new ChatLunaError(ChatLunaErrorCode.MODEL_INIT_ERROR, e)
        }
    }

    async getModels(): Promise<ModelInfo[]> {
        if (this._models) {
            return Object.values(this._models)
        }

        const models = await this.refreshModels()

        this._models = {}

        for (const model of models) {
            this._models[model.name] = model
        }

        return models
    }

    protected _createModel(
        model: string
    ): ChatLunaChatModel | ChatLunaEmbeddings {
        const info = this._models[model]

        if (info == null) {
            throw new ChatLunaError(ChatLunaErrorCode.MODEL_NOT_FOUND)
        }

        if (info.type === ModelType.llm) {
            return new ChatLunaChatModel({
                modelInfo: info,
                requester: this._requester,
                model,
                maxTokenLimit: this._config.maxTokens,
                modelMaxContextSize: info.maxTokens,
                frequencyPenalty: this._config.frequencyPenalty,
                presencePenalty: this._config.presencePenalty,
                timeout: this._config.timeout,
                temperature: this._config.temperature,
                maxRetries: this._config.maxRetries,
                llmType: 'ollama'
            })
        }

        return new ChatLunaEmbeddings({
            model,
            client: this._requester,
            maxRetries: this._config.maxRetries
        })
    }
}
