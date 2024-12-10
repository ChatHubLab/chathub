import { Context } from 'koishi'
import { PlatformModelAndEmbeddingsClient } from 'koishi-plugin-chatluna/llm-core/platform/client'
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
import { Config } from './index'
import { ZhipuRequester } from './requester'
import { ZhipuClientConfig } from './types'
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'

export class ZhipuClient extends PlatformModelAndEmbeddingsClient<ZhipuClientConfig> {
    platform = 'zhipu'

    private _requester: ZhipuRequester

    private _models: Record<string, ModelInfo>

    constructor(
        ctx: Context,
        private _config: Config,
        clientConfig: ZhipuClientConfig,
        plugin: ChatLunaPlugin<ZhipuClientConfig, Config>
    ) {
        super(ctx, clientConfig)

        this._requester = new ZhipuRequester(clientConfig, plugin)
    }

    async init(): Promise<void> {
        await this.getModels()
    }

    async refreshModels(): Promise<ModelInfo[]> {
        const rawModels = [
            ['GLM-4-Plus', 128000],
            ['GLM-4-0520', 128000],
            ['GLM-4-Long', 1024000],
            ['GLM-4-AirX', 8192],
            ['GLM-4-Air', 128000],
            ['GLM-4-FlashX', 128000],
            ['GLM-4-Flash', 128000],
            ['GLM-4V', 2048],
            ['GLM-4V-PLUS', 8192],
            ['GLM-4V-Flash', 8192]
            //   ['GLM-4-AllTools', 128000]
        ] as [string, number][]

        const embeddings = ['embedding-2', 'embedding-3']

        return rawModels
            .map(([model, maxTokens]) => {
                return {
                    name: model,
                    functionCall: model !== 'GLM-4V',
                    type: ModelType.llm,
                    supportMode: ['all'],
                    maxTokens
                } as ModelInfo
            })
            .concat(
                embeddings.map((model) => {
                    return {
                        name: model,
                        type: ModelType.embeddings,
                        supportMode: ['all'],
                        maxTokens: 8192,
                        functionCall: false
                    } satisfies ModelInfo
                })
            )
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

        if (info.type === ModelType.embeddings) {
            return new ChatLunaEmbeddings({
                client: this._requester,
                model,
                maxRetries: this._config.maxRetries
            })
        }

        return new ChatLunaChatModel({
            modelInfo: info,
            requester: this._requester,
            model: model.toLocaleLowerCase(),
            modelMaxContextSize: info.maxTokens,
            maxTokenLimit: this._config.maxTokens,
            frequencyPenalty: this._config.frequencyPenalty,
            presencePenalty: this._config.presencePenalty,
            timeout: this._config.timeout,
            temperature: this._config.temperature,
            maxRetries: this._config.maxRetries,
            llmType: 'zhipu'
        })
    }
}
