import { PlatformModelAndEmbeddingsClient } from 'koishi-plugin-chatluna/llm-core/platform/client'
import { ClientConfig } from 'koishi-plugin-chatluna/llm-core/platform/config'
import {
    ChatHubBaseEmbeddings,
    ChatLunaChatModel,
    ChatLunaEmbeddings
} from 'koishi-plugin-chatluna/llm-core/platform/model'
import {
    ModelInfo,
    ModelType
} from 'koishi-plugin-chatluna/llm-core/platform/types'
import { Context } from 'koishi'
import { Config } from '.'
import {
    ChatLunaError,
    ChatLunaErrorCode
} from 'koishi-plugin-chatluna/utils/error'
import { WenxinRequester } from './requester'
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'

export class WenxinClient extends PlatformModelAndEmbeddingsClient<ClientConfig> {
    platform = 'wenxin'

    private _requester: WenxinRequester

    private _models: Record<string, ModelInfo>

    constructor(
        ctx: Context,
        private _config: Config,
        clientConfig: ClientConfig,
        plugin: ChatLunaPlugin<ClientConfig, Config>
    ) {
        super(ctx, clientConfig)

        this._requester = new WenxinRequester(clientConfig, _config, plugin)
    }

    async init(): Promise<void> {
        await this.getModels()
    }

    async refreshModels(): Promise<ModelInfo[]> {
        const rawModels = [
            ['ernie-4.0-8k', 8000], // ERNIE-4.0-8K
            ['ernie-4.0-8k-preview', 8000], // ERNIE-4.0-8K-Preview
            ['ernie-4.0-8k-latest', 8000], // ERNIE-4.0-8K-Latest
            ['ernie-4.0-turbo-8k', 8000], // ERNIE-4.0-Turbo-8K
            ['ernie-4.0-turbo-8k-preview', 8000], // ERNIE-4.0-Turbo-8K-Preview
            ['ernie-4.0-turbo-8k-latest', 8000], // ERNIE-4.0-Turbo-8K-Latest
            ['ernie-4.0-turbo-128k', 128000], // ERNIE-4.0-Turbo-128K
            ['ernie-3.5-8k', 4096], // ERNIE-3.5-8K
            ['ernie-3.5-8k-preview', 4096], // ERNIE-3.5-8K-Preview
            ['ernie-3.5-128k', 128000], // ERNIE-3.5-128K
            ['ernie-speed-pro-128k', 128000], // ERNIE-Speed-Pro-128K
            ['ernie-speed-8k', 4096], // ERNIE-Speed-8K
            ['ernie-speed-128k', 128000], // ERNIE-Speed-128K
            ['ernie-character-8k', 8000], // ERNIE-Character-8K
            ['ernie-character-fiction-8k', 8000], // ERNIE-Character-Fiction-8K
            ['ernie-lite-8k', 8000], // ERNIE-Lite-8K
            ['ernie-lite-pro-128k', 128000], // ERNIE-Lite-Pro-128K
            ['ernie-tiny-8k', 8000], // ERNIE-Tiny-8K
            ['ernie-novel-8k', 8000], // ERNIE-Novel-8K
            ['deepseek-v3', 128000], // DeepSeek-V3 (未提供上下文大小)
            ['deepseek-r1', 128000], // DeepSeek-R1 (未提供上下文大小)
            ['deepseek-r1-distill-qwen-32b', 8000], // DeepSeek-R1-Distill-Qwen-32B (未提供上下文大小)
            ['deepseek-r1-distill-qwen-14b', 8000] // DeepSeek-R1-Distill-Qwen-14B (未提供上下文大小)
        ] as [string, number][]

        return rawModels
            .map(([model, maxTokens]) => {
                return {
                    name: model,
                    type: ModelType.llm,
                    functionCall: model.includes('ERNIE-3.5'),
                    supportMode: ['all'],
                    maxTokens
                }
            })
            .concat([
                {
                    name: 'embedding-v1',
                    type: ModelType.embeddings,
                    functionCall: false,
                    supportMode: ['all'],
                    maxTokens: 4000
                } satisfies ModelInfo
            ])
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
    ): ChatLunaChatModel | ChatHubBaseEmbeddings {
        const info = this._models[model]

        if (info == null) {
            throw new ChatLunaError(ChatLunaErrorCode.MODEL_NOT_FOUND)
        }

        if (info.type === ModelType.llm) {
            return new ChatLunaChatModel({
                modelInfo: info,
                requester: this._requester,
                model,
                modelMaxContextSize: info.maxTokens,
                maxTokenLimit: this._config.maxTokens,
                frequencyPenalty: this._config.frequencyPenalty,
                presencePenalty: this._config.presencePenalty,
                timeout: this._config.timeout,
                temperature: this._config.temperature,
                maxRetries: this._config.maxRetries,
                llmType: 'wenxin'
            })
        }

        return new ChatLunaEmbeddings({
            client: this._requester,
            maxRetries: this._config.maxRetries
        })
    }
}
