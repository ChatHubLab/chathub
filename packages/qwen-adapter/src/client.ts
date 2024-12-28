import { Context } from 'koishi'
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
import {
    ChatLunaError,
    ChatLunaErrorCode
} from 'koishi-plugin-chatluna/utils/error'
import { Config } from '.'
import { QWenRequester } from './requester'
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'

export class QWenClient extends PlatformModelAndEmbeddingsClient<ClientConfig> {
    platform = 'qwen'

    private _requester: QWenRequester

    private _models: Record<string, ModelInfo>

    constructor(
        ctx: Context,
        private _config: Config,
        clientConfig: ClientConfig,
        plugin: ChatLunaPlugin
    ) {
        super(ctx, clientConfig)

        this._requester = new QWenRequester(clientConfig, _config, plugin)
    }

    async init(): Promise<void> {
        await this.getModels()
    }

    async refreshModels(): Promise<ModelInfo[]> {
        const rawModels: [string, number | undefined][] = [
            ['qwen-turbo', 1000000],
            ['qwen-long', 1000000],
            ['qwen-plus', 131072],
            ['qwen-max', 30720],
            ['qwen-max-latest', 30720],
            ['qwen-plus-latest', 1280000],
            ['qwen-turbo-latest', 1280000],
            ['qwen-vl-max', 32000],
            ['qwen-vl-max-latest', 32000],
            ['qwen-vl-plus', 8000],
            ['qwen-vl-plus-latest', 1280000],
            ['qwen-vl-ocr', 34096],
            ['qwen-vl-ocr-latest', 34096],
            ['qwq-32b-preview', 30720],
            ['qvq-72b-preview', 30720],
            ['qwen-math-plus', 4000],
            ['qwen-math-turbo', 4000],
            ['qwen2.5-72b-instruct', 131072],
            ['qwen2.5-32b-instruct', 129024],
            ['qwen2.5-14b-instruct', 8192],
            ['qwen2.5-7b-instruct', 32768],
            ['qwen2.5-3b-instruct', 30720],
            ['qwen2.5-1.5b-instruct', 30720],
            ['qwen2.5-0.5b-instruct', 30720],
            ['text-embedding-v1', 2048],
            ['text-embedding-v2', 2048],
            ['text-embedding-v3', 8192]
        ] as [string, number][]

        const additionalModels = this._config.additionalModels.map(
            ({ model, modelType: llmType, contextSize: token }) => {
                return {
                    name: model,
                    type: ModelType.llm,
                    functionCall: llmType === 'LLM 大语言模型（函数调用）',
                    maxTokens: token ?? 4096,
                    supportMode: ['all']
                } as ModelInfo
            }
        )

        return rawModels
            .map(([model, token]) => {
                return {
                    name: model,
                    type: model.includes('embedding')
                        ? ModelType.embeddings
                        : ModelType.llm,
                    maxTokens: token,
                    functionCall:
                        model.includes('qwen-plus') ||
                        model.includes('qwen-max') ||
                        model.includes('qwen-turbo'),
                    supportMode: ['all']
                } as ModelInfo
            })
            .concat(additionalModels)
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
                timeout: this._config.timeout,
                temperature: this._config.temperature,
                maxRetries: this._config.maxRetries,
                llmType: 'qwen'
            })
        }

        return new ChatLunaEmbeddings({
            client: this._requester,
            model: info.name,
            batchSize: 5,
            maxRetries: this._config.maxRetries
        })
    }
}
