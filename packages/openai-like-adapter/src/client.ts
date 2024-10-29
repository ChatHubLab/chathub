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
import { OpenAIRequester } from './requester'
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'
import { getModelContextSize } from 'koishi-plugin-chatluna/llm-core/utils/count_tokens'

export class OpenAIClient extends PlatformModelAndEmbeddingsClient {
    platform = 'openai'

    private _requester: OpenAIRequester

    private _models: Record<string, ModelInfo>

    constructor(
        ctx: Context,
        private _config: Config,
        clientConfig: ClientConfig,
        plugin: ChatLunaPlugin
    ) {
        super(ctx, clientConfig)
        this.platform = _config.platform
        this._requester = new OpenAIRequester(clientConfig, _config, plugin)
    }

    async init(): Promise<void> {
        await this.getModels()
    }

    async refreshModels(): Promise<ModelInfo[]> {
        try {
            const rawModels = this._config.pullModels
                ? await this._requester.getModels()
                : []

            const additionalModels = this._config.additionalModels.map(
                ({ model, modelType, contextSize }) =>
                    ({
                        name: model,
                        type:
                            modelType === 'Embeddings 嵌入模型'
                                ? ModelType.embeddings
                                : ModelType.llm,
                        functionCall:
                            modelType === 'LLM 大语言模型（函数调用）',
                        maxTokens: contextSize ?? 4096,
                        supportMode: ['all']
                    }) as ModelInfo
            )

            const filteredModels = rawModels.filter(
                (model) =>
                    !['whisper', 'tts', 'dall-e', 'image', 'rerank'].some(
                        (keyword) => model.includes(keyword)
                    )
            )

            const formattedModels = filteredModels.map(
                (model) =>
                    ({
                        name: model,
                        type: model.includes('embed')
                            ? ModelType.embeddings
                            : ModelType.llm,
                        functionCall: true,
                        supportMode: ['all']
                    }) as ModelInfo
            )

            return formattedModels.concat(additionalModels)
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
                maxTokenLimit: this._config.maxTokens,
                modelMaxContextSize: this._getModelMaxContextSize(info),
                frequencyPenalty: this._config.frequencyPenalty,
                presencePenalty: this._config.presencePenalty,
                timeout: this._config.timeout,
                temperature: this._config.temperature,
                maxRetries: this._config.maxRetries,
                llmType: 'openai'
            })
        }

        return new ChatLunaEmbeddings({
            client: this._requester,
            model,
            maxRetries: this._config.maxRetries
        })
    }

    private _getModelMaxContextSize(info: ModelInfo): number {
        const maxTokens = info.maxTokens

        if (maxTokens != null) {
            return maxTokens
        }

        const modelName = info.name

        if (modelName.startsWith('gpt') || modelName.startsWith('o1')) {
            return getModelContextSize(modelName)
        }

        // compatible with Anthropic, Google, ...
        const modelMaxContextSizeTable: { [key: string]: number } = {
            claude: 2000000,
            'gemini-1.5-pro': 1048576,
            'gemini-1.5-flash': 2097152,
            'gemini-1.0-pro': 30720,
            deepseek: 128000,
            'llama3.1': 128000,
            'command-r-plus': 128000,
            'moonshot-v1-8k': 8192,
            'moonshot-v1-32k': 32000,
            'moonshot-v1-128k': 128000,
            Qwen2: 32000
        }

        for (const key in modelMaxContextSizeTable) {
            if (modelName.toLowerCase().includes(key)) {
                return modelMaxContextSizeTable[key]
            }
        }

        return 8192
    }
}
