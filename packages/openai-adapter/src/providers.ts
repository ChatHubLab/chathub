import OpenAIPlugin from '.';
import { ChatHubBaseChatModel, CreateParams, EmbeddingsProvider, ModelProvider } from '@dingyi222666/koishi-plugin-chathub/lib/llm-core/model/base'
import { Api } from './api';
import { OpenAIChatModel, OpenAIEmbeddings } from './models';
import { BaseChatModel } from 'langchain/chat_models/base';
import { Embeddings } from 'langchain/embeddings/base';
import { sleep } from 'koishi';


export class OpenAIModelProvider extends ModelProvider {

    private _models: string[] | null = null

    private _API: Api | null = null

    name = "openai"
    description?: string = "OpenAI model provider, provide gpt3.5/gpt4 model"

    constructor(private readonly config: OpenAIPlugin.Config) {
        super()
        this._API = new Api(config)
    }

    async listModels(): Promise<string[]> {
        if (this._models) {
            return this._models
        }

        this._models = (await this._API.listModels()).filter((id) => id.startsWith("gpt"))

        return this._models
    }

    async isSupported(modelName: string): Promise<boolean> {
        return (await this.listModels()).includes(modelName)
    }

    isSupportedChatMode(modelName: string, chatMode: string): Promise<boolean> {
        return this.isSupported(modelName)
    }

    async recommendModel(): Promise<string> {
        const models = await this.listModels()

        const gpt3With16Models = models.filter(x => x.startsWith("gpt-3.5-turbo-16k"))

        if (gpt3With16Models.length > 0) {
            return gpt3With16Models[0]
        }

        const gpt3Models = models.filter(x => x.startsWith("gpt-3.5-turbo"))

        if (gpt3Models.length > 0) {
            return gpt3Models[0]
        }

        return models[0]
    }


    async createModel(modelName: string, params: CreateParams): Promise<ChatHubBaseChatModel> {
        const hasModel = (await this.listModels()).includes(modelName)

        if (!hasModel) {
            throw new Error(`Can't find model ${modelName}`)
        }

        params.client = this._API
        return new OpenAIChatModel(modelName, this.config, params)
    }

    getExtraInfo(): Record<string, any> {
        return this.config
    }
}

export class OpenAIEmbeddingsProvider extends EmbeddingsProvider {

    private _API: Api | null = null


    private _models: string[] | null = null

    name = "openai"
    description?: string = "OpenAI embeddings provider, provide text-embedding-ada-002"

    constructor(private readonly config: OpenAIPlugin.Config) {
        super()
        this._API = new Api(config)
    }

    async createEmbeddings(modelName: string, params: CreateParams): Promise<Embeddings> {
        return new OpenAIEmbeddings(this.config, {
            client: this._API,
        })
    }

    async listEmbeddings(): Promise<string[]> {
        if (this._models) {
            return this._models
        }

        this._models = (await this._API.listModels()).filter((id) => id === "text-embedding-ada-002")

        return this._models
    }

    async isSupported(modelName: string): Promise<boolean> {
        return (await this.listEmbeddings()).includes(modelName)
    }


    getExtraInfo(): Record<string, any> {
        return this.config
    }
}