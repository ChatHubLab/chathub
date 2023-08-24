import { Tiktoken } from 'js-tiktoken';
import { BaseChatModel, BaseChatModelCallOptions } from 'langchain/chat_models/base';
import { BaseLanguageModelCallOptions } from 'langchain/dist/base_language';
import { EmbeddingsRequestParams, EmbeddingsRequester, ModelRequestParams, ModelRequester } from './api';
import { CallbackManagerForLLMRun } from 'langchain/callbacks';
import { AIMessage, AIMessageChunk, BaseMessage, ChatGeneration, ChatGenerationChunk, ChatResult } from 'langchain/schema';
import { encodingForModel } from '../utils/tiktoken';
import { getModelContextSize, getModelNameForTiktoken } from '../utils/count_tokens';
import { createLogger } from '../utils/logger';
import { StructuredTool } from 'langchain/tools';
import { Embeddings, EmbeddingsParams } from 'langchain/embeddings/base';
import { chunkArray } from '../utils/chunk';

const logger = createLogger("@dingyi222666/chathub/llm-core/model/base");

export interface ChatHubModelCallOptions extends BaseChatModelCallOptions {

    model: string

    /** Sampling temperature to use */
    temperature?: number;

    /**
     * Maximum number of tokens to generate in the completion. -1 returns as many
     * tokens as possible given the prompt and the model's maximum context size.
     */
    maxTokens?: number;

    /** Total probability mass of tokens to consider at each step */
    topP?: number;

    /** Penalizes repeated tokens according to frequency */
    frequencyPenalty?: number;

    /** Penalizes repeated tokens */
    presencePenalty?: number;

    /** Number of completions to generate for each prompt */
    n?: number;

    /** Dictionary used to adjust the probability of specific tokens being generated */
    logitBias?: Record<string, number>;

    id?:string;

    stream: boolean

    tools?: StructuredTool[];
}

export interface ChatHubModelInput extends ChatHubModelCallOptions {
    llmType?: string


    requester: ModelRequester
}



export class ChatHubChatModel extends BaseChatModel<ChatHubModelCallOptions> {

    protected __encoding: Tiktoken

    private _requester: ModelRequester
    private _modelName: string


    lc_serializable = false;

    constructor(private _options: ChatHubModelInput) {
        super(_options)
        this._requester = _options.requester
        this._modelName = _options.model
    }

    get callKeys(): (keyof ChatHubModelCallOptions)[] {
        return [
            ...(super.callKeys as (keyof ChatHubModelCallOptions)[]),
        ];
    }

    /**
     * Get the parameters used to invoke the model
     */
    invocationParams(
        options?: this["ParsedCallOptions"]
    ): ChatHubModelCallOptions {
        const maxTokens = options?.maxTokens ?? this._options.maxTokens;
        return {
            model: options.model ?? this._options.model,
            temperature: options.temperature ?? this._options.temperature,
            topP: options.topP ?? this._options.topP,
            frequencyPenalty: options.frequencyPenalty ?? this._options.frequencyPenalty,
            presencePenalty: options.presencePenalty ?? this._options.presencePenalty,
            n: options.n ?? this._options.n,
            logitBias: options.logitBias ?? this._options.logitBias,
            maxTokens: maxTokens === -1 ? undefined : maxTokens,
            stop: options?.stop ?? this._options.stop,
            stream: options.stream ?? this._options.stream,
        };
    }

    async *_streamResponseChunks(
        messages: BaseMessage[],
        options: this["ParsedCallOptions"],
        runManager?: CallbackManagerForLLMRun
    ): AsyncGenerator<ChatGenerationChunk> {
        const params = this.invocationParams(options);
        const stream = await this.createStreamWithRetry({
            ...params,
            input: messages
        })

        for await (const chunk of stream) {
            yield chunk
            // eslint-disable-next-line no-void
            void runManager?.handleLLMNewToken(chunk.text ?? "");
        }
    }

    async _generate(messages: BaseMessage[], options: this['ParsedCallOptions'], runManager?: CallbackManagerForLLMRun): Promise<ChatResult> {
        const params = this.invocationParams(options);
        let response: ChatGeneration
        if (params.stream) {

            const stream = this._streamResponseChunks(
                messages,
                options,
                runManager
            );
            for await (const chunk of stream) {
                response = chunk
            }
        } else {
            response = await this.completionWithRetry({
                ...params,
                input: messages
            });
        }

        return {
            generations: [response]
        }
    }

    /**
     ** Creates a streaming request with retry.
     * @param request The parameters for creating a completion.
     ** @returns A streaming request.
     */
    private async createStreamWithRetry(
        params: ModelRequestParams
    ) {
        const makeCompletionRequest = async () =>
            this._requester.completionStream(params);
        return this.caller.call(makeCompletionRequest);
    }

    /** @ignore */
    private async completionWithRetry(
        params: ModelRequestParams
    ) {
        const makeCompletionRequest = async () =>
            this._requester.completion(params);
        return this.caller.call(makeCompletionRequest);
    }

    async clearContext(): Promise<void> {
        await this._requester.dispose()

        this.caller
            .call(this._requester.init.bind(this._requester))
    }


    getModelMaxContextSize() {
        const modelName = this._modelName ?? "gpt2"
        return getModelContextSize(modelName)
    }


    async getNumTokens(text: string) {
        // fallback to approximate calculation if tiktoken is not available
        let numTokens = Math.ceil(text.length / 4);

        if (!this.__encoding) {
            try {
                this.__encoding = await encodingForModel(
                    "modelName" in this
                        ? getModelNameForTiktoken(this.modelName as string)
                        : "gpt2"
                );
            } catch (error) {
                logger.warn(
                    "Failed to calculate number of tokens, falling back to approximate count",
                    error
                );
            }
        }


        if (this.__encoding) {
            numTokens = this.__encoding.encode(text).length;
        }
        return numTokens;
    }

    _llmType(): string {
        return this._options.llmType ?? "openai"
    }

    _modelType(): string {
        return "base_chat_model"
    }


    /** @ignore */
    _combineLLMOutput(...llmOutputs: any[]): any {
    }

}


export interface ChatHubEmbeddingsParams extends EmbeddingsParams {

    /**
     * Timeout to use when making requests.
     */
    timeout?: number;

    /**
     * The maximum number of documents to embed in a single request. This is
     * limited by the OpenAI API to a maximum of 2048.
     */
    batchSize?: number;

    /**
     * Whether to strip new lines from the input text. This is recommended by
     * OpenAI, but may not be suitable for all use cases.
     */
    stripNewLines?: boolean;

    maxRetries?: number

    client?: EmbeddingsRequester

    model?: string
}


export class ChatHubEmbeddings
    extends Embeddings {
    modelName = "text-embedding-ada-002";

    batchSize = 512;

    stripNewLines = true;

    timeout?: number;

    private _client: EmbeddingsRequester;

    constructor(
        fields?: ChatHubEmbeddingsParams,
    ) {

        super(fields);

        this.batchSize = fields?.batchSize ?? this.batchSize;
        this.stripNewLines = fields?.stripNewLines ?? this.stripNewLines;
        this.timeout = fields?.timeout ?? 1000 * 60
        this.modelName = fields?.model ?? this.modelName

        this._client = fields?.client
    }

    async embedDocuments(texts: string[]): Promise<number[][]> {
        const subPrompts = chunkArray(
            this.stripNewLines ? texts.map((t) => t.replaceAll("\n", " ")) : texts,
            this.batchSize
        );

        const embeddings: number[][] = [];

        for (let i = 0; i < subPrompts.length; i += 1) {
            const input = subPrompts[i];
            const { data } = await this._embeddingWithRetry({
                model: this.modelName,
                input,
            });
            for (let j = 0; j < input.length; j += 1) {
                embeddings.push(data[j] as number[]);
            }
        }

        return embeddings;
    }

    async embedQuery(text: string): Promise<number[]> {
        const { data } = await this._embeddingWithRetry({
            model: this.modelName,
            input: this.stripNewLines ? text.replaceAll("\n", " ") : text,
        });
        return data as number[];
    }

    private _embeddingWithRetry(request: EmbeddingsRequestParams) {
        request.timeout = this.timeout
        return this.caller.call(
            async (
                request: EmbeddingsRequestParams,
            ) => new Promise<{ data: number[] | number[][] }>(
                async (resolve, reject) => {

                    const timeout = setTimeout(
                        () => {
                            reject(Error(`timeout when calling ${this.modelName} embeddings`))
                        }, this.timeout ?? 1000 * 30)


                    const data = await this._client.embeddings(request)

                    clearTimeout(timeout)

                    if (data) {
                        resolve({
                            data: data
                        })
                        return
                    }

                    reject(Error(`error when calling ${this.modelName} embeddings, Result: ` + JSON.stringify(data)))
                }),
            request,
        );
    }
}