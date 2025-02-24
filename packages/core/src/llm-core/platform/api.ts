import { BaseMessage } from '@langchain/core/messages'
import { StructuredTool } from '@langchain/core/tools'
import { ChatGeneration, ChatGenerationChunk } from '@langchain/core/outputs'

export interface BaseRequestParams {
    /**
     * Timeout to use when making requests to OpenAI.
     */
    timeout?: number
    /**
     ** The signal to use for cancellation.
     **/
    signal?: AbortSignal

    /** Model name to use */
    model?: string
}

export interface ModelRequestParams extends BaseRequestParams {
    /** Sampling temperature to use */
    temperature?: number

    /**
     * Maximum number of tokens to generate in the completion. -1 returns as many
     * tokens as possible given the prompt and the model's maximum context size.
     */
    maxTokens?: number

    /** Total probability mass of tokens to consider at each step */
    topP?: number

    /** Penalizes repeated tokens according to frequency */
    frequencyPenalty?: number

    /** Penalizes repeated tokens */
    presencePenalty?: number

    /** Number of completions to generate for each prompt */
    n?: number

    /** Dictionary used to adjust the probability of specific tokens being generated */
    logitBias?: Record<string, number>

    /** Unique string identifier representing your end-user, which can help OpenAI to monitor and detect abuse. */
    user?: string

    /** List of stop words to use when generating */
    stop?: string[] | string

    /**
     * Input messages to use for model completion.
     */
    input: BaseMessage[]

    id?: string

    tools?: StructuredTool[]
}

export interface EmbeddingsRequestParams extends BaseRequestParams {
    input: string | string[]
}

export interface BaseRequester {
    init(): Promise<void>

    dispose(): Promise<void>
}

export abstract class ModelRequester implements BaseRequester {
    async completion(params: ModelRequestParams): Promise<ChatGeneration> {
        const stream = this.completionStream(params)

        // get final result
        let result: ChatGenerationChunk

        for await (const chunk of stream) {
            result = result != null ? result.concat(chunk) : chunk
        }

        return result
    }

    abstract completionStream(
        params: ModelRequestParams
    ): AsyncGenerator<ChatGenerationChunk>

    abstract init(): Promise<void>

    abstract dispose(model?: string, id?: string): Promise<void>
}

export interface EmbeddingsRequester {
    embeddings(params: EmbeddingsRequestParams): Promise<number[] | number[][]>
}
