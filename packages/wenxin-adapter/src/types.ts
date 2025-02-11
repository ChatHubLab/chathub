export type WenxinMessageRole =
    | 'assistant'
    | 'user'
    | 'system'
    | 'function'
    | 'tool'

export interface WenxinMessage {
    role: WenxinMessageRole
    content?: string

    name?: string
    tool_calls?: ChatCompletionRequestMessageToolCall[]
    tool_call_id?: string
}

/**
 * Interface representing the usage of tokens in a chat completion.
 */
export interface TokenUsage {
    completionTokens?: number
    promptTokens?: number
    totalTokens?: number
}

/**
 * Interface representing a request for a chat completion.
 */
export interface ChatCompletionRequest {
    messages: WenxinMessage[]
    stream?: boolean
    user_id?: string
    functions?: ChatCompletionFunction[]
    temperature?: number
    top_p?: number
    penalty_score?: number
    system?: string
}

export interface ChatCompletionFunction {
    name: string
    description?: string
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    parameters?: { [key: string]: any }
}

/**
 * Interface representing a response from a chat completion.
 */
export interface ChatCompletionResponse {
    id: string
    object: string
    created: number
    result: string
    need_clear_history: boolean
    usage: TokenUsage
    function_call?: {
        name: string
        thoughts?: string
        arguments: string
    }
}

/**
 *
 * @export
 * @interface CreateEmbeddingResponse
 */
export interface CreateEmbeddingResponse {
    /**
     *
     * @type {string}
     * @memberof CreateEmbeddingResponse
     */
    object: string
    /**
     *
     * @type {string}
     * @memberof CreateEmbeddingResponse
     */
    model: string
    /**
     *
     * @type {Array<CreateEmbeddingResponseDataInner>}
     * @memberof CreateEmbeddingResponse
     */
    data: CreateEmbeddingResponseDataInner[]
    /**
     *
     * @type {CreateEmbeddingResponseUsage}
     * @memberof CreateEmbeddingResponse
     */
    usage: CreateEmbeddingResponseUsage
}

export interface CreateEmbeddingRequest {
    model: string
    input: string | string[]
}

/**
 *
 * @export
 * @interface CreateEmbeddingResponseDataInner
 */
export interface CreateEmbeddingResponseDataInner {
    /**
     *
     * @type {number}
     * @memberof CreateEmbeddingResponseDataInner
     */
    index: number
    /**
     *
     * @type {string}
     * @memberof CreateEmbeddingResponseDataInner
     */
    object: string
    /**
     *
     * @type {Array<number>}
     * @memberof CreateEmbeddingResponseDataInner
     */
    embedding: number[]
}
/**
 *
 * @export
 * @interface CreateEmbeddingResponseUsage
 */
export interface CreateEmbeddingResponseUsage {
    /**
     *
     * @type {number}
     * @memberof CreateEmbeddingResponseUsage
     */
    prompt_tokens: number
    /**
     *
     * @type {number}
     * @memberof CreateEmbeddingResponseUsage
     */
    total_tokens: number
}

export interface ChatCompletionRequestMessageToolCall {
    id: string
    type: 'function'
    function: {
        name: string
        arguments: string
    }
}
