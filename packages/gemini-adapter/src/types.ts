export interface ChatCompletionResponseMessage {
    role: string
    parts?: ChatPart[]
}

export type ChatPart =
    | ChatMessagePart
    | ChatUploadDataPart
    | ChatFunctionCallingPart
    | ChatFunctionResponsePart

export type ChatMessagePart = {
    text: string
    thought?: boolean
}

export type ChatUploadDataPart = {
    inline_data: {
        mime_type: string
        data?: string
    }
}

export type ChatFunctionCallingPart = {
    functionCall: {
        name: string
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        args?: any
    }
}

export type ChatFunctionResponsePart = {
    functionResponse: {
        name: string
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        response: any
    }
}

export interface ChatResponse {
    candidates: {
        content: ChatCompletionResponseMessage
        finishReason: string
        index: number
        safetyRatings: {
            category: string
            probability: string
        }[]
    }[]
    promptFeedback: {
        safetyRatings: {
            category: string
            probability: string
        }[]
    }
}

export interface ChatCompletionFunction {
    name: string
    description?: string
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    parameters?: { [key: string]: any }
}

export interface ChatCompletionMessageFunctionCall {
    name: string
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    args?: any
}

export interface CreateEmbeddingResponse {
    embeddings: {
        values: number[]
    }[]
}

export type ChatCompletionResponseMessageRoleEnum =
    | 'system'
    | 'model'
    | 'user'
    | 'function'
