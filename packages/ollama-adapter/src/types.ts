export interface OllamaRequest {
    model: string
    options: {
        temperature: number
        top_k?: number
        top_p: number
        stop: string
    }
    keep_alive?: number
    messages: OllamaMessage[]
    stream: boolean
}

export interface OllamaDeltaResponse {
    model: string
    message: OllamaMessage
    done: boolean
}

export interface OllamaMessage {
    role: string
    content: string
    images?: string[]
}

export interface OllamaEmbedRequest {
    model: string
    input: string | string[]
}

export interface OllamaEmbedResponse {
    model: string
    embeddings: number[][]
}
