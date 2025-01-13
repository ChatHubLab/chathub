export interface ClaudeRequest {
    model: string
    max_tokens: number
    temperature?: number
    top_p?: number
    top_k?: number
    stream?: boolean
    stop_sequences?: string[]
    messages: ClaudeMessage[]
    tools?: CluadeTool[]
}

export interface CluadeTool {
    name: string
    description: string
    input_schema: object
}

export interface ClaudeMessage {
    role: string
    content?:
        | string
        | (
              | {
                    type: 'text'
                    text: string
                }
              | {
                    type: 'image'
                    source: {
                        type: string
                        media_type: string
                        data: string
                    }
                }
              | {
                    type: 'tool_use'
                    id: string
                    name: string
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    input: Record<string, any>
                }
              | {
                    type: 'tool_result'
                    tool_use_id: string
                    content: string
                }
          )[]
}

type ClaudeDeltaResponseType =
    | 'content_block_delta'
    | 'content_block_start'
    | 'message_start'

type ResponsePayload<T extends ClaudeDeltaResponseType> = {
    content_block_delta: {
        type: string
        text: string
        partial_json?: string
    }
    content_block_start: {
        type: string
        id: string
        name: string
        input: object
        text?: string
    }
    message_start: {
        id: string
        type: string
        role: string
        model: string
        stop_sequence?: string
    }
}[T]

type PayloadMapping = {
    content_block_delta: { delta: ResponsePayload<'content_block_delta'> }
    content_block_start: {
        content_block: ResponsePayload<'content_block_start'>
    }
    message_start: { message: ResponsePayload<'message_start'> }
}

export type ClaudeDeltaResponse = {
    [T in ClaudeDeltaResponseType]: {
        type: T
        index: number
    } & PayloadMapping[T]
}[ClaudeDeltaResponseType]

export type ChatCompletionResponseMessageRoleEnum =
    | 'system'
    | 'assistant'
    | 'user'
    | 'tool'
