import { ClientConfig } from 'koishi-plugin-chatluna/llm-core/platform/config'

export interface DifyClientConfig extends ClientConfig {
    additionalModel: Map<
        string,
        { apiKey: string; workflowName: string; workflowType: string }
    >
}

export interface AssistantStreamResponse {
    event: string
    message_id?: string
    conversation_id: string
    answer?: string
    created_at?: number
    task_id?: string
    audio?: string
}
