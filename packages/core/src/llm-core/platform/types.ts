import { BufferMemory } from 'koishi-plugin-chatluna/llm-core/memory/langchain'
import { ChatHubBaseEmbeddings, ChatLunaChatModel } from './model'
import { ChatLunaLLMChainWrapper } from '../chain/base'
import { StructuredTool } from '@langchain/core/tools'
import { BaseMessage } from '@langchain/core/messages'
import { Dict, Session } from 'koishi'
import { PresetTemplate } from 'koishi-plugin-chatluna/llm-core/prompt'
import { ChatLunaSaveableVectorStore } from 'koishi-plugin-chatluna/llm-core/model/base'
export interface ChatLunaChainInfo {
    name: string
    description?: Dict<string>
    createFunction: (
        params: CreateChatLunaLLMChainParams
    ) => Promise<ChatLunaLLMChainWrapper>
}

export interface CreateToolParams {
    model: ChatLunaChatModel
    embeddings: ChatHubBaseEmbeddings
    conversationId?: string
    preset?: string
    userId?: string
}

export interface CreateVectorStoreParams {
    key?: string
    embeddings: ChatHubBaseEmbeddings
    //  topK?: number
}

export interface CreateChatLunaLLMChainParams {
    botName: string
    model: ChatLunaChatModel
    embeddings?: ChatHubBaseEmbeddings
    historyMemory: BufferMemory
    preset: () => Promise<PresetTemplate>
    supportChatChain?: boolean
    vectorStoreName?: string
}

export interface ChatLunaTool {
    createTool: (
        params: CreateToolParams,
        session?: Session
    ) => Promise<StructuredTool>
    selector: (history: BaseMessage[]) => boolean
    authorization?: (session: Session) => boolean
    alwaysRecreate?: boolean
}

export type CreateVectorStoreFunction = (
    params: CreateVectorStoreParams
) => Promise<ChatLunaSaveableVectorStore>

export interface PlatformClientName {
    default: never
}

export type PlatformClientNames = keyof PlatformClientName | string

export interface ModelInfo {
    name: string

    type: ModelType

    maxTokens?: number

    functionCall?: boolean

    supportMode?: string[]
}

export enum ModelType {
    all,
    llm,
    embeddings
}
