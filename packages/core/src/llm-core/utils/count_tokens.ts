import { MessageType } from '@langchain/core/messages'
import { type TiktokenModel } from 'js-tiktoken/lite'
import { encodingForModel } from './tiktoken'
import {
    ChatLunaError,
    ChatLunaErrorCode
} from 'koishi-plugin-chatluna/utils/error'

// https://www.npmjs.com/package/js-tiktoken

const tiktokenModels: string[] = [
    'davinci-002',
    'babbage-002',
    'text-davinci-003',
    'text-davinci-002',
    'text-davinci-001',
    'text-curie-001',
    'text-babbage-001',
    'text-ada-001',
    'davinci',
    'curie',
    'babbage',
    'ada',
    'code-davinci-002',
    'code-davinci-001',
    'code-cushman-002',
    'code-cushman-001',
    'davinci-codex',
    'cushman-codex',
    'text-davinci-edit-001',
    'code-davinci-edit-001',
    'text-embedding-ada-002',
    'text-embedding-3-small',
    'text-embedding-3-large',
    'text-similarity-davinci-001',
    'text-similarity-curie-001',
    'text-similarity-babbage-001',
    'text-similarity-ada-001',
    'text-search-davinci-doc-001',
    'text-search-curie-doc-001',
    'text-search-babbage-doc-001',
    'text-search-ada-doc-001',
    'code-search-babbage-code-001',
    'code-search-ada-code-001',
    'gpt2',
    'gpt-3.5-turbo',
    'gpt-35-turbo',
    'gpt-3.5-turbo-0301',
    'gpt-3.5-turbo-0613',
    'gpt-3.5-turbo-1106',
    'gpt-3.5-turbo-0125',
    'gpt-3.5-turbo-16k',
    'gpt-3.5-turbo-16k-0613',
    'gpt-3.5-turbo-instruct',
    'gpt-3.5-turbo-instruct-0914',
    'gpt-4',
    'gpt-4-0314',
    'gpt-4-0613',
    'gpt-4-32k',
    'gpt-4-32k-0314',
    'gpt-4-32k-0613',
    'gpt-4-turbo',
    'gpt-4-turbo-2024-04-09',
    'gpt-4-turbo-preview',
    'gpt-4-1106-preview',
    'gpt-4-0125-preview',
    'gpt-4-vision-preview',
    'gpt-4o',
    'gpt-4o-2024-05-13',
    'gpt-4o-2024-08-06',
    'gpt-4o-mini-2024-07-18',
    'gpt-4o-mini',
    'o1-mini',
    'o1-preview',
    'o1-preview-2024-09-12',
    'o1-mini-2024-09-12',
    'chatgpt-4o-latest',
    'gpt-4o-realtime',
    'gpt-4o-realtime-preview-2024-10-01'
]
export const getModelNameForTiktoken = (modelName: string): TiktokenModel => {
    if (modelName.startsWith('gpt-3.5-turbo-16k')) {
        return 'gpt-3.5-turbo-16k'
    }

    if (modelName.startsWith('gpt-3.5-turbo-')) {
        return 'gpt-3.5-turbo'
    }

    if (modelName.startsWith('gpt-4-32k')) {
        return 'gpt-4-32k'
    }

    if (modelName.startsWith('gpt-4-')) {
        return 'gpt-4'
    }

    if (modelName.startsWith('gpt-4o')) {
        return 'gpt-4o'
    }

    if (modelName.startsWith('o1')) {
        return 'o1-mini'
    }

    if (tiktokenModels.includes(modelName)) {
        return modelName as TiktokenModel
    }

    // most model the context size is larger than gpt-3.5-turbo
    return 'o1-mini'
}

export const getEmbeddingContextSize = (modelName?: string): number => {
    switch (modelName) {
        case 'text-embedding-3-large':
        case 'text-embedding-3-small':
        case 'text-embedding-ada-002':
            return 8191
        default:
            return 2046
    }
}

interface CalculateMaxTokenProps {
    prompt: string
    modelName: TiktokenModel
}

export const calculateMaxTokens = async ({
    prompt,
    modelName
}: CalculateMaxTokenProps) => {
    // fallback to approximate calculation if tiktoken is not available
    let numTokens = Math.ceil(prompt.length / 4)

    try {
        numTokens = (await encodingForModel(modelName)).encode(prompt).length
    } catch (error) {}

    const maxTokens = getModelContextSize(modelName)
    return maxTokens - numTokens
}

export function messageTypeToOpenAIRole(type: MessageType): string {
    switch (type) {
        case 'system':
            return 'system'
        case 'ai':
            return 'assistant'
        case 'human':
            return 'user'
        case 'function':
            return 'function'
        case 'tool':
            return 'tool'
        default:
            throw new Error(`Unknown message type: ${type}`)
    }
}

export const getModelContextSize = (modelName: string): number => {
    switch (getModelNameForTiktoken(modelName)) {
        case 'gpt-4o':
        case 'gpt-4o-2024-05-13':
        case 'gpt-4-0125-preview':
        case 'o1-mini':
            return 128000
        case 'gpt-3.5-turbo-16k':
            return 16384
        case 'gpt-3.5-turbo':
            return 4096
        case 'gpt-4-32k':
            return 32768
        case 'gpt-4':
            return 8192
        case 'text-davinci-003':
            return 4097
        case 'text-curie-001':
            return 2048
        case 'text-babbage-001':
            return 2048
        case 'text-ada-001':
            return 2048
        case 'code-davinci-002':
            return 8000
        case 'code-cushman-001':
            return 2048
        default:
            return 4097
    }
}

export function parseRawModelName(modelName: string): [string, string] {
    if (modelName == null) {
        throw new ChatLunaError(ChatLunaErrorCode.MODEL_NOT_FOUND)
    }
    return modelName.split(/(?<=^[^\/]+)\//) as [string, string]
}
