import { BaseMessage } from '@langchain/core/messages'
import type { PostHandler } from '../../utils/types'

export interface RawPreset {
    keywords: string[]
    prompts: {
        role: 'user' | 'system' | 'assistant'
        type?: 'personality' | 'description' | 'first_message' | 'scenario'
        content: string
    }[]
    format_user_prompt?: string
    world_lores?: (
        | {
              scanDepth?: number
              tokenLimit?: number
              recursiveScan?: boolean
              maxRecursionDepth?: number
              insertPosition?:
                  | 'before_char_defs'
                  | 'after_char_defs'
                  | 'before_scenario'
                  | 'after_scenario'
                  | 'before_example_messages'
                  | 'after_example_messages'
          }
        | {
              keywords: string | (string | RegExp)[]
              content: string
              insertPosition?:
                  | 'before_char_defs'
                  | 'after_char_defs'
                  | 'before_scenario'
                  | 'after_scenario'
                  | 'before_example_messages'
                  | 'after_example_messages'
              scanDepth?: number
              recursiveScan?: boolean
              maxRecursionDepth?: number
              matchWholeWord?: boolean
              constant?: boolean
              caseSensitive?: boolean
              enabled?: boolean
              order?: number
          }
    )[]
    version?: string
    authors_note?: AuthorsNote
    knowledge?: KnowledgeConfig
    config?: {
        longMemoryPrompt?: string
        loreBooksPrompt?: string
        longMemoryExtractPrompt?: string
        postHandler?: PostHandler
    }
}

export interface RoleBook {
    keywords: (string | RegExp)[]
    content: string
    scanDepth?: number
    recursiveScan?: boolean
    maxRecursionDepth?: number
    matchWholeWord?: boolean
    caseSensitive?: boolean
    enabled?: boolean
    constant?: boolean
    order?: number
    insertPosition?:
        | 'before_char_defs'
        | 'after_char_defs'
        | 'before_example_messages'
        | 'after_example_messages'
}

export type RoleBookConfig = Omit<PresetTemplate['loreBooks'], 'items'>

export interface PresetTemplate {
    version?: string
    triggerKeyword: string[]
    rawText: string
    messages: BaseMessage[]
    formatUserPromptString?: string
    path?: string
    loreBooks?: {
        scanDepth?: number
        items: RoleBook[]
        tokenLimit?: number
        recursiveScan?: boolean
        maxRecursionDepth?: number
        insertPosition?:
            | 'before_char_defs'
            | 'after_char_defs'
            | 'before_example_messages'
            | 'after_example_messages'
    }
    authorsNote?: AuthorsNote
    knowledge?: KnowledgeConfig
    config: {
        longMemoryPrompt?: string
        loreBooksPrompt?: string
        longMemoryExtractPrompt?: string
        postHandler?: PostHandler
    }
}

export interface KnowledgeConfig {
    knowledge: string[] | string
    prompt?: string
}

export interface AuthorsNote {
    content: string
    insertPosition?: 'after_char_defs' | 'in_chat'
    insertDepth?: number
    insertFrequency?: number
}

export function isRoleBook(obj: unknown): obj is RoleBook {
    return (
        typeof obj === 'object' &&
        obj !== null &&
        'keywords' in obj &&
        'content' in obj
    )
}

export function isRoleBookConfig(obj: unknown): obj is RoleBookConfig {
    return !isRoleBook(obj) && typeof obj === 'object' && obj !== null
}
