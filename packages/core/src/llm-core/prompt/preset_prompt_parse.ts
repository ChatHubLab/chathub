import {
    AIMessage,
    BaseMessage,
    BaseMessageFields,
    HumanMessage,
    SystemMessage
} from '@langchain/core/messages'
import { load } from 'js-yaml'
import { logger } from 'koishi-plugin-chatluna'
import {
    isRoleBook,
    isRoleBookConfig,
    PresetTemplate,
    RawPreset,
    RoleBookConfig
} from './type'

export function loadPreset(rawText: string): PresetTemplate {
    try {
        return loadYamlPreset(rawText)
    } catch {
        return loadTxtPreset(rawText)
    }
}

function createMessage(
    role: string,
    content: string,
    type?: string
): BaseMessage {
    if (content == null) {
        throw new Error('Content is required')
    }

    const fields: BaseMessageFields = {
        content: content.trim(),
        additional_kwargs: { type }
    }

    switch (role) {
        case 'assistant':
        case 'ai':
        case 'model':
            return new AIMessage(fields)
        case 'user':
        case 'human':
            return new HumanMessage(fields)
        case 'system':
            return new SystemMessage(fields)
        default:
            throw new Error(`Unknown role: ${role}`)
    }
}

function loadYamlPreset(rawText: string): PresetTemplate {
    const rawJson = load(rawText) as RawPreset

    let loreBooks: PresetTemplate['loreBooks'] | undefined = {
        items: []
    }

    let authorsNote: PresetTemplate['authorsNote'] | undefined

    if (rawJson.world_lores) {
        const config = rawJson.world_lores.find(
            isRoleBookConfig
        ) as RoleBookConfig

        const items = rawJson.world_lores.filter(isRoleBook).map((item) => ({
            ...item,
            keywords: Array.isArray(item.keywords)
                ? item.keywords
                : [item.keywords]
        }))

        loreBooks = {
            ...config,
            items
        }
    } else {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        loreBooks = undefined
    }

    if (rawJson.authors_note || rawJson['author_notes']) {
        authorsNote = rawJson.authors_note || rawJson['author_notes']
        authorsNote.insertFrequency = authorsNote.insertFrequency ?? 1
        authorsNote.insertPosition = authorsNote.insertPosition ?? 'in_chat'
        authorsNote.insertDepth = authorsNote.insertDepth ?? 0
    }

    return {
        triggerKeyword: rawJson.keywords,
        rawText,
        messages: rawJson.prompts.map((message) =>
            createMessage(message.role, message.content, message.type)
        ),
        formatUserPromptString: rawJson.format_user_prompt,
        loreBooks,
        authorsNote,
        knowledge: rawJson?.knowledge,
        version: rawJson?.version,
        config: rawJson.config ?? {}
    }
}

function loadTxtPreset(rawText: string): PresetTemplate {
    const triggerKeyword: string[] = []
    const messages: BaseMessage[] = []
    let formatUserPromptString = '{prompt}'

    logger?.warn(
        'TXT Preset is deprecated and will be removed in 1.0. ' +
            'Please migrate to YAML preset format. ' +
            'For more migrate information, visit: https://chatluna.chat/guide/preset-system/introduction.html'
    )

    const chunks = rawText
        .replace(/#.*\r?\n/g, '')
        .replace(/\r\n/g, '\n')
        .split(/\n\n/)

    for (const chunk of chunks) {
        const match = chunk.match(/^\s*([a-zA-Z_]+)\s*:\s*(.*)$/s)
        if (!match) continue

        const [, role, content] = match

        if (role === 'keyword') {
            triggerKeyword.push(...content.split(',').map((k) => k.trim()))
        } else if (role === 'format_user_prompt') {
            formatUserPromptString = content.trim()
        } else {
            messages.push(createMessage(role, content))
        }
    }

    if (triggerKeyword.length === 0) throw new Error('No trigger keyword found')
    if (messages.length === 0) throw new Error('No preset messages found')

    return {
        rawText,
        triggerKeyword,
        messages,
        formatUserPromptString,
        config: {}
    }
}

export * from './format'
export * from './type'
