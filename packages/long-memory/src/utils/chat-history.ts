import { Context } from 'koishi'
import { Config, logger } from '..'
import { parseRawModelName } from 'koishi-plugin-chatluna/llm-core/utils/count_tokens'
import { BaseMessage } from '@langchain/core/messages'
import { ChatInterface } from 'koishi-plugin-chatluna/llm-core/chat/app'
import { EnhancedMemory } from '../types'
import { ChatLunaChatModel } from 'koishi-plugin-chatluna/llm-core/platform/model'
import { parseEnhancedMemories, parseResultContent } from './parse'

export async function generateNewQuestion(
    ctx: Context,
    config: Config,
    chatHistory: string,
    question: string
): Promise<string> {
    const [platform, modelName] = parseRawModelName(
        config.longMemoryExtractModel
    )

    const model = await ctx.chatluna.createChatModel(platform, modelName)

    const prompt = `
Given the following conversation history and the user's question, generate a new search query that will help retrieve relevant information from a long-term memory database. The search query should be concise and focused on the key information needs.

If you think the user's question is a casual greeting, a simple question that doesn't need memory retrieval, or is not related to any previous context, just respond with "[skip]".

Conversation History:
${chatHistory}

User Question: ${question}

New Search Query:
`

    const result = await model.invoke(prompt)

    return result.content as string
}

export async function selectChatHistory(
    messages: BaseMessage[],
    currentMessageId?: string,
    count: number = 10
): Promise<string> {
    if (!messages || messages.length === 0) {
        return ''
    }

    // 找到当前消息的索引
    let currentIndex = messages.length - 1
    if (currentMessageId) {
        const index = messages.findIndex((m) => m.id === currentMessageId)
        if (index !== -1) {
            currentIndex = index
        }
    }

    // 选择最近的count条消息
    const startIndex = Math.max(0, currentIndex - count)
    const selectedMessages = messages.slice(startIndex, currentIndex + 1)

    // 格式化聊天历史
    return selectedMessages
        .map((m) => {
            const role = m.getType() === 'human' ? 'User' : 'Assistant'
            const content =
                typeof m.content === 'string'
                    ? m.content
                    : JSON.stringify(m.content)
            return `${role}: ${content}`
        })
        .join('\n')
}

// 从聊天历史中提取记忆
export async function extractMemoriesFromChat(
    ctx: Context,
    config: Config,
    chatInterface: ChatInterface,
    chatHistory: string
): Promise<EnhancedMemory[]> {
    const preset = await chatInterface.preset
    const input = (
        preset.config?.longMemoryExtractPrompt ?? ENHANCED_MEMORY_PROMPT
    ).replaceAll('{user_input}', chatHistory)

    const [platform, modelName] = parseRawModelName(
        config.longMemoryExtractModel
    )

    const model = (await ctx.chatluna.createChatModel(
        platform,
        modelName
    )) as ChatLunaChatModel

    const extractMemory = async () => {
        const result = await model.invoke(input)

        try {
            // 尝试解析为增强记忆数组
            const enhancedMemories = parseEnhancedMemories(
                result.content as string
            )
            if (enhancedMemories.length > 0) {
                return enhancedMemories
            }
        } catch (e) {
            logger?.debug(`Failed to parse enhanced memories: ${e}`)
        }

        // 回退到普通记忆解析
        return parseResultContent(result.content as string)
    }

    let memories: EnhancedMemory[] = []

    for (let i = 0; i < 2; i++) {
        try {
            memories = await extractMemory()
            if (memories && memories.length > 0) {
                break
            }
        } catch (e) {
            logger?.error(e)
            logger?.warn(`Error extracting long memory of ${i} times`)
        }
    }

    if (!memories || memories.length === 0) {
        return []
    }

    logger?.debug(`Long memory extract: ${JSON.stringify(memories)}`)

    return memories
}

const ENHANCED_MEMORY_PROMPT = `You are now a Memory Extraction expert.

Your task is to extract memory content related to the conversation based on the given context and the specifications below:


Extract key memories from this chat as a JSON array of structured memory objects:
{user_input}

Guidelines:
1. Focus on extracting factual information, preferences, and important details about the user.
2. Capture information that would be valuable to remember for future conversations.
3. Each memory should be a complete, standalone sentence that makes sense without additional context.
4. Avoid creating memories about the current conversation flow or meta-discussions.
5. Do not include greetings, pleasantries, or other conversation fillers.
6. The memories output language should be same as the user input language.
7. If the context don't include any relevant information, return an empty array: []
8. Categorize each memory into one of these types:
   - "factual": Objective information and facts
   - "preference": User likes, dislikes, and preferences
   - "personal": Personal details about the user
   - "contextual": Information relevant to the current context
   - "temporal": Time-sensitive information
   - "task": Task or goal-related information
   - "skill": User's skills or abilities
   - "interest": User's interests or hobbies
   - "habit": User's habits or routines
   - "event": Event-related information
   - "location": Location-related information
   - "relationship": Information about user's relationships

9. Assign an importance score (1-10) to each memory:
   - 10: Critical information that must be remembered
   - 7-9: Very important information
   - 4-6: Moderately important information
   - 1-3: Less important details

Format your response as a valid JSON array of objects with this structure:
[
  {
    "content": "The user prefers dark chocolate over milk chocolate.",
    "type": "preference",
    "importance": 6
  },
  {
    "content": "The user has a meeting scheduled on March 15, 2025.",
    "type": "temporal",
    "importance": 8
  }
]

The memories output language should be same as the user input language!!!
The memories output language should be same as the user input language!!!
If no meaningful memories can be extracted, return an empty array: []

Output:`
