import { Context } from 'koishi'
import { VectorStore, VectorStoreRetriever } from '@langchain/core/vectorstores'
import { ChatInterface } from 'koishi-plugin-chatluna/llm-core/chat/app'
import { BaseMessage, HumanMessage } from '@langchain/core/messages'
import { ScoreThresholdRetriever } from 'koishi-plugin-chatluna/llm-core/retrievers'
import { parseRawModelName } from 'koishi-plugin-chatluna/llm-core/utils/count_tokens'
import { ChatLunaChatModel } from 'koishi-plugin-chatluna/llm-core/platform/model'
import { ChatLunaSaveableVectorStore } from 'koishi-plugin-chatluna/llm-core/model/base'
import { calculateSimilarity } from '../similarity'
import crypto from 'crypto'
import { Config, logger } from 'koishi-plugin-chatluna-long-memory'
import { Document } from '@langchain/core/documents'
import { getMessageContent } from 'koishi-plugin-chatluna/utils/string'

// 记忆类型枚举
enum MemoryType {
    FACTUAL = 'factual', // 事实性知识（长期有效）
    PREFERENCE = 'preference', // 用户偏好（长期有效）
    PERSONAL = 'personal', // 个人信息（长期有效）
    CONTEXTUAL = 'contextual', // 上下文相关（中期有效）
    TEMPORAL = 'temporal', // 时间相关（短期有效）
    TASK = 'task', // 任务相关（中期有效）
    SKILL = 'skill', // 技能相关（长期有效）
    INTEREST = 'interest', // 兴趣爱好（长期有效）
    HABIT = 'habit', // 习惯相关（长期有效）
    EVENT = 'event', // 事件相关（短期有效）
    LOCATION = 'location', // 位置相关（中期有效）
    RELATIONSHIP = 'relationship' // 关系相关（长期有效）
}

// 根据记忆类型和重要性计算过期时间
function calculateExpirationDate(type: MemoryType, importance: number): Date {
    const now = new Date()

    // 将重要性（1-10）映射到0-1的范围，用于计算过期时间
    const importanceFactor = Math.min(Math.max(importance, 1), 10) / 10

    switch (type) {
        case MemoryType.FACTUAL:
        case MemoryType.PREFERENCE:
        case MemoryType.PERSONAL:
        case MemoryType.SKILL:
        case MemoryType.INTEREST:
        case MemoryType.HABIT:
        case MemoryType.RELATIONSHIP: {
            // 长期记忆 - 1-12个月
            const longExpirationDate = new Date(now)

            // 如果重要性为10，则设置为永不过期（12个月）
            if (importance === 10) {
                longExpirationDate.setMonth(longExpirationDate.getMonth() + 12)
            } else {
                // 根据重要性调整过期时间，范围是1-12个月
                const monthsToAdd = 1 + importanceFactor * 11
                longExpirationDate.setMonth(
                    longExpirationDate.getMonth() + Math.floor(monthsToAdd)
                )
            }
            return longExpirationDate
        }

        case MemoryType.CONTEXTUAL:
        case MemoryType.TASK:
        case MemoryType.LOCATION: {
            // 中期记忆 - 1-3周
            const mediumExpirationDate = new Date(now)
            // 根据重要性调整过期时间
            const daysToAdd = 7 + importanceFactor * 14 // 1-3周（7-21天）
            mediumExpirationDate.setDate(
                mediumExpirationDate.getDate() + Math.floor(daysToAdd)
            )
            return mediumExpirationDate
        }

        case MemoryType.TEMPORAL:
        case MemoryType.EVENT: {
            // 短期记忆 - 12小时到2天
            const shortExpirationDate = new Date(now)
            // 根据重要性调整过期时间
            const hoursToAdd = 12 + importanceFactor * 36 // 12-48小时（0.5-2天）
            shortExpirationDate.setHours(
                shortExpirationDate.getHours() + Math.floor(hoursToAdd)
            )
            return shortExpirationDate
        }

        default: {
            // 默认一周
            const defaultExpirationDate = new Date(now)
            defaultExpirationDate.setDate(defaultExpirationDate.getDate() + 7)
            return defaultExpirationDate
        }
    }
}

// 记忆结构接口
interface EnhancedMemory {
    content: string // 记忆内容
    type: MemoryType // 记忆类型
    importance: number // 重要性 (1-10)
    expirationDate?: Date // 过期时间（可选）
    rawId?: string // 原始ID
}

// 将增强记忆转换为Document
function enhancedMemoryToDocument(memory: EnhancedMemory): Document {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const metadata: Record<string, any> = {
        type: memory.type,
        importance: memory.importance
    }

    if (memory.expirationDate) {
        metadata.expirationDate = memory.expirationDate.toISOString()
    }

    if (memory.rawId) {
        metadata.rawId = memory.rawId
    }

    return new Document({
        pageContent: memory.content,
        metadata
    })
}

// 将Document转换为增强记忆
function documentToEnhancedMemory(document: Document): EnhancedMemory {
    const metadata = document.metadata || {}

    const memory: EnhancedMemory = {
        content: document.pageContent,
        type: metadata.type || MemoryType.FACTUAL,
        importance: metadata.importance || 5
    }

    if (metadata.expirationDate) {
        memory.expirationDate = new Date(metadata.expirationDate)
    }

    if (metadata.rawId) {
        memory.rawId = metadata.rawId
    }

    return memory
}

// 检查记忆是否过期
function isMemoryExpired(memory: EnhancedMemory): boolean {
    if (!memory.expirationDate) return false
    return new Date() > memory.expirationDate
}

// 从消息中解析需要使用的层级名称
function resolveLayerNames(): string[] {
    return ['user', 'preset']
}

// Interface for memory retrieval layer
interface MemoryRetrievalLayer {
    // Retrieve memory based on the search content
    retrieveMemory(searchContent: string): Promise<EnhancedMemory[]>
    // Add new memories
    addMemories(memories: EnhancedMemory[]): Promise<void>
    // Initialize the layer
    initialize(): Promise<void>
}

// Base class for memory retrieval layer
abstract class BaseMemoryRetrievalLayer implements MemoryRetrievalLayer {
    protected vectorStore: ChatLunaSaveableVectorStore
    protected retriever: VectorStoreRetriever<ChatLunaSaveableVectorStore>

    constructor(
        protected ctx: Context,
        protected config: Config,
        protected memoryId: string
    ) {}

    abstract retrieveMemory(searchContent: string): Promise<EnhancedMemory[]>
    abstract addMemories(memories: EnhancedMemory[]): Promise<void>
    abstract initialize(): Promise<void>
}

// Standard vector store-based memory retrieval layer
class VectorStoreMemoryLayer extends BaseMemoryRetrievalLayer {
    constructor(
        protected ctx: Context,
        protected config: Config,
        protected memoryId: string
    ) {
        super(ctx, config, memoryId)

        ctx.setInterval(
            async () => {
                await this.cleanupExpiredMemories()
            },
            1000 * 60 * 5
        )
    }

    async initialize(): Promise<void> {
        this.retriever = await createVectorStoreRetriever(
            this.ctx,
            this.config,
            this.memoryId
        )
        this.vectorStore = this.retriever.vectorStore
    }

    async retrieveMemory(searchContent: string): Promise<EnhancedMemory[]> {
        let memory = await this.retriever.invoke(searchContent)

        if (this.config.longMemoryTFIDFThreshold > 0) {
            memory = filterMemory(
                memory,
                searchContent,
                this.config.longMemoryTFIDFThreshold
            )
        }

        return memory.map(documentToEnhancedMemory)
    }

    async addMemories(memories: EnhancedMemory[]): Promise<void> {
        if (!this.vectorStore) {
            logger?.warn('Vector store not initialized')
            return
        }

        if (
            this.config.longMemoryDuplicateThreshold < 1 &&
            this.config.longMemoryDuplicateCheck
        ) {
            memories = await filterSimilarMemory(
                memories,
                this.vectorStore,
                this.config.longMemoryDuplicateThreshold
            )
        }

        if (memories.length === 0) return

        await this.vectorStore.addDocuments(
            memories.map(enhancedMemoryToDocument),
            {}
        )

        if (this.vectorStore instanceof ChatLunaSaveableVectorStore) {
            logger?.debug('saving vector store')
            try {
                await this.vectorStore.save()
            } catch (e) {
                console.error(e)
            }
        }
    }

    async cleanupExpiredMemories(): Promise<void> {
        if (!this.vectorStore) {
            return
        }

        try {
            // 获取所有记忆
            const allMemories = await this.vectorStore.similaritySearch(
                '',
                1000
            )

            // 找出过期的记忆
            const expiredMemoriesIds: string[] = []

            for (const doc of allMemories) {
                const memory = documentToEnhancedMemory(doc)
                if (isMemoryExpired(memory) && doc.metadata?.raw_id) {
                    expiredMemoriesIds.push(doc.metadata.raw_id)
                }
            }

            if (expiredMemoriesIds.length > 0) {
                logger?.info(
                    `Found ${expiredMemoriesIds.length} expired memories to delete`
                )

                // 检查向量存储是否支持删除操作
                if (typeof this.vectorStore.delete === 'function') {
                    // 删除指定ID的记忆
                    await this.vectorStore.delete({ ids: expiredMemoriesIds })

                    // 保存向量存储
                    if (
                        this.vectorStore instanceof ChatLunaSaveableVectorStore
                    ) {
                        await this.vectorStore.save()
                    }

                    logger?.debug(
                        `Deleted ${expiredMemoriesIds.length} expired memories`
                    )
                } else {
                    logger?.warn('Vector store does not support deletion')
                }
            }
        } catch (e) {
            logger?.error(`Error cleaning up expired memories: ${e}`)
        }
    }
}

// 解析增强记忆
function parseEnhancedMemories(content: string): EnhancedMemory[] {
    // 预处理内容，移除可能的 markdown 代码块标记
    content = preprocessContent(content)

    try {
        // 尝试直接解析 JSON
        const result = tryParseJSON(content)
        if (result) {
            if (Array.isArray(result)) {
                // 处理 JSON 数组
                return result
                    .map((item) => createEnhancedMemoryFromItem(item))
                    .filter((item) => item.content) // 过滤掉没有内容的记忆
            } else if (typeof result === 'object' && result !== null) {
                // 尝试从对象中提取数组
                const possibleArrays = extractArraysFromObject(result)
                if (possibleArrays.length > 0) {
                    return possibleArrays[0]
                        .map((item) => createEnhancedMemoryFromItem(item))
                        .filter((item) => item.content)
                }
            }
        }

        // 尝试修复常见的 JSON 格式错误并重新解析
        const fixedContent = attemptToFixJSON(content)
        if (fixedContent !== content) {
            const fixedResult = tryParseJSON(fixedContent)
            if (fixedResult && Array.isArray(fixedResult)) {
                return fixedResult
                    .map((item) => createEnhancedMemoryFromItem(item))
                    .filter((item) => item.content)
            }
        }

        // 尝试使用正则表达式提取数组内容
        const extractedItems = extractArrayItemsWithRegex(content)
        if (extractedItems.length > 0) {
            return extractedItems
                .map((item) => createEnhancedMemoryFromItem(item))
                .filter((item) => item.content)
        }
    } catch (e) {
        logger?.error(`Error parsing enhanced memories: ${e}`)
    }

    // 如果所有解析方法都失败，将整个内容作为一条记忆
    if (content && content.trim()) {
        return [createDefaultMemory(content.trim(), MemoryType.CONTEXTUAL)]
    }

    return []
}

// 解析结果内容
function parseResultContent(content: string): EnhancedMemory[] {
    // 预处理内容，移除可能的 markdown 代码块标记
    content = preprocessContent(content)

    try {
        // 尝试直接解析 JSON
        const result = tryParseJSON(content)
        if (result) {
            if (Array.isArray(result)) {
                // 如果是数组，直接处理
                return result.map((item) =>
                    createDefaultMemory(
                        typeof item === 'string' ? item : JSON.stringify(item),
                        MemoryType.FACTUAL
                    )
                )
            } else if (typeof result === 'object' && result !== null) {
                // 尝试从对象中提取数组
                const possibleArrays = extractArraysFromObject(result)
                if (possibleArrays.length > 0) {
                    return possibleArrays[0].map((item) =>
                        createDefaultMemory(
                            typeof item === 'string'
                                ? item
                                : JSON.stringify(item),
                            MemoryType.FACTUAL
                        )
                    )
                }
            }
        }

        // 尝试修复常见的 JSON 格式错误并重新解析
        const fixedContent = attemptToFixJSON(content)
        if (fixedContent !== content) {
            const fixedResult = tryParseJSON(fixedContent)
            if (fixedResult && Array.isArray(fixedResult)) {
                return fixedResult.map((item) =>
                    createDefaultMemory(
                        typeof item === 'string' ? item : JSON.stringify(item),
                        MemoryType.FACTUAL
                    )
                )
            }
        }

        // 尝试使用正则表达式提取数组内容
        const extractedItems = extractArrayItemsWithRegex(content)
        if (extractedItems.length > 0) {
            return extractedItems.map((item) =>
                createDefaultMemory(
                    typeof item === 'string' ? item : JSON.stringify(item),
                    MemoryType.FACTUAL
                )
            )
        }
    } catch (e) {
        logger?.error(`Error parsing result content: ${e}`)
    }

    // 如果所有解析方法都失败，将整个内容作为一条记忆
    if (content && content.trim()) {
        return [createDefaultMemory(content.trim(), MemoryType.CONTEXTUAL)]
    }

    return []
}

/**
 * 预处理内容，移除可能的 markdown 代码块标记
 */
function preprocessContent(content: string): string {
    // 移除 markdown 代码块标记 (```json 和 ```)
    content = content.replace(
        /```(?:json|javascript|js)?\s*([\s\S]*?)```/g,
        '$1'
    )

    // 移除前后可能的空白字符
    content = content.trim()

    return content
}

/**
 * 尝试解析 JSON，失败时返回 null
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function tryParseJSON(content: string): any {
    try {
        return JSON.parse(content)
    } catch (e) {
        return null
    }
}

/**
 * 尝试修复常见的 JSON 格式错误
 */
function attemptToFixJSON(content: string): string {
    let fixedContent = content

    // 修复缺少引号的键名
    fixedContent = fixedContent.replace(
        /(\{|\,)\s*([a-zA-Z0-9_]+)\s*\:/g,
        '$1"$2":'
    )

    // 修复使用单引号而非双引号的情况
    fixedContent = fixedContent.replace(/(\{|\,)\s*'([^']+)'\s*\:/g, '$1"$2":')
    fixedContent = fixedContent.replace(/\:\s*'([^']+)'/g, ':"$1"')

    // 修复缺少逗号的情况
    fixedContent = fixedContent.replace(/"\s*\}\s*"/g, '","')
    fixedContent = fixedContent.replace(/"\s*\{\s*"/g, '",{"')

    // 修复多余的逗号
    fixedContent = fixedContent.replace(/,\s*\}/g, '}')
    fixedContent = fixedContent.replace(/,\s*\]/g, ']')

    // 修复不完整的数组
    if (fixedContent.includes('[') && !fixedContent.includes(']')) {
        fixedContent += ']'
    }

    // 修复不完整的对象
    if (fixedContent.includes('{') && !fixedContent.includes('}')) {
        fixedContent += '}'
    }

    // 如果内容不是以 [ 开头但包含 [ 字符，尝试提取数组部分
    if (!fixedContent.trim().startsWith('[') && fixedContent.includes('[')) {
        const arrayMatch = fixedContent.match(/\[([\s\S]*)\]/)
        if (arrayMatch && arrayMatch[0]) {
            fixedContent = arrayMatch[0]
        }
    }

    return fixedContent
}

/**
 * 从对象中提取所有数组
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function extractArraysFromObject(obj: any): any[][] {
    return Object.values(obj).filter(
        (value) => Array.isArray(value) && value.length > 0
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ) as any[][]
}

/**
 * 使用正则表达式提取数组项
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function extractArrayItemsWithRegex(content: string): any[] {
    // 尝试提取方括号中的内容
    const arrayRegex = /\[([\s\S]*)\]/s
    const arrayMatch = content.match(arrayRegex)

    if (arrayMatch && arrayMatch[1]) {
        // 尝试分割数组项
        // 处理带引号的字符串项
        const items = arrayMatch[1]
            .split(/,(?=\s*['"]|\s*\{)/g)
            .map((item) => item.trim())
            .filter((item) => item.length > 0)

        // 尝试解析每个项
        return items.map((item) => {
            // 如果项看起来像对象，尝试解析它
            if (item.startsWith('{') && item.endsWith('}')) {
                try {
                    return JSON.parse(item)
                } catch (e) {
                    // 尝试修复并重新解析
                    const fixedItem = attemptToFixJSON(item)
                    try {
                        return JSON.parse(fixedItem)
                    } catch (e) {
                        // 如果仍然失败，返回原始字符串
                        return item
                    }
                }
            }

            // 移除引号
            return item.replace(/^['"]|['"]$/g, '')
        })
    }

    return []
}

/**
 * 从项创建增强记忆对象
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function createEnhancedMemoryFromItem(item: any): EnhancedMemory {
    if (typeof item === 'string') {
        // 如果是字符串，创建默认增强记忆
        return createDefaultMemory(item, MemoryType.CONTEXTUAL)
    } else if (typeof item === 'object' && item !== null) {
        // 验证并规范化对象
        const memory: EnhancedMemory = {
            content: item.content || item.text || item.memory || '',
            type: Object.values(MemoryType).includes(item.type)
                ? item.type
                : MemoryType.CONTEXTUAL,
            importance:
                typeof item.importance === 'number' ? item.importance : 5
        }

        // 自动计算过期时间，忽略模型提供的过期时间
        memory.expirationDate = calculateExpirationDate(
            memory.type,
            memory.importance
        )

        return memory
    }

    // 默认情况
    return createDefaultMemory(String(item), MemoryType.CONTEXTUAL)
}

/**
 * 创建默认记忆对象
 */
function createDefaultMemory(
    content: string,
    type: MemoryType,
    importance: number = 5
): EnhancedMemory {
    const memory: EnhancedMemory = {
        content,
        type,
        importance
    }

    // 自动计算过期时间
    memory.expirationDate = calculateExpirationDate(
        memory.type,
        memory.importance
    )

    return memory
}

async function filterSimilarMemory(
    memoryArray: EnhancedMemory[],
    vectorStore: VectorStore,
    similarityThreshold: number
): Promise<EnhancedMemory[]> {
    const result: EnhancedMemory[] = []

    const existingMemories = await vectorStore.similaritySearch('', 1000)

    for (const memory of memoryArray) {
        let isSimilar = false

        for (const existingMemory of existingMemories) {
            const similarity = calculateSimilarity(
                memory.content,
                existingMemory.pageContent
            )

            if (similarity.score >= similarityThreshold) {
                isSimilar = true
                break
            }
        }

        if (!isSimilar) {
            result.push(memory)
        } else {
            logger?.debug(
                `Skip similar memory: ${memory.content}, threshold: ${similarityThreshold}`
            )
        }
    }

    return result
}

function filterMemory(
    memory: Document[],
    searchContent: string,
    threshold: number
): Document[] {
    const result: Document[] = []

    for (const doc of memory) {
        const similarity = calculateSimilarity(searchContent, doc.pageContent)

        if (similarity.score >= threshold) {
            result.push(doc)
        } else {
            logger?.debug(
                `Skip memory: ${doc.pageContent}, similarity: ${similarity}, threshold: ${threshold}`
            )
        }
    }

    return result
}

function resolveLongMemoryId(
    message: HumanMessage,
    conversationId: string,
    layerType: string
) {
    const preset = message.additional_kwargs?.preset as string

    if (!preset) {
        return conversationId
    }

    const userId = message.id

    let hash = crypto.createHash('sha256')

    if (layerType === 'preset') {
        hash = hash.update(`${preset}`)
    } else {
        hash = hash.update(`${preset}-${userId}`)
    }

    const hex = hash.digest('hex')

    logger?.debug(`Long memory id: ${preset}-${userId} => ${hex}`)

    return hex
}

async function createVectorStoreRetriever(
    ctx: Context,
    config: Config,
    longMemoryId: string
): Promise<VectorStoreRetriever<ChatLunaSaveableVectorStore>> {
    const [platform, model] = parseRawModelName(
        ctx.chatluna.config.defaultEmbeddings
    )
    const embeddingModel = await ctx.chatluna.createEmbeddings(platform, model)

    const vectorStore = await ctx.chatluna.platform.createVectorStore(
        ctx.chatluna.config.defaultVectorStore,
        {
            embeddings: embeddingModel,
            key: longMemoryId
        }
    )

    const retriever = ScoreThresholdRetriever.fromVectorStore(vectorStore, {
        minSimilarityScore: Math.min(0.1, config.longMemorySimilarity - 0.3), // Finds results with at least this similarity score
        maxK: 50, // The maximum K value to use. Use it based to your chunk size to make sure you don't run out of tokens
        kIncrement: 2, // How much to increase K by each time. It'll fetch N results, then N + kIncrement, then N + kIncrement * 2, etc.,
        searchType: 'mmr'
    })

    return retriever
}

async function generateNewQuestion(
    chatInterface: ChatInterface,
    config: Config,
    chatHistory: string,
    question: string
): Promise<string> {
    const [platform, modelName] = parseRawModelName(
        config.longMemoryExtractModel
    )

    const model = await chatInterface.ctx.chatluna.createChatModel(
        platform,
        modelName
    )

    const prompt = `
Given the following conversation history and the user's question, generate a new search query that will help retrieve relevant information from a long-term memory database. The search query should be concise and focused on the key information needs.

If you think the user's question is a casual greeting, a simple question that doesn't need memory retrieval, or is not related to any previous context, just respond with "[skip]".

Conversation History:
${chatHistory}

User Question: ${question}

New Search Query:
`

    const messages: BaseMessage[] = [new HumanMessage(prompt)]
    const result = await model.invoke(messages)

    return result.content as string
}

async function selectChatHistory(
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

const ENHANCED_MEMORY_PROMPT = `Extract key memories from this chat as a JSON array of structured memory objects:
{user_input}

Guidelines:
1. Focus on extracting factual information, preferences, and important details about the user.
2. Capture information that would be valuable to remember for future conversations.
3. Each memory should be a complete, standalone sentence that makes sense without additional context.
4. Avoid creating memories about the current conversation flow or meta-discussions.
5. Do not include greetings, pleasantries, or other conversation fillers.
6. Categorize each memory into one of these types:
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

7. Assign an importance score (1-10) to each memory:
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

If no meaningful memories can be extracted, return an empty array: []`

// 从聊天历史中提取记忆
async function extractMemoriesFromChat(
    ctx: Context,
    config: Config,
    chatInterface: ChatInterface,
    chatHistory: string
): Promise<EnhancedMemory[]> {
    const preset = await chatInterface.preset
    const input = (
        preset.config?.longMemoryExtractPrompt ?? ENHANCED_MEMORY_PROMPT
    ).replaceAll('{user_input}', chatHistory)

    const messages: BaseMessage[] = [new HumanMessage(input)]
    const [platform, modelName] = parseRawModelName(
        config.longMemoryExtractModel
    )

    const model = (await ctx.chatluna.createChatModel(
        platform,
        modelName
    )) as ChatLunaChatModel

    const extractMemory = async () => {
        const result = await model.invoke(messages)

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
        logger?.warn('Error extracting long memory')
        return []
    }

    logger?.debug(`Long memory extract: ${JSON.stringify(memories)}`)

    return memories
}

// 应用函数
export function apply(ctx: Context, config: Config) {
    // 缓存长期记忆层
    const memoryLayers: Record<string, VectorStoreMemoryLayer[]> = {}

    // 获取或创建记忆层
    async function getOrCreateMemoryLayer(
        message: HumanMessage,
        conversationId: string
    ): Promise<VectorStoreMemoryLayer[]> {
        const layers = memoryLayers[conversationId]

        if (!layers) {
            const layers = await Promise.all(
                resolveLayerNames().map(async (layerId) => {
                    const memoryId = resolveLongMemoryId(
                        message,
                        conversationId,
                        layerId
                    )
                    const layer = new VectorStoreMemoryLayer(
                        ctx,
                        config,
                        memoryId
                    )

                    await layer.initialize()
                    return layer
                })
            )

            memoryLayers[conversationId] = layers

            return layers
        }

        return layers
    }

    // 在聊天前处理长期记忆
    ctx.on(
        'chatluna/before-chat',
        async (conversationId, message, promptVariables, chatInterface) => {
            const layers = await getOrCreateMemoryLayer(message, conversationId)

            let searchContent =
                (message.additional_kwargs['raw_content'] as string | null) ??
                getMessageContent(message.content)

            if (config.longMemoryNewQuestionSearch) {
                const chatHistory = await selectChatHistory(
                    await chatInterface.chatHistory.getMessages(),
                    message.id,
                    config.longMemoryInterval
                )

                searchContent = await generateNewQuestion(
                    chatInterface,
                    config,
                    chatHistory,
                    searchContent
                )

                if (searchContent === '[skip]') {
                    logger?.debug("Don't search long memory")
                    return
                }
            }

            logger?.debug(`Long memory search: ${searchContent}`)

            // 使用记忆层检索记忆
            const memories = await Promise.all(
                layers.map((layer) => layer.retrieveMemory(searchContent))
            )

            // 过滤掉过期的记忆
            const validMemories = memories
                .flat()
                .filter((memory) => !isMemoryExpired(memory))

            logger?.debug(`Long memory: ${JSON.stringify(validMemories)}`)

            promptVariables['long_memory'] = validMemories.map(
                enhancedMemoryToDocument
            )
        }
    )

    // 在聊天后处理长期记忆
    ctx.on(
        'chatluna/after-chat',
        async (
            conversationId,
            sourceMessage,
            _,
            promptVariables,
            chatInterface
        ) => {
            if (config.longMemoryExtractModel === '无') {
                logger?.warn(
                    'Long memory extract model is not set, skip long memory'
                )
                return undefined
            }

            const chatCount = promptVariables['chatCount'] as number

            if (chatCount % config.longMemoryInterval !== 0) return undefined

            const layer = await getOrCreateMemoryLayer(
                sourceMessage,
                conversationId
            ).then((layers) => layers[0])

            const chatHistory = await selectChatHistory(
                await chatInterface.chatHistory.getMessages(),
                sourceMessage.id ?? undefined,
                config.longMemoryInterval
            )

            // 提取记忆
            const memories = await extractMemoriesFromChat(
                ctx,
                config,
                chatInterface,
                chatHistory
            )

            if (memories.length === 0) return

            // 添加记忆到记忆层
            await layer.addMemories(memories)
        }
    )

    // 清理聊天历史时清理长期记忆缓存
    ctx.on(
        'chatluna/clear-chat-history',
        async (conversationId, _chatInterface) => {
            // 删除特定会话的记忆层
            delete memoryLayers[conversationId]
        }
    )

    // 定期清理过期记忆
    ctx.setInterval(
        async () => {
            for (const [, layers] of Object.entries(memoryLayers)) {
                for (const layer of layers) {
                    await layer.cleanupExpiredMemories()
                }
            }
        },
        1000 * 60 * 10
    ) // 每10分钟清理一次过期记忆
}
