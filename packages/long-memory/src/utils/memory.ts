import { EnhancedMemory, MemoryType } from '../types'
import { Document } from '@langchain/core/documents'

// 根据记忆类型和重要性计算过期时间
export function calculateExpirationDate(
    type: MemoryType,
    importance: number
): Date {
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

// 将增强记忆转换为Document
export function enhancedMemoryToDocument(memory: EnhancedMemory): Document {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const metadata: Record<string, any> = {
        type: memory.type,
        importance: memory.importance
    }

    if (memory.expirationDate) {
        metadata.expirationDate = memory.expirationDate.toISOString()
    }

    return new Document({
        pageContent: memory.content,
        metadata
    })
}

// 将Document转换为增强记忆
export function documentToEnhancedMemory(document: Document): EnhancedMemory {
    const metadata = document.metadata || {}

    const memory: EnhancedMemory = {
        content: document.pageContent,
        type: metadata.type || MemoryType.FACTUAL,
        importance: metadata.importance || 5
    }

    if (metadata.expirationDate) {
        memory.expirationDate = new Date(metadata.expirationDate)
    }

    if (metadata.raw_id) {
        memory.rawId = metadata.raw_id
    }

    return memory
}

// 检查记忆是否过期
export function isMemoryExpired(memory: EnhancedMemory): boolean {
    if (!memory.expirationDate) return false
    return new Date() > memory.expirationDate
}

/**
 * 从项创建增强记忆对象
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function createEnhancedMemoryFromItem(item: any): EnhancedMemory {
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
export function createDefaultMemory(
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
