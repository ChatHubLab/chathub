import { logger } from '..'
import { EnhancedMemory, MemoryType } from '../types'
import { createDefaultMemory, createEnhancedMemoryFromItem } from './memory'

// 解析增强记忆
export function parseEnhancedMemories(content: string): EnhancedMemory[] {
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
export function parseResultContent(content: string): EnhancedMemory[] {
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
export function extractArraysFromObject(obj: any): any[][] {
    return Object.values(obj).filter(
        (value) => Array.isArray(value) && value.length > 0
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ) as any[][]
}

/**
 * 使用正则表达式提取数组项
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function extractArrayItemsWithRegex(content: string): any[] {
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
