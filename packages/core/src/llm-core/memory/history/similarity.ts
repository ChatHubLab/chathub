export interface SimilarityResult {
    score: number
    details: {
        cosine: number
        levenshtein: number
        jaccard: number
    }
}

export function calculateSimilarity(
    str1: string,
    str2: string
): SimilarityResult {
    const normalize = (str: string): string => {
        return str.toLowerCase().trim().replace(/\s+/g, ' ')
    }
    const text1 = normalize(str1)
    const text2 = normalize(str2)

    // Levenshtein
    function levenshteinDistance(s1: string, s2: string): number {
        const dp: number[][] = Array(s1.length + 1)
            .fill(null)
            .map(() => Array(s2.length + 1).fill(0))

        for (let i = 0; i <= s1.length; i++) dp[i][0] = i
        for (let j = 0; j <= s2.length; j++) dp[0][j] = j

        for (let i = 1; i <= s1.length; i++) {
            for (let j = 1; j <= s2.length; j++) {
                if (s1[i - 1] === s2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1]
                } else {
                    dp[i][j] = Math.min(
                        dp[i - 1][j] + 1, // 删除
                        dp[i][j - 1] + 1, // 插入
                        dp[i - 1][j - 1] + 1 // 替换
                    )
                }
            }
        }

        return 1 - dp[s1.length][s2.length] / Math.max(s1.length, s2.length)
    }

    // Jaccard
    function jaccardSimilarity(s1: string, s2: string): number {
        const words1 = new Set(s1.split(/\s+/))
        const words2 = new Set(s2.split(/\s+/))

        const intersection = new Set([...words1].filter((x) => words2.has(x)))
        const union = new Set([...words1, ...words2])

        return intersection.size / union.size
    }

    // 余弦相似度
    function cosineSimilarity(s1: string, s2: string): number {
        const getWordVector = (str: string): Map<string, number> => {
            const words = str.split(/\s+/)
            const vector = new Map<string, number>()
            words.forEach((word) => {
                vector.set(word, (vector.get(word) || 0) + 1)
            })
            return vector
        }

        const vector1 = getWordVector(s1)
        const vector2 = getWordVector(s2)

        let dotProduct = 0
        for (const [word, count1] of vector1) {
            const count2 = vector2.get(word) || 0
            dotProduct += count1 * count2
        }

        const magnitude1 = Math.sqrt(
            [...vector1.values()].reduce((sum, count) => sum + count * count, 0)
        )
        const magnitude2 = Math.sqrt(
            [...vector2.values()].reduce((sum, count) => sum + count * count, 0)
        )

        return dotProduct / (magnitude1 * magnitude2) || 0
    }

    // 5. 计算综合相似度分数
    const cosine = cosineSimilarity(text1, text2)
    const levenshtein = levenshteinDistance(text1, text2)
    const jaccard = jaccardSimilarity(text1, text2)

    // 加权平均
    const weights = {
        cosine: 0.4, // 词频相似度权重
        levenshtein: 0.3, // 编辑距离权重
        jaccard: 0.3 // 词集合相似度权重
    }

    const score =
        cosine * weights.cosine +
        levenshtein * weights.levenshtein +
        jaccard * weights.jaccard

    return {
        score,
        details: { cosine, levenshtein, jaccard }
    }
}
