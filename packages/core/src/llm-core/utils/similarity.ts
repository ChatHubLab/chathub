import { Jieba } from '@node-rs/jieba'
import { dict } from '@node-rs/jieba/dict'
import TinySegmenter from 'tiny-segmenter'

const jieba = Jieba.withDict(dict)
const segmenter = new TinySegmenter()

const SIMILARITY_WEIGHTS = {
    cosine: 0.4,
    levenshtein: 0.3,
    jaccard: 0.3
} as const

export interface SimilarityResult {
    score: number
    details: {
        cosine: number
        levenshtein: number
        jaccard: number
    }
}

class TextTokenizer {
    private static isChinese(text: string): boolean {
        return /[\u4e00-\u9fa5]/.test(text)
    }

    private static isJapanese(text: string): boolean {
        return /[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]/.test(text)
    }

    static tokenize(text: string): string[] {
        if (this.isChinese(text)) {
            return jieba.cut(text, false)
        }
        if (this.isJapanese(text)) {
            return segmenter.segment(text)
        }
        return text.split(/\s+/)
    }

    static normalize(text: string): string {
        return text.toLowerCase().trim().replace(/\s+/g, ' ')
    }
}

export class SimilarityCalculator {
    private static levenshteinDistance(s1: string, s2: string): number {
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
                        dp[i - 1][j] + 1,
                        dp[i][j - 1] + 1,
                        dp[i - 1][j - 1] + 1
                    )
                }
            }
        }

        return 1 - dp[s1.length][s2.length] / Math.max(s1.length, s2.length)
    }

    private static jaccardSimilarity(s1: string, s2: string): number {
        const words1 = new Set(TextTokenizer.tokenize(s1))
        const words2 = new Set(TextTokenizer.tokenize(s2))

        const intersection = new Set([...words1].filter(x => words2.has(x)))
        const union = new Set([...words1, ...words2])

        return intersection.size / union.size
    }

    private static cosineSimilarity(s1: string, s2: string): number {
        const getWordVector = (str: string): Map<string, number> => {
            const words = TextTokenizer.tokenize(str)
            return words.reduce((vector, word) => {
                vector.set(word, (vector.get(word) || 0) + 1)
                return vector
            }, new Map<string, number>())
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

        if (magnitude1 === 0 || magnitude2 === 0) return 0
        return dotProduct / (magnitude1 * magnitude2)
    }

    public static calculate(str1: string, str2: string): SimilarityResult {
        if (!str1 || !str2) {
            throw new Error('Input strings cannot be empty')
        }

        const text1 = TextTokenizer.normalize(str1)
        const text2 = TextTokenizer.normalize(str2)

        const cosine = this.cosineSimilarity(text1, text2)
        const levenshtein = this.levenshteinDistance(text1, text2)
        const jaccard = this.jaccardSimilarity(text1, text2)

        const score =
            cosine * SIMILARITY_WEIGHTS.cosine +
            levenshtein * SIMILARITY_WEIGHTS.levenshtein +
            jaccard * SIMILARITY_WEIGHTS.jaccard

        return {
            score,
            details: { cosine, levenshtein, jaccard }
        }
    }
}

export function calculateSimilarity(
    str1: string,
    str2: string
): SimilarityResult {
    return SimilarityCalculator.calculate(str1, str2)
}
