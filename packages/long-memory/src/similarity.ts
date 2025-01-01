import { cut } from 'jieba-wasm'
import TinySegmenter from 'tiny-segmenter'
import stopwords from 'stopwords-iso'

const segmenter = new TinySegmenter()

const SIMILARITY_WEIGHTS = {
    cosine: 0.3,
    levenshtein: 0.2,
    jaccard: 0.2,
    bm25: 0.3
} as const

function validateAndAdjustWeights(weights: typeof SIMILARITY_WEIGHTS) {
    const totalWeight = Object.values(weights).reduce(
        (sum, weight) => sum + weight,
        0
    )

    if (Math.abs(totalWeight - 1) > 0.0001) {
        const adjustmentFactor = 1 / totalWeight
        return Object.fromEntries(
            Object.entries(weights).map(([key, value]) => [
                key,
                value * adjustmentFactor
            ])
        ) as typeof SIMILARITY_WEIGHTS
    }

    return weights
}

const VALIDATED_WEIGHTS = validateAndAdjustWeights(SIMILARITY_WEIGHTS)

export interface SimilarityResult {
    score: number
    details: {
        cosine: number
        levenshtein: number
        jaccard: number
        bm25: number
    }
}

class TextTokenizer {
    private static stopwords = new Set([
        ...stopwords.zh,
        ...stopwords.en,
        ...stopwords.ja
    ])

    private static readonly REGEX = {
        chinese: /[\u4e00-\u9fff]/,
        japanese: /[\u3040-\u30ff\u3400-\u4dbf]/,
        english: /[a-zA-Z]/
    }

    private static detectLanguages(text: string): Set<string> {
        const languages = new Set<string>()

        if (TextTokenizer.REGEX.chinese.test(text)) languages.add('zh')
        if (TextTokenizer.REGEX.japanese.test(text)) languages.add('ja')
        if (TextTokenizer.REGEX.english.test(text)) languages.add('en')

        return languages
    }

    static tokenize(text: string): string[] {
        const languages = TextTokenizer.detectLanguages(text)
        let tokens: string[] = []

        if (languages.size === 1 && languages.has('en')) {
            tokens = text.split(/\s+/)
            return this.removeStopwords(tokens)
        }

        let currentText = text

        if (languages.has('zh')) {
            const zhTokens = cut(currentText, false)
            currentText = zhTokens.join('▲')
        }

        if (languages.has('ja')) {
            const segments = segmenter.segment(currentText)
            currentText = segments.join('▲')
        }

        if (languages.has('en')) {
            currentText = currentText.replace(/\s+/g, '▲')
        }

        tokens = currentText.split('▲').filter(Boolean)

        return this.removeStopwords(tokens)
    }

    static normalize(text: string): string {
        return text
            .toLowerCase()
            .trim()
            .replace(/[^\w\s\u4e00-\u9fff\u3040-\u30ff\u3400-\u4dbf]/g, '')
            .replace(/\s+/g, ' ')
    }

    static removeStopwords(tokens: string[]): string[] {
        return tokens.filter((token) => {
            if (!token || /^\d+$/.test(token)) return false

            if (
                token.length === 1 &&
                !TextTokenizer.REGEX.chinese.test(token) &&
                !TextTokenizer.REGEX.japanese.test(token)
            ) {
                return false
            }

            return !TextTokenizer.stopwords.has(token)
        })
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

        const intersection = new Set([...words1].filter((x) => words2.has(x)))
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

    private static calculateBM25Similarity(s1: string, s2: string): number {
        const k1 = 1.5 // 词频饱和参数
        const b = 0.75 // 文档长度归一化参数
        const epsilon = 0.25 // 平滑因子

        const tokens1 = TextTokenizer.tokenize(s1)
        const tokens2 = TextTokenizer.tokenize(s2)

        if (tokens1.length === 0 || tokens2.length === 0) {
            return 0
        }

        const doc1Length = tokens1.length
        const doc2Length = tokens2.length
        const avgDocLength = (doc1Length + doc2Length) / 2

        const termFreqDoc1 = new Map<string, number>()
        const termFreqDoc2 = new Map<string, number>()
        const uniqueTerms = new Set([...tokens1, ...tokens2])

        tokens1.forEach((token) => {
            termFreqDoc1.set(token, (termFreqDoc1.get(token) || 0) + 1)
        })

        tokens2.forEach((token) => {
            termFreqDoc2.set(token, (termFreqDoc2.get(token) || 0) + 1)
        })

        // 计算双向 BM25 得分
        let score1to2 = 0
        let score2to1 = 0
        let maxScore1to2 = 0
        let maxScore2to1 = 0

        for (const term of uniqueTerms) {
            // 计算 doc1 -> doc2 的方向
            const tf1 = termFreqDoc1.get(term) || 0
            const docFreq1 = (termFreqDoc2.get(term) || 0) > 0 ? 1 : 0
            if (tf1 > 0) {
                const idf1 = Math.log((2 - docFreq1 + epsilon) / (docFreq1 + epsilon) + 1)
                const numerator1 = tf1 * (k1 + 1)
                const denominator1 = tf1 + k1 * (1 - b + b * (doc1Length / avgDocLength))
                score1to2 += idf1 * (numerator1 / denominator1)

                const maxTf1 = Math.max(tf1, termFreqDoc2.get(term) || 0)
                const maxNumerator1 = maxTf1 * (k1 + 1)
                const maxDenominator1 = maxTf1 + k1 * (1 - b + b * (doc1Length / avgDocLength))
                maxScore1to2 += idf1 * (maxNumerator1 / maxDenominator1)
            }

            // 计算 doc2 -> doc1 的方向
            const tf2 = termFreqDoc2.get(term) || 0
            const docFreq2 = (termFreqDoc1.get(term) || 0) > 0 ? 1 : 0
            if (tf2 > 0) {
                const idf2 = Math.log((2 - docFreq2 + epsilon) / (docFreq2 + epsilon) + 1)
                const numerator2 = tf2 * (k1 + 1)
                const denominator2 = tf2 + k1 * (1 - b + b * (doc2Length / avgDocLength))
                score2to1 += idf2 * (numerator2 / denominator2)

                const maxTf2 = Math.max(tf2, termFreqDoc1.get(term) || 0)
                const maxNumerator2 = maxTf2 * (k1 + 1)
                const maxDenominator2 = maxTf2 + k1 * (1 - b + b * (doc2Length / avgDocLength))
                maxScore2to1 += idf2 * (maxNumerator2 / maxDenominator2)
            }
        }

        const normalizedScore1 = maxScore1to2 > 0 ? score1to2 / maxScore1to2 : 0
        const normalizedScore2 = maxScore2to1 > 0 ? score2to1 / maxScore2to1 : 0

        return (normalizedScore1 + normalizedScore2) / 2
    }

    public static calculate(str1: string, str2: string): SimilarityResult {
        if (!str1 || !str2) {
            throw new Error('Input strings cannot be empty')
        }

        const text1 = TextTokenizer.normalize(str1)
        const text2 = TextTokenizer.normalize(str2)

        const cosine = SimilarityCalculator.cosineSimilarity(text1, text2)
        const levenshtein = SimilarityCalculator.levenshteinDistance(
            text1,
            text2
        )
        const jaccard = SimilarityCalculator.jaccardSimilarity(text1, text2)
        const bm25 = SimilarityCalculator.calculateBM25Similarity(text1, text2)

        const score =
            cosine * VALIDATED_WEIGHTS.cosine +
            levenshtein * VALIDATED_WEIGHTS.levenshtein +
            jaccard * VALIDATED_WEIGHTS.jaccard +
            bm25 * VALIDATED_WEIGHTS.bm25

        return {
            score,
            details: { cosine, levenshtein, jaccard, bm25 }
        }
    }
}

export function calculateSimilarity(
    str1: string,
    str2: string
): SimilarityResult {
    return SimilarityCalculator.calculate(str1, str2)
}
