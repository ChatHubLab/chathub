/* eslint-disable max-len */
/* eslint-disable @typescript-eslint/naming-convention */
import { Context, Logger, Schema } from 'koishi'
import { ClientConfig } from 'koishi-plugin-chatluna/llm-core/platform/config'
import { PlatformService } from 'koishi-plugin-chatluna/llm-core/platform/service'
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'
import { createLogger } from 'koishi-plugin-chatluna/utils/logger'
import { ChatLunaBrowsingChain } from './chain/browsing_chain'
import { PuppeteerBrowserTool } from './tools/puppeteerBrowserTool'
import { apply as configApply } from './config'
import { parseRawModelName } from 'koishi-plugin-chatluna/llm-core/utils/count_tokens'
import { ChatLunaChatModel } from 'koishi-plugin-chatluna/llm-core/platform/model'
import { SearchManager } from './provide'
import { providerPlugin } from './plugin'
import { SearchTool } from './tools/search'
export let logger: Logger

export function apply(ctx: Context, config: Config) {
    logger = createLogger(ctx, 'chatluna-search-service')
    const plugin = new ChatLunaPlugin<ClientConfig, Config>(
        ctx,
        config,
        'search-service',
        false
    )

    ctx.on('ready', async () => {
        plugin.registerToService()

        const summaryModel = config.enhancedSummary
            ? await createModel(ctx, config.summaryModel)
            : undefined

        const searchManager = new SearchManager(ctx, config)

        providerPlugin(ctx, config, plugin, searchManager)

        plugin.registerTool('web-search', {
            async createTool(params, session) {
                const browserTool = new PuppeteerBrowserTool(
                    ctx,
                    summaryModel ?? params.model,
                    params.embeddings
                )
                return new SearchTool(searchManager, browserTool)
            },
            selector() {
                return true
            }
        })

        plugin.registerTool('web-browser', {
            async createTool(params, session) {
                return new PuppeteerBrowserTool(
                    ctx,
                    summaryModel ?? params.model,
                    params.embeddings
                )
            },
            selector() {
                return true
            }
        })

        plugin.registerChatChainProvider(
            'browsing',
            {
                'zh-CN': '浏览模式，可以从外部获取信息',
                'en-US': 'Browsing mode, can get information from web'
            },
            async (params) => {
                const tools = getTools(
                    ctx.chatluna.platform,
                    (name) =>
                        name === 'web-search' ||
                        name === 'web-browser' ||
                        name === 'puppeteer_browser'
                )

                const model = params.model
                const options = {
                    preset: params.preset,
                    botName: params.botName,
                    embeddings: params.embeddings,
                    historyMemory: params.historyMemory,
                    enhancedSummary: config.enhancedSummary,
                    summaryModel: summaryModel ?? params.model,
                    thoughtMessage: ctx.chatluna.config.showThoughtMessage,
                    searchPrompt: config.searchPrompt,
                    newQuestionPrompt: config.newQuestionPrompt,
                    searchConfidenceThreshold: config.searchConfidenceThreshold
                }

                return ChatLunaBrowsingChain.fromLLMAndTools(
                    model,
                    tools,
                    options
                )
            }
        )
    })

    configApply(ctx, config)
}

function getTools(service: PlatformService, filter: (name: string) => boolean) {
    const tools = service.getTools().filter(filter)

    return tools.map((name) => ({
        name,
        tool: service.getTool(name)
    }))
}

export async function createModel(ctx: Context, model: string) {
    if (model == null) {
        return null
    }

    const [platform, modelName] = parseRawModelName(model)
    await ctx.chatluna.awaitLoadPlatform(platform)
    return ctx.chatluna.createChatModel(
        platform,
        modelName
    ) as Promise<ChatLunaChatModel>
}

export interface Config extends ChatLunaPlugin.Config {
    searchEngine: string[]
    topK: number
    enhancedSummary: boolean
    summaryModel: string

    serperApiKey: string
    serperCountry: string
    serperLocation: string
    serperSearchResults: number

    bingSearchApiKey: string
    bingSearchLocation: string
    azureLocation: string

    wikipediaBaseURL: string[]
    maxWikipediaDocContentLength: number

    tavilyApiKey: string

    searxngBaseURL: string

    puppeteerTimeout: number
    puppeteerIdleTimeout: number

    searchPrompt: string
    newQuestionPrompt: string
    searchConfidenceThreshold: number
}

export const Config: Schema<Config> = Schema.intersect([
    ChatLunaPlugin.Config,

    Schema.object({
        searchEngine: Schema.array(
            Schema.union([
                Schema.const('bing-web').description('Bing (Web)'),
                Schema.const('bing-api').description('Bing (API)'),
                Schema.const('duckduckgo-lite').description(
                    'DuckDuckGo (Lite)'
                ),
                Schema.const('serper').description('Serper (Google)'),
                Schema.const('tavily').description('Tavily (API)'),
                Schema.const('google-web').description('Google (Web)'),
                Schema.const('wikipedia').description('Wikipedia'),
                Schema.const('searxng').description('SearxNG')
            ])
        )
            .default(['bing-web'])
            .role('select'),
        topK: Schema.number().min(2).max(50).step(1).default(5),
        enhancedSummary: Schema.boolean().default(false),
        fastEnhancedSummary: Schema.boolean().default(false),
        puppeteerTimeout: Schema.number().default(60000),
        puppeteerIdleTimeout: Schema.number().default(300000),
        summaryModel: Schema.dynamic('model'),
        searchConfidenceThreshold: Schema.percent().step(0.01).default(0.5)
    }),

    Schema.object({
        serperApiKey: Schema.string().role('secret'),
        serperCountry: Schema.string().default('cn'),
        serperLocation: Schema.string().default('zh-cn'),
        serperSearchResults: Schema.number().min(2).max(20).default(10)
    }),

    Schema.object({
        bingSearchApiKey: Schema.string().role('secret'),
        bingSearchLocation: Schema.string().default('zh-CN'),
        azureLocation: Schema.string().default('global')
    }),

    Schema.object({
        tavilyApiKey: Schema.string().role('secret')
    }),

    Schema.object({
        wikipediaBaseURL: Schema.array(Schema.string()).default([
            'https://en.wikipedia.org/w/api.php',
            'https://mzh.moegirl.org.cn/api.php'
        ]),
        maxWikipediaDocContentLength: Schema.number().default(5000)
    }),

    Schema.object({
        searxngBaseURL: Schema.string().default('https://paulgo.io')
    }),

    Schema.object({
        searchPrompt: Schema.string()
            .role('textarea')
            .default(
                `GOAL: Generate a concise, informative answer based solely on the provided search results (URL and content).

INSTRUCTIONS:
- CRITICAL: Use the exact same language as the input. Do not translate or change the language under any circumstances.
- Use only information from the search results
- Adopt an unbiased, journalistic tone
- Combine results into a coherent answer
- Avoid repetition
- Use bullet points for readability
- Cite sources using superscript numbers in square brackets (e.g., [^1], [^2]) at the end of relevant sentences/paragraphs
- For multiple citations in one sentence, use [^1][^2]
- Never repeat the same citation number in a sentence
- If results refer to different entities with the same name, provide separate answers
- Match the system message style
- List sources as numbered references at the end using Markdown syntax
- If image sources are present in the context, include them using Markdown image syntax: ![alt text](image_url)

Content within 'context' html blocks is from a knowledge bank, not user conversation.

<context>
    {context}
</context>

IMPORTANT: Your response MUST be in the same language as the original input. This is crucial for maintaining context and accuracy. Do not translate or change the language under any circumstances.

REMEMBER: If no relevant context is found, provide an answer based on your knowledge, but inform the user it may not be current or fully accurate. Suggest they verify the information. Content within 'context' html blocks is from a knowledge bank, not user conversation.

FINAL REMINDER: Ensure that your entire response, including any explanations or suggestions, is in the exact same language as the original input.`
            ),
        newQuestionPrompt: Schema.string()
            .role('textarea')
            .default(
                `Rephrase the follow-up question as a clear, concise, standalone question optimized for search engine queries, based on the context of the previous conversation.

Rules:
- CRITICAL: Use the exact same language as the input. Do not translate or modify the language in any way, even if the language or phrasing seems ambiguous or unclear. The output must maintain the integrity of the original input.
- Clarity: The rephrased question must be unambiguous, clear, and grammatically correct. Ensure that the question makes sense on its own and can be easily understood without additional context. Remove any extra words that don't add value or clarity.
- Search Optimization: The question must be rephrased in a way that would be effective for a search engine. Use common search terms and focus on the keywords that would help users find relevant information quickly. Pay special attention to extracting Chinese proper nouns (e.g., names of shows, places, etc.), ensuring that they are correctly identified and used as the key terms in the rephrased question.
- Exclusions:
    - If the question is based on casual conversation or seeks personal opinions (e.g., “What do you think?” or “How are you today?”), output [skip].
    - If the question involves simple calculations that can be easily deduced (e.g., "What’s 2 + 2?"), or asks for information that has already been addressed in the chat history, output [skip].
    - If the question is conversational or rhetorical in nature (e.g., "I wonder why..."), output [skip].
- Knowledge Requests:
    - If the question is asking for a deeper explanation, new knowledge, or facts that are not directly provided in the chat history, rephrase it to clearly seek that information. Examples include technical inquiries, how-to guides, explanations of concepts, or requests for clarifications on complex topics.
    - If the question implies a need for a detailed answer or background knowledge (e.g., "How does AI work?" or "What are the steps to set up a server?"), ensure the question is clear, specific, and well-defined for a search engine query.
- Avoid Overly Broad Questions: Do not leave the question too vague or general. Instead of rephrasing something like “Tell me about technology,” focus on specific subtopics, such as “What are the latest advancements in AI technology?” or “How do quantum computers work?”
- Avoid Redundancy: If the question is too similar to one already covered in the conversation, or if it has already been answered directly, output [skip]. Avoid rephrasing questions that do not bring new or unique queries into the conversation.
- Maintain Context and Accuracy: Your rephrased question must directly reflect the user’s intended information-seeking query, based on the conversation so far. Do not make assumptions about what the user is asking for if it isn't clearly expressed.

IMPORTANT: Your rephrased question or [skip] MUST be in the same language as the original input. This is crucial for maintaining context, accuracy, and the user’s intent.

Chat History:
{chat_history}
Follow-up Input: {question}
Standalone Question or [skip]:
`
            )
    })
]).i18n({
    'zh-CN': require('./locales/zh-CN.schema.yml'),
    'en-US': require('./locales/en-US.schema.yml')
}) as Schema<Config>

export const inject = ['chatluna', 'puppeteer']

export const name = 'chatluna-search-service'
