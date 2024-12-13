import { Context, Schema } from 'koishi'
import { SearchManager, SearchProvider } from '../provide'
import { SearchResult, SummaryType } from '../types'
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'
import { Config, createModel, logger } from '..'
import { ChatLunaChatModel } from 'koishi-plugin-chatluna/llm-core/platform/model'

// See https://github.com/langchain-ai/langchainjs/blob/fc21aa4df583a5e5de425b6b15f39a5014743bac/libs/langchain-community/src/tools/wikipedia_query_run.ts#L1

/**
 * Interface for the parameters that can be passed to the
 * WikipediaQueryRun constructor.
 */
export interface WikipediaQueryRunParams {
    topKResults?: number
    maxDocContentLength?: number
    baseUrl?: string
}

/**
 * Type alias for URL parameters. Represents a record where keys are
 * strings and values can be string, number, boolean, undefined, or null.
 */
type UrlParameters = Record<
    string,
    string | number | boolean | undefined | null
>

/**
 * Interface for the structure of search results returned by the Wikipedia
 * API.
 */
interface SearchResults {
    query: {
        search: {
            title: string
        }[]
    }
}

/**
 * Interface for the structure of a page returned by the Wikipedia API.
 */
interface Page {
    pageid: number
    ns: number
    title: string
    extract: string
}

/**
 * Interface for the structure of a page result returned by the Wikipedia
 * API.
 */
interface PageResult {
    batchcomplete: string
    query: {
        pages: Record<string, Page>
    }
}

class WikipediaSearchProvider extends SearchProvider {
    protected topKResults = 3

    protected maxDocContentLength = 5000

    protected baseUrl = 'https://en.wikipedia.org/w/api.php'

    constructor(
        ctx: Context,
        config: Config,
        plugin: ChatLunaPlugin,
        params: WikipediaQueryRunParams,
        private model: ChatLunaChatModel
    ) {
        super(ctx, config, plugin)

        this.topKResults = params.topKResults ?? this.topKResults
        this.maxDocContentLength =
            params.maxDocContentLength ?? this.maxDocContentLength
        this.baseUrl = params.baseUrl ?? this.baseUrl

        if (!model) {
            logger?.warn(
                'No summary model provided, skipping enhanced keywrod extract'
            )
        }
    }

    async search(
        query: string,
        limit = this.config.topK
    ): Promise<SearchResult[]> {
        if (this.model) {
            query = await this._extractKeyword(query)
            logger?.debug(`Extracted keyword For Wikipedia: ${query}`)
        }

        const searchResults = await this._fetchSearchResults(query)
        const summaries: SearchResult[] = []

        const topK = Math.min(limit, searchResults.query.search.length)

        const documentContentLength = (this.maxDocContentLength / topK) * 2

        for (let i = 0; i < topK; i += 1) {
            const page = searchResults.query.search[i].title

            try {
                const pageDetails = await this._fetchPage(page, true)

                if (!pageDetails) {
                    continue
                }
                const pageUrl = await this._getPageUrl(page)
                summaries.push({
                    title: page,
                    description: pageDetails.extract.slice(
                        0,
                        documentContentLength
                    ),
                    url: pageUrl
                })
            } catch (error) {
                logger?.error(`Failed to fetch page "${page}": ${error}`)
            }
        }

        if (summaries.length === 0) {
            return [
                {
                    title: 'No results found',
                    description: 'No good Wikipedia Search Result was found',
                    url: ''
                }
            ]
        }

        return summaries
    }

    private async _extractKeyword(query: string): Promise<string> {
        const result = await this.model.invoke(
            PROMPT.replace(/{query}/g, query)
        )
        return (result.content as string).trim()
    }

    /**
     * Fetches the content of a specific Wikipedia page. It returns the
     * extracted content as a string.
     * @param page The specific Wikipedia page to fetch its content.
     * @param redirect A boolean value to indicate whether to redirect or not.
     * @returns The extracted content of the specific Wikipedia page as a string.
     */
    public async content(page: string, redirect = true): Promise<string> {
        try {
            const result = await this._fetchPage(page, redirect)
            return result.extract
        } catch (error) {
            logger?.error(error)
            throw new Error(
                `Failed to fetch content for page "${page}": ${error}`
            )
        }
    }

    /**
     * Builds a URL for the Wikipedia API using the provided parameters.
     * @param parameters The parameters to be used in building the URL.
     * @returns A string representing the built URL.
     */
    protected buildUrl<P extends UrlParameters>(parameters: P): string {
        const nonUndefinedParams: [string, string][] = Object.entries(
            parameters
        )
            .filter(([_, value]) => value !== undefined)
            .map(([key, value]) => [key, `${value}`])
        const searchParams = new URLSearchParams(nonUndefinedParams)
        return `${this.baseUrl}?${searchParams}`
    }

    private async _getPageUrl(title: string): Promise<string> {
        const params = new URLSearchParams({
            action: 'query',
            prop: 'info',
            inprop: 'url',
            titles: title,
            format: 'json'
        })

        const response = await this._plugin.fetch(
            `${this.baseUrl}?${params.toString()}`
        )
        if (!response.ok) throw new Error('Network response was not ok')

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const data = (await response.json()) as any
        const pages = data.query.pages
        const pageId = Object.keys(pages)[0]
        return pages[pageId].fullurl
    }

    private async _fetchSearchResults(query: string): Promise<SearchResults> {
        const searchParams = new URLSearchParams({
            action: 'query',
            list: 'search',
            srsearch: query,
            format: 'json'
        })

        const response = await this._plugin.fetch(
            `${this.baseUrl}?${searchParams.toString()}`
        )
        if (!response.ok) throw new Error('Network response was not ok')

        const data = (await response.json()) as SearchResults

        return data
    }

    private async _fetchPage(page: string, redirect: boolean): Promise<Page> {
        const params = new URLSearchParams({
            action: 'query',
            prop: 'extracts',
            explaintext: 'true',
            redirects: redirect ? '1' : '0',
            format: 'json',
            titles: page
        })

        const response = await this._plugin.fetch(
            `${this.baseUrl}?${params.toString()}`
        )
        if (!response.ok) throw new Error('Network response was not ok')

        const data = (await response.json()) as PageResult
        const { pages } = data.query
        const pageId = Object.keys(pages)[0]

        return pages[pageId]
    }

    static schema = Schema.const('wikipedia').i18n({
        '': 'Wikipedia'
    })

    name = 'wikipedia'
}

const PROMPT = `Extract the most important single-word keyword for a Wikipedia search from the query. The keyword should match the query's language and represent the main subject likely to have a Wikipedia article.

Examples:
Query: "What are the health benefits of drinking green tea?"
Keyword: "tea"

Query: "长城是什么时候建造的？"
Keyword: "长城"

Query: "江戸時代の武士の生活について教えてください。"
Keyword: "武士"

Query: "{query}"
Keyword:`

export async function apply(
    ctx: Context,
    config: Config,
    plugin: ChatLunaPlugin,
    manager: SearchManager
) {
    if (!config.searchEngine.includes('wikipedia')) {
        return
    }

    const summaryModel =
        config.summaryType === SummaryType.Quality
            ? await createModel(ctx, config.summaryModel)
            : undefined

    const wikipediaBaseURLs = config.wikipediaBaseURL
    for (const baseURL of wikipediaBaseURLs) {
        manager.addProvider(
            new WikipediaSearchProvider(
                ctx,
                config,
                plugin,
                {
                    baseUrl: baseURL,
                    maxDocContentLength:
                        config.summaryType !== SummaryType.Balanced
                            ? config.maxWikipediaDocContentLength
                            : 100000
                },
                summaryModel
            )
        )
    }
}
