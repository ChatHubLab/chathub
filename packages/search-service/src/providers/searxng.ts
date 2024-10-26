import { Context, Schema } from 'koishi'
import { SearchManager, SearchProvider } from '../provide'
import { SearchResult } from '../types'
import { Config } from '..'
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'

class SearxNGSearchProvider extends SearchProvider {
    async search(query: string, limit = this.config.topK) {
        const searchUrl = new URL(this.config.searxngBaseURL)

        const params = {
            q: query,
            format: 'json'
        }

        Object.entries(params).forEach(([key, value]) => {
            searchUrl.searchParams.append(key, value)
        })

        const response = await this._plugin.fetch(searchUrl.toString())

        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`)
        }

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const res: any = await response.json()
        const results = res.results as {
            url: string
            title: string
            content: string
            thumbnail?: string
        }[]

        if (results.length === 0) {
            return [
                {
                    title: 'No results found',
                    description: 'No results found',
                    url: ''
                }
            ]
        }

        const snippets = results.map(
            (item): SearchResult => ({
                url: item.url,
                title: item.title,
                description: item.content,
                image: item.thumbnail
            })
        )

        return snippets.slice(0, limit)
    }

    static schema = Schema.const('searxng').i18n({
        '': 'SearxNG'
    })

    name = 'searxng'
}

export function apply(
    ctx: Context,
    config: Config,
    plugin: ChatLunaPlugin,
    manager: SearchManager
) {
    const searchEngines = config.searchEngine
    if (searchEngines.includes('searxng')) {
        manager.addProvider(new SearxNGSearchProvider(ctx, config, plugin))
    }
}
