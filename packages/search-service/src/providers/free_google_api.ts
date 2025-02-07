import { Context, Schema } from 'koishi'
import { SearchManager, SearchProvider } from '../provide'
import { SearchResult } from '../types'
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'
import { Config } from '..'

class FreeGoogleSearchProvider extends SearchProvider {
    async search(
        query: string,
        limit = this.config.topK
    ): Promise<SearchResult[]> {
        const baseURL = this.config.freeSearchBaseURL

        const response = await this._plugin.fetch(
            `${baseURL}/search?keyword=${encodeURIComponent(query)}&count=${limit}`,
            {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    'User-Agent':
                        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
                }
            }
        )

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const res = (await response.json()) as SearchResult[]

        if (!res || res.length === 0) {
            return [
                {
                    title: 'No results found',
                    description: 'No good search result found',
                    url: ''
                }
            ]
        }

        return res.slice(0, limit)
    }

    static schema = Schema.const('free-google-api').i18n({
        '': 'Free Google (API & Web)'
    })

    name = 'free-google-api'
}

export function apply(
    ctx: Context,
    config: Config,
    plugin: ChatLunaPlugin,
    manager: SearchManager
) {
    if (config.searchEngine.includes('free-google-api')) {
        manager.addProvider(new FreeGoogleSearchProvider(ctx, config, plugin))
    }
}
