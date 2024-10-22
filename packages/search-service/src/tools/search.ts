import { Tool } from '@langchain/core/tools'
import { SearchManager } from '../provide'

export class SearchTool extends Tool {
    name = 'web_search'

    // eslint-disable-next-line max-len
    description = `a search engine. useful for when you need to answer questions about current events. input should be a raw string of keyword. About Search Keywords, you should cut what you are searching for into several keywords and separate them with spaces. For example, "What is the weather in Beijing today?" would be "Beijing weather today"`

    constructor(private searchManager: SearchManager) {
        super({})
    }

    async _call(arg: string): Promise<string> {
        const results = await this.searchManager.search(arg)

        return JSON.stringify(results)
    }
}
