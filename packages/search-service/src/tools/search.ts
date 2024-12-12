import { Tool } from '@langchain/core/tools'
import { SearchManager } from '../provide'
import { PuppeteerBrowserTool } from './puppeteerBrowserTool'

export class SearchTool extends Tool {
    name = 'web_search'

    // eslint-disable-next-line max-len
    description = `An search engine. Useful for when you need to answer questions about current events. Input should be a raw string of keyword. About Search Keywords, you should cut what you are searching for into several keywords and separate them with spaces. For example, "What is the weather in Beijing today?" would be "Beijing weather today"`

    constructor(
        private searchManager: SearchManager,
        private browserTool: PuppeteerBrowserTool
    ) {
        super({})
    }

    async _call(arg: string): Promise<string> {
        const results = await this.searchManager.search(arg)

        return JSON.stringify(results)
    }
}
