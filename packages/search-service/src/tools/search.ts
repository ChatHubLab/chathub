import { Tool } from '@langchain/core/tools'
import { SearchManager } from '../provide'
import { PuppeteerBrowserTool } from './puppeteerBrowserTool'
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters'
import { MemoryVectorStore } from 'koishi-plugin-chatluna/llm-core/vectorstores'
import { Embeddings } from '@langchain/core/embeddings'
import { Document } from '@langchain/core/documents'
import { SearchResult } from '../types'

export class SearchTool extends Tool {
    name = 'web_search'

    // eslint-disable-next-line max-len
    description = `An search engine. Useful for when you need to answer questions about current events. Input should be a raw string of keyword. About Search Keywords, you should cut what you are searching for into several keywords and separate them with spaces. For example, "What is the weather in Beijing today?" would be "Beijing weather today"`

    private _textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 300,
        chunkOverlap: 50
    })

    constructor(
        private searchManager: SearchManager,
        private browserTool: PuppeteerBrowserTool,
        private embeddings: Embeddings
    ) {
        super({})
    }

    async _call(arg: string): Promise<string> {
        let enableEnhancedSummary = this.searchManager.config.enhancedSummary

        if (arg.startsWith('$$character-')) {
            arg = arg.slice('$$character-'.length)
        } else {
            enableEnhancedSummary = false
        }

        const startSearchTime = performance.now() // Start timing the search
        const results = await this.searchManager.search(arg)
        const endSearchTime = performance.now() // End timing the search
        console.log(
            `Search time: ${((endSearchTime - startSearchTime) / 1000).toFixed(3)} ms`
        )

        if (enableEnhancedSummary) {
            return JSON.stringify(results)
        }

        const vectorStore = new MemoryVectorStore(this.embeddings)

        const promises = results.map(async (result, k) => {
            let pageContent = result.description

            if (pageContent == null || pageContent.length < 500) {
                const browserContent: string = await this.browserTool.invoke({
                    url: result.url,
                    action: 'text'
                })

                if (!browserContent.includes('Error getting page text:')) {
                    pageContent = browserContent
                }
            }

            if (pageContent == null) {
                return
            }

            const chunks = await this._textSplitter
                .splitText(pageContent)
                .then((chunks) => {
                    return chunks.map(
                        (chunk) =>
                            ({
                                pageContent: chunk,
                                metadata: result
                            }) satisfies Document
                    )
                })

            await vectorStore.addDocuments(chunks)
        })

        await Promise.all(promises)

        const searchResults = await vectorStore.similaritySearch(
            arg,
            this.searchManager.config.topK * 2
        )

        return JSON.stringify(
            searchResults.map((document) =>
                Object.assign({}, document.metadata as SearchResult, {
                    content: document.pageContent
                })
            )
        )
    }
}
