import { Tool } from '@langchain/core/tools'
import { SearchManager } from '../provide'
import { PuppeteerBrowserTool } from './puppeteerBrowserTool'
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters'
import { MemoryVectorStore } from 'koishi-plugin-chatluna/llm-core/vectorstores'
import { Embeddings } from '@langchain/core/embeddings'
import { Document } from '@langchain/core/documents'
import { SearchResult, SummaryType } from '../types'

export class SearchTool extends Tool {
    name = 'web_search'

    // eslint-disable-next-line max-len
    description = `An search engine. Useful for when you need to answer questions about current events. Input should be a raw string of keyword. About Search Keywords, you should cut what you are searching for into several keywords and separate them with spaces. For example, "What is the weather in Beijing today?" would be "Beijing weather today"`

    private _textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 500,
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
        const documents = await this.fetchSearchResult(arg)

        if (this.searchManager.config.summaryType !== SummaryType.Speed) {
            return JSON.stringify(
                documents.map((document) =>
                    Object.assign({}, document.metadata as SearchResult, {
                        content: document.pageContent
                    })
                )
            )
        }

        return JSON.stringify(this._reRankDocuments(arg, documents))
    }

    private async fetchSearchResult(query: string) {
        const results = await this.searchManager.search(query)

        if (this.searchManager.config.summaryType === SummaryType.Quality) {
            return await Promise.all(
                results.map(async (result, k) => {
                    let pageContent = result.description

                    if (pageContent == null || pageContent.length < 500) {
                        const browserContent: string =
                            await this.browserTool.invoke({
                                url: result.url,
                                action: 'summary'
                            })

                        if (
                            !browserContent.includes('Error getting page text:')
                        ) {
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
                                    }) as Document
                            )
                        })

                    return chunks
                })
            ).then((documents) => documents.flat())
        } else if (
            this.searchManager.config.summaryType === SummaryType.Balanced
        ) {
            return await Promise.all(
                results.map(async (result, k) => {
                    let pageContent = result.description

                    if (pageContent == null || pageContent.length < 500) {
                        const browserContent: string =
                            await this.browserTool.invoke({
                                url: result.url,
                                action: 'text'
                            })

                        if (
                            !browserContent.includes('Error getting page text:')
                        ) {
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
                                    }) as Document
                            )
                        })

                    return chunks
                })
            ).then((documents) => documents.flat())
        }

        return results.map(
            (result) =>
                ({
                    pageContent: result.description,
                    metadata: result
                }) as Document
        )
    }

    private async _reRankDocuments(query: string, documents: Document[]) {
        const vectorStore = new MemoryVectorStore(this.embeddings)
        await vectorStore.addDocuments(documents)

        const searchResult = await vectorStore.similaritySearchWithScore(
            query,
            this.searchManager.config.topK * 3
        )

        return searchResult
            .filter(
                (result) =>
                    result[1] > this.searchManager.config.searchThreshold
            )
            .map((result) => result[0].metadata as SearchResult)
            .slice(0, this.searchManager.config.topK)
    }
}
