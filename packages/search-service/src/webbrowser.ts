import { CallbackManagerForToolRun } from '@langchain/core/callbacks/manager'
import { Document } from '@langchain/core/documents'
import { Embeddings } from '@langchain/core/embeddings'
import { BaseLanguageModel } from '@langchain/core/language_models/base'
import { AIMessageChunk } from '@langchain/core/messages'
import { Tool, ToolParams } from '@langchain/core/tools'
import * as cheerio from 'cheerio'
import { chatLunaFetch, randomUA } from 'koishi-plugin-chatluna/utils/request'
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter'
import { MemoryVectorStore } from 'langchain/vectorstores/memory'
import { Response } from 'undici/types/fetch'
import { logger } from '.'

export const parseInputs = (inputs: string): [string, string] => {
    const [baseUrl, task] = inputs.split(',').map((input) => {
        let t = input.trim()
        t = t.startsWith('"') ? t.slice(1) : t
        t = t.endsWith('"') ? t.slice(0, -1) : t
        // it likes to put / at the end of urls, wont matter for task
        t = t.endsWith('/') ? t.slice(0, -1) : t
        return t.trim()
    })

    return [baseUrl, task]
}

export const getText = (
    html: string,
    baseUrl: string,
    summary: boolean
): string => {
    // scriptingEnabled so noscript elements are parsed
    const $ = cheerio.load(html, { scriptingEnabled: true })

    let text = ''

    // lets only get the body if its a summary, dont need to summarize header or footer etc
    const rootElement = summary ? 'body ' : '*'

    $(`${rootElement}:not(style):not(script):not(svg)`).each(
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (_i, elem: any) => {
            // we dont want duplicated content as we drill down so remove children
            let content = $(elem)
                .clone()
                .children()
                .remove()
                .end()
                .text()
                .trim()
            const $el = $(elem)

            // if its an ahref, print the content and url
            let href = $el.attr('href')
            if ($el.prop('tagName')?.toLowerCase() === 'a' && href) {
                if (!href.startsWith('http')) {
                    try {
                        href = new URL(href, baseUrl).toString()
                    } catch {
                        // if this fails thats fine, just no url for this
                        href = ''
                    }
                }

                const imgAlt = $el.find('img[alt]').attr('alt')?.trim()
                if (imgAlt) {
                    content += ` ${imgAlt}`
                }

                text += ` [${content}](${href})`
            }
            // otherwise just print the content
            else if (content !== '') {
                text += ` ${content}`
            }
        }
    )

    return text.trim().replace(/\n+/g, ' ')
}

const getHtml = async (baseUrl: string, h: Headers) => {
    // const domain = new URL(baseUrl).hostname

    const headers = { ...h }
    // these appear to be positional, which means they have to exist in the headers passed in
    //  headers.Host = domain;
    //   headers["Alt-Used"] = domain;

    let htmlResponse: Response
    try {
        htmlResponse = await chatLunaFetch(baseUrl, {
            headers
        })
    } catch (e) {
        logger.error(`Error fetching ${baseUrl}`, e)
        throw e
    }

    const allowedContentTypes = [
        'text/html',
        'application/json',
        'application/xml',
        'application/javascript',
        'text/plain'
    ]

    const contentType = htmlResponse.headers.get('content-type')
    const contentTypeArray = contentType.split(';')
    if (
        contentTypeArray[0] &&
        !allowedContentTypes.includes(contentTypeArray[0])
    ) {
        throw new Error('returned page was not utf8')
    }
    return await htmlResponse.text()
}

const DEFAULT_HEADERS = {
    Accept: 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    Referer: 'https://www.google.com/',
    Connection: 'keep-alive'
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Headers = Record<string, any>

export interface WebBrowserArgs extends ToolParams {
    model: BaseLanguageModel

    embeddings: Embeddings

    headers?: Headers
}

export class WebBrowser extends Tool {
    private _model: BaseLanguageModel

    private _embeddings: Embeddings

    private _headers: Headers

    constructor({
        model,
        headers,
        embeddings,
        verbose,
        callbacks
    }: WebBrowserArgs) {
        super({ verbose, callbacks })

        this._model = model
        this._embeddings = embeddings
        DEFAULT_HEADERS['User-Agent'] = randomUA()
        this._headers = headers || DEFAULT_HEADERS
    }

    /** @ignore */
    async _call(arg: string, runManager?: CallbackManagerForToolRun) {
        let url: string
        let task: string
        let rawContent = false

        try {
            const parsed = JSON.parse(arg) as {
                url: string
                task: string
                raw_content: boolean
            }
            url = parsed.url
            task = parsed.task
            rawContent = parsed.raw_content
        } catch (e) {
            ;[url, task] = parseInputs(arg)
        }
        const baseUrl = url
        const doSummary = !task

        let text: string
        try {
            const html = await getHtml(baseUrl, this._headers)
            text = getText(html, baseUrl, doSummary)
        } catch (e) {
            logger.error(`Error getting html for ${baseUrl}`)
            logger.error(e)

            if (e.cause) {
                logger.error(e.cause)
            }

            if (e) {
                return e.toString()
            }
            return 'There was a problem connecting to the site'
        }

        const textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: 2000,
            chunkOverlap: 200
        })
        const texts = await textSplitter.splitText(text)

        /* // if we want a summary grab first 4
        if (doSummary) {
            context = texts.slice(0, 3).join('\n')
        }
        // search term well embed and grab top 3
        else { */
        const docs = texts.map(
            (pageContent) =>
                new Document({
                    pageContent,
                    metadata: []
                })
        )

        const vectorStore = await MemoryVectorStore.fromDocuments(
            docs,
            this._embeddings
        )
        const results = await vectorStore.similaritySearch(task, 10)
        const context = results.map((res) => res.pageContent).join('\n')
        /* } */

        const input = `Text:${context}\n\nI need ${
            doSummary ? 'a summary' : task
            // eslint-disable-next-line max-len
        } from the above text, you need provide up to 5 markdown links from within that would be of interest (always including URL and text). Please ensure that the linked information is all within the text and that you do not falsely generate any information. Need output to Chinese. Links should be provided, if present, in markdown syntax as a list under the heading "Relevant Links:".`

        return rawContent
            ? text
            : this._model
                  .invoke(input)
                  .then((value: AIMessageChunk) => value.content)
    }

    name = 'web-browser'

    // eslint-disable-next-line max-len
    description = `useful for when you need to find something on or summarize a webpage. input should be a comma separated list of "ONE valid http URL including protocol","what you want to find on the page or empty string for a summary".`
}
