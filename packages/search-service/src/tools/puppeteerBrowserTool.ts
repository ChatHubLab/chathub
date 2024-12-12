/* eslint-disable max-len */
import { StructuredTool } from '@langchain/core/tools'
import { Context } from 'koishi'
import type { Page, PuppeteerLifeCycleEvent } from 'puppeteer-core'
import type {} from 'koishi-plugin-puppeteer'
import { BaseLanguageModel } from '@langchain/core/language_models/base'
import { Embeddings } from '@langchain/core/embeddings'
import { z } from 'zod'
import { LRUCache } from 'lru-cache'

export interface PuppeteerBrowserToolOptions {
    timeout?: number
    idleTimeout?: number
    waitUntil?: PuppeteerLifeCycleEvent
    fastMode?: boolean
}

export class PuppeteerBrowserTool extends StructuredTool {
    name = 'web_browser'
    description = `A powerful tool designed for seamless web browsing.
    Available actions:
    - open [url]: Open a web page (required first action)
    - summarize [search_text?]: Simple summarize the current page, optionally with a search text.
    - text [search_text?]: Get the content of the current page, optionally with a search text
    - select [selector]: Select content from a specific div
    - previous: Go to the previous page
    - get-html: Get the HTML content of the current page
    - get-structured-urls: Get structured URLs from the current page
    Every action must be input with the URL of the page. Like this: {{
        action: 'summarize',
        params: 'xxx',
        url: 'https://example.com'
    }}
    After using this tool, you must process the result before considering using it again in the next turn.`

    private pages: LRUCache<string, Page>
    private lastActionTime: number = Date.now()
    private readonly timeout: number = 30000 // 30 seconds timeout
    private readonly idleTimeout: number = 180000 // 5 minutes idle timeout
    private model: BaseLanguageModel
    private embeddings: Embeddings
    private ctx: Context
    private waitUntil: PuppeteerLifeCycleEvent
    private fastMode: boolean
    schema = z.object({
        action: z.string().describe('The action to perform'),
        params: z.string().optional().describe('The parameters for the action'),
        url: z.string().optional().describe('The URL to action on')
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
    }) as any

    private actions: Record<
        string,
        (url: string, params?: string) => Promise<string>
    > = {
        open: this.openPage.bind(this),
        summarize: this.summarizePage.bind(this),
        text: this.getPageText.bind(this),
        select: this.selectDiv.bind(this),
        previous: this.goToPreviousPage.bind(this),
        'get-html': this.getHtml.bind(this),
        'get-structured-urls': this.getStructuredUrls.bind(this)
    }

    constructor(
        ctx: Context,
        model: BaseLanguageModel,
        embeddings: Embeddings,
        options: PuppeteerBrowserToolOptions = {}
    ) {
        super()

        this.ctx = ctx
        this.model = model
        this.embeddings = embeddings
        this.timeout = options.timeout || this.timeout
        this.idleTimeout = options.idleTimeout || this.idleTimeout
        this.fastMode = options.fastMode || false

        this.pages = new LRUCache<string, Page>({
            max: 20,
            dispose: (value, key, reason) => {
                value.close().catch((err) => {
                    this.ctx.logger.error(
                        `Error closing page ${key}: ${err.message}`
                    )
                })
            }
        })

        this.waitUntil = options.waitUntil || this.waitUntil

        this.startIdleTimer()
    }

    async _call(input: {
        url: string
        action: string
        params: string
    }): Promise<string> {
        try {
            const { action, params, url } = input

            this.lastActionTime = Date.now()

            if (this.actions[action]) {
                return await this.actions[action](url, params)
            } else {
                return `Unknown action: ${action}. Available actions: ${Object.keys(this.actions).join(', ')}`
            }
        } catch (error) {
            if (error instanceof Error) {
                return `Error: ${error.message}`
            }
            return 'An unknown error occurred'
        }
    }

    private async getPage(url: string) {
        if (!this.pages.has(url)) {
            const puppeteer = this.ctx.puppeteer
            if (!puppeteer) {
                throw new Error('Puppeteer service is not available')
            }
            const page = await puppeteer.browser.newPage()
            await page.goto(url, {
                waitUntil: this.waitUntil,
                timeout: this.timeout
            })
            this.pages.set(url, page)
        } else {
            this.pages.get(url)
        }

        return this.pages.get(url)
    }

    private async openPage(url: string, params?: string): Promise<string> {
        try {
            await this.getPage(url ?? params)
            return 'Page opened successfully'
        } catch (error) {
            console.error(error)
            return `Error opening page: ${error.message}`
        }
    }

    private async summarizePage(
        url: string,
        searchText?: string
    ): Promise<string> {
        try {
            const text = await this.getPageText(url)
            if (text.includes('Error getting page text')) {
                return text
            }
            return this.summarizeText(text, searchText)
        } catch (error) {
            console.error(error)
            return `Error summarizing page: ${error.message}`
        }
    }

    private async getPageText(
        url: string,
        searchText?: string
    ): Promise<string> {
        try {
            const page = await this.getPage(url)
            if (!page) return 'No page is open, please use open action first'

            const text = await page.evaluate(() => {
                // fix esbuild
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                window['__name'] = (func: any) => func

                const findMainContent = () => {
                    const candidates: {
                        element: Element
                        score: number
                    }[] = []

                    // Helper to calculate text density
                    const getTextDensity = (element: Element) => {
                        const text = element.textContent || ''
                        const html = element.innerHTML
                        return text.length / (html.length || 1)
                    }

                    // Helper to check if element is likely navigation/header/footer
                    const isBoilerplate = (element: Element) => {
                        const className = element.className.toLowerCase()
                        const id = element.id.toLowerCase()
                        return /nav|header|footer|sidebar|comment|menu|copyright|related|recommend|advertisement|ad-|social|share/i.test(
                            `${className} ${id}`
                        )
                    }

                    // Helper to calculate hierarchical p-tag score
                    const getParagraphScore = (node: Element): number => {
                        let value = 0

                        for (const child of Array.from(node.children)) {
                            if (child.tagName.toLowerCase() === 'p') {
                                const text = child.textContent || ''
                                value += text.trim().length
                            } else {
                                value += getParagraphScore(child) * 0.5
                            }
                        }

                        return value
                    }

                    // Helper to calculate table content score
                    const getTableScore = (node: Element): number => {
                        let score = 0

                        // 计算表格内容的丰富度
                        const rows = node.querySelectorAll('tr').length
                        const cells = node.querySelectorAll('td, th').length

                        if (rows > 0 && cells > 0) {
                            // 基础分数：行数 * 单元格平均数
                            score += (cells / rows) * rows * 2

                            // 表头加分
                            const headers = node.querySelectorAll('th').length
                            if (headers > 0) score += headers * 5

                            // 表格标题加分
                            const caption = node.querySelector('caption')
                            if (caption) score += 10

                            // 内容丰富度加分
                            const textLength = node.textContent?.length || 0
                            if (textLength > 0) {
                                score += Math.min(textLength / 100, 50) // 最多加50分
                            }
                        }

                        return score
                    }

                    // Common content patterns
                    const contentPatterns = [
                        /article|post|content|main|body|text/i,
                        /^(article|main|content)$/i
                    ]

                    // Specific class/id scoring patterns
                    const specificPatterns = {
                        content:
                            /content|article-content|post-content|entry-content|main-content/i,
                        table: /table-content|data-table|grid|list/i,
                        article: /article|post|entry|blog/i,
                        main: /main|primary|central/i
                    }

                    // Score each potential content container
                    document
                        .querySelectorAll('div, article, main, section, table')
                        .forEach((element) => {
                            if (isBoilerplate(element)) return

                            let score = 0
                            const identifiers =
                                `${element.className} ${element.id}`.toLowerCase()

                            // 标签评分
                            const tagName = element.tagName.toLowerCase()
                            if (tagName === 'article') score += 30
                            if (tagName === 'main') score += 25
                            if (tagName === 'table') score += 15

                            // 特定类名/ID评分
                            Object.entries(specificPatterns).forEach(
                                ([key, pattern]) => {
                                    if (pattern.test(identifiers)) {
                                        switch (key) {
                                            case 'content':
                                                score += 40
                                                break
                                            case 'table':
                                                score += 25
                                                break
                                            case 'article':
                                                score += 30
                                                break
                                            case 'main':
                                                score += 20
                                                break
                                        }
                                    }
                                }
                            )

                            // 通用内容模式评分
                            contentPatterns.forEach((pattern) => {
                                if (pattern.test(identifiers)) score += 20
                            })

                            // 内容密度评分
                            const density = getTextDensity(element)
                            score += density * 50

                            // 段落评分
                            const paragraphs =
                                element.getElementsByTagName('p').length
                            score += paragraphs * 3

                            // 标题评分
                            const headings =
                                element.querySelectorAll(
                                    'h1,h2,h3,h4,h5,h6'
                                ).length
                            score += headings * 5

                            // 表格内容评分
                            if (
                                tagName === 'table' ||
                                element.querySelector('table')
                            ) {
                                score += getTableScore(element)
                            }

                            // 层级段落评分
                            const paragraphScore = getParagraphScore(element)
                            score += paragraphScore * 2

                            // 长度惩罚
                            const text = element.textContent || ''
                            if (text.length < 250) score *= 0.7

                            // 位置评分
                            const rect = element.getBoundingClientRect()
                            const verticalCenter = Math.abs(
                                0.5 -
                                    rect.top /
                                        document.documentElement.scrollHeight
                            )
                            score *= 1 - verticalCenter * 0.3

                            // 表格特定优化
                            if (
                                tagName === 'table' ||
                                element.querySelector('table')
                            ) {
                                // 如果是数据展示类的表格，降低文本长度惩罚
                                if (
                                    text.length < 250 &&
                                    element.querySelectorAll('td').length > 20
                                ) {
                                    score *= 1.5 // 补偿一些分数
                                }
                            }

                            candidates.push({ element, score })
                        })

                    // Return highest scoring element
                    candidates.sort((a, b) => b.score - a.score)
                    return candidates[0]?.element || document.body
                }

                const mainContent = findMainContent()
                let structuredText = ''

                const processNode = (node: Node, depth: number = 0) => {
                    if (node.nodeType === Node.TEXT_NODE) {
                        const trimmedText = node.textContent?.trim()
                        if (trimmedText) {
                            structuredText += ' ' + trimmedText
                        }
                    } else if (node.nodeType === Node.ELEMENT_NODE) {
                        const element = node as Element
                        const tagName = element.tagName.toLowerCase()

                        switch (tagName) {
                            case 'p':
                            case 'h1':
                            case 'h2':
                            case 'h3':
                            case 'h4':
                            case 'h5':
                            case 'h6':
                                structuredText += '\n'.repeat(depth > 0 ? 1 : 2)
                                structuredText += `${'#'.repeat(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'].indexOf(tagName) + 1)} `
                                for (const child of element.childNodes) {
                                    processNode(child, depth + 1)
                                }
                                structuredText += '\n'
                                break
                            case 'ul':
                            case 'ol':
                                structuredText += '\n'
                                for (const child of element.childNodes) {
                                    processNode(child, depth + 1)
                                }
                                structuredText += '\n'
                                break
                            case 'li':
                                if (element.textContent?.trim()) {
                                    structuredText +=
                                        '\n' + '  '.repeat(depth) + '- '
                                    for (const child of element.childNodes) {
                                        processNode(child, depth + 1)
                                    }
                                }
                                break
                            case 'br':
                                structuredText += '\n'
                                break
                            case 'strong':
                            case 'b':
                                structuredText += ` **${element.textContent?.trim()}** `
                                break
                            case 'em':
                            case 'i':
                                structuredText += ` *${element.textContent?.trim()}* `
                                break
                            case 'code':
                                structuredText += ` \`${element.textContent?.trim()}\` `
                                break
                            case 'pre':
                                structuredText +=
                                    '\n```\n' +
                                    element.textContent?.trim() +
                                    '\n```\n'
                                break
                            case 'blockquote':
                                structuredText +=
                                    '\n> ' +
                                    element.textContent
                                        ?.trim()
                                        .replace(/\n/g, '\n> ') +
                                    '\n'
                                break
                            case 'table':
                                structuredText += '\n'
                                for (const child of element.childNodes) {
                                    processNode(child, depth + 1)
                                }
                                structuredText += '\n'
                                break
                            case 'tr':
                                structuredText += '|'
                                for (const child of element.childNodes) {
                                    processNode(child, depth + 1)
                                }
                                structuredText += '\n'
                                break
                            case 'th':
                            case 'td':
                                structuredText += ` ${element.textContent?.trim()} |`
                                break
                            case 'mark':
                            case 'u':
                                structuredText += ` __${element.textContent?.trim()}__ `
                                break
                            case 'del':
                            case 's':
                                structuredText += ` ~~${element.textContent?.trim()}~~ `
                                break
                            case 'sup':
                                structuredText += `^${element.textContent?.trim()}`
                                break
                            case 'sub':
                                structuredText += `~${element.textContent?.trim()}`
                                break
                            case 'kbd':
                                structuredText += ` <kbd>${element.textContent?.trim()}</kbd> `
                                break
                            case 'cite':
                            case 'dfn':
                                structuredText += ` *${element.textContent?.trim()}* `
                                break
                            case 'span': {
                                const className = element.className

                                if (className.includes('highlight')) {
                                    structuredText += ` **${element.textContent?.trim()}** `
                                } else if (className.includes('italic')) {
                                    structuredText += ` *${element.textContent?.trim()}* `
                                } else {
                                    structuredText += ` ${element.textContent?.trim()} `
                                }
                                break
                            }
                            case 'abbr': {
                                const title = element.getAttribute('title')
                                structuredText += title
                                    ? ` ${element.textContent?.trim()} (${title})`
                                    : ` ${element.textContent?.trim()}`
                                break
                            }
                            case 'q':
                                structuredText += ` "${element.textContent?.trim()}" `
                                break
                            case 'time': {
                                const datetime =
                                    element.getAttribute('datetime')
                                structuredText += datetime
                                    ? ` ${element.textContent?.trim()} [${datetime}]`
                                    : ` ${element.textContent?.trim()}`
                                break
                            }
                            case 'details':
                                structuredText += '\n<details>\n'
                                for (const child of element.childNodes) {
                                    processNode(child, depth + 1)
                                }
                                structuredText += '\n</details>\n'
                                break
                            case 'summary':
                                structuredText += '<summary>'
                                for (const child of element.childNodes) {
                                    processNode(child, depth + 1)
                                }
                                structuredText += '</summary>\n'
                                break
                            case 'figure':
                                structuredText += '\n'
                                for (const child of element.childNodes) {
                                    processNode(child, depth + 1)
                                }
                                structuredText += '\n'
                                break
                            case 'figcaption':
                                structuredText += `\n_${element.textContent?.trim()}_\n`
                                break
                            case 'hr':
                                structuredText += '\n---\n'
                                break
                            case 'dl':
                                structuredText += '\n'
                                for (const child of element.childNodes) {
                                    processNode(child, depth + 1)
                                }
                                structuredText += '\n'
                                break
                            case 'dt':
                                structuredText += `\n**${element.textContent?.trim()}**`
                                break
                            case 'dd':
                                structuredText += `: ${element.textContent?.trim()}\n`
                                break
                            case 'var':
                                structuredText += ` _${element.textContent?.trim()}_ `
                                break
                            case 'samp':
                                structuredText += ` \`${element.textContent?.trim()}\` `
                                break
                            default:
                                if (
                                    tagName !== 'script' &&
                                    tagName !== 'style' &&
                                    tagName !== 'meta'
                                ) {
                                    for (const child of element.childNodes) {
                                        processNode(child, depth)
                                    }
                                }
                        }
                    }
                }

                // 添加相关链接部分
                const getRelatedLinks = (content: Element) => {
                    const currentUrl = window.location.href
                    const currentHost = window.location.hostname
                    const currentPath = window.location.pathname

                    interface LinkGroup {
                        samePath: string[]
                        sameHost: string[]
                        external: string[]
                    }

                    const links: LinkGroup = {
                        samePath: [],
                        sameHost: [],
                        external: []
                    }

                    // 获取链接的上下文（前后50个字符）
                    const getLinkContext = (link: Element): string => {
                        const parent = link.parentElement
                        if (!parent) return ''

                        const text = parent.textContent || ''
                        const linkText = link.textContent || ''
                        const linkPos = text.indexOf(linkText)

                        if (linkPos === -1) return ''

                        const start = Math.max(0, linkPos - 50)
                        const end = Math.min(
                            text.length,
                            linkPos + linkText.length + 50
                        )

                        return text.slice(start, end).trim()
                    }

                    content.querySelectorAll('a[href]').forEach((link) => {
                        const href = link.getAttribute('href')
                        if (!href) return

                        try {
                            const url = new URL(href, currentUrl)
                            const linkText = link.textContent?.trim()
                            const context = getLinkContext(link)

                            // 忽略空链接、锚点链接和常见功能性链接
                            if (
                                !linkText ||
                                url.href === currentUrl ||
                                href.startsWith('#') ||
                                /login|signup|register|cart|search|account/i.test(
                                    url.pathname
                                )
                            ) {
                                return
                            }

                            const linkMd = `- [${linkText}](${url.href})${context ? `\n  > ${context}` : ''}`

                            if (url.hostname === currentHost) {
                                if (
                                    url.pathname.startsWith(currentPath) ||
                                    currentPath.startsWith(url.pathname)
                                ) {
                                    if (!links.samePath.includes(linkMd)) {
                                        links.samePath.push(linkMd)
                                    }
                                } else {
                                    if (!links.sameHost.includes(linkMd)) {
                                        links.sameHost.push(linkMd)
                                    }
                                }
                            } else {
                                if (!links.external.includes(linkMd)) {
                                    links.external.push(linkMd)
                                }
                            }
                        } catch (e) {
                            // 忽略无效链接
                        }
                    })

                    let relatedLinksText = ''

                    // 只有当有链接时才添加标题
                    if (
                        links.samePath.length > 0 ||
                        links.sameHost.length > 0 ||
                        links.external.length > 0
                    ) {
                        relatedLinksText = '\n\n## Related Links\n\n'

                        if (links.samePath.length > 0) {
                            relatedLinksText +=
                                '### Same Section\n' +
                                links.samePath.slice(0, 2).join('\n') +
                                '\n\n'
                        }

                        if (links.sameHost.length > 0) {
                            relatedLinksText +=
                                '### Same Site\n' +
                                links.sameHost.slice(0, 2).join('\n') +
                                '\n\n'
                        }

                        if (links.external.length > 0) {
                            relatedLinksText +=
                                '### External References\n' +
                                links.external.slice(0, 2).join('\n') +
                                '\n\n'
                        }
                    }

                    return relatedLinksText
                }

                // 处理主要内容
                processNode(mainContent)

                const findBestLinkContainer = (
                    mainContent: Element
                ): Element => {
                    const MAX_PARENT_DEPTH = 6 // 限制最大父级查找深度
                    let current: Element | null = mainContent

                    for (let depth = 0; depth < MAX_PARENT_DEPTH; depth++) {
                        if (!current) break

                        current = current.parentElement
                    }

                    return mainContent // 如果没找到，返回原始内容
                }

                // 使用方式
                structuredText += getRelatedLinks(
                    findBestLinkContainer(mainContent)
                )

                return structuredText.trim().replace(/\n{3,}/g, '\n\n')
            })

            // 去除空行，并去除首尾空行，还去除多空格为单空格
            return text
                .trim()
                .replace(/\n{3,}/g, '\n\n')
                .trim()
        } catch (error) {
            console.error(error)
            return `Error getting page text: ${error.message}`
        }
    }

    private async summarizeText(
        text: string,
        searchText?: string
    ): Promise<string> {
        try {
            const input = `Text: ${text}

Please provide a comprehensive summary of the above text${searchText ? `, focusing on "${searchText}"` : ''}. Your summary should include:

1. Overview of main topics or themes (1 paragraph)
2. Breakdown of key points, arguments, or findings (4-5 paragraphs)
3. Supporting evidence, data, or examples (2-3 paragraphs)
4. Contrasting viewpoints or limitations (1 paragraph, if applicable)
5. Conclusions drawn from the main points (1 paragraph)

Guidelines:
 - Organize content into clear, logical paragraphs
 - Maintain an objective tone, avoiding bias
 - Use transitional phrases for smooth flow
 - Include relevant quotes or statistics
 - Incorporate up to 5 important links from the text
 - Ensure all information is accurate and derived from the text
 - Use the same language as the input text; do not translate.

Aim for a balanced, informative summary that provides a comprehensive understanding of the original content.

CRITICAL: Your summary MUST be in the same language as the original text. Do not translate or change the language under any circumstances.`

            const summary = await this.model.invoke(input)
            return summary.content
        } catch (error) {
            console.error(error)
            return `Error summarizing text: ${error.message}`
        }
    }

    private async selectDiv(url: string, selector: string): Promise<string> {
        try {
            const page = await this.getPage(url)
            if (!page) return 'No page is open'
            const content = await page.evaluate((sel) => {
                const element = document.querySelector(sel)
                return element ? element.textContent : 'Element not found'
            }, selector)
            return content || 'No content found'
        } catch (error) {
            console.error(`Error selecting div: ${error}`)
            return `Error selecting div: ${error.message}`
        }
    }

    private async goToPreviousPage(url: string): Promise<string> {
        try {
            const page = await this.getPage(url)
            if (!page) return 'No page is open'
            await page.goBack({
                waitUntil: 'networkidle2',
                timeout: this.timeout
            })
            return 'Navigated to previous page'
        } catch (error) {
            console.error(`Error navigating to previous page: ${error.message}`)
            return `Error navigating to previous page: ${error}`
        }
    }

    private async getHtml(url: string): Promise<string> {
        try {
            const page = await this.getPage(url)
            if (!page) return 'No page is open'
            return await page.content()
        } catch (error) {
            console.error(error)
            return `Error getting HTML: ${error.message}`
        }
    }

    private async getStructuredUrls(url: string): Promise<string> {
        try {
            const page = await this.getPage(url)
            if (!page) return 'No page is open'
            return await page.evaluate(() => {
                const urlStructure: { [key: string]: string[] } = {
                    search: [],
                    navigation: [],
                    external: [],
                    other: []
                }

                const currentHost = window.location.hostname

                document.querySelectorAll('a').forEach((a) => {
                    const href = a.href
                    if (!href) return

                    const url = new URL(href)
                    const linkText = a.textContent?.trim() || ''

                    if (url.hostname === currentHost) {
                        if (
                            url.pathname.includes('search') ||
                            url.search.includes('q=')
                        ) {
                            urlStructure.search.push(`${linkText}: ${href}`)
                        } else if (
                            a.closest('nav') ||
                            a.matches('header a, footer a')
                        ) {
                            urlStructure.navigation.push(`${linkText}: ${href}`)
                        } else {
                            urlStructure.other.push(`${linkText}: ${href}`)
                        }
                    } else {
                        urlStructure.external.push(`${linkText}: ${href}`)
                    }
                })

                return JSON.stringify(urlStructure, null, 2)
            })
        } catch (error) {
            console.error(error)
            return `Error getting structured URLs: ${error.message}`
        }
    }

    private startIdleTimer() {
        this.ctx.setInterval(() => {
            if (Date.now() - this.lastActionTime > this.idleTimeout) {
                this.closeBrowser()
            }
        }, 60000) // Check every minute
        this.ctx.on('dispose', async () => {
            this.closeBrowser()
        })
    }

    async closeBrowser() {
        try {
            if (this.pages) {
                this.pages.clear()
            }
        } catch (error) {
            this.ctx.logger.error(error)
        }
    }
}
