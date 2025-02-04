/* eslint-disable no-template-curly-in-string */
/* eslint-disable n/no-path-concat */
import { Marked, marked, Token } from 'marked'
import { createLogger } from 'koishi-plugin-chatluna/utils/logger'
import { readFileSync, writeFileSync } from 'fs'
import { Context, h, Logger, Schema } from 'koishi'
import markedKatex from 'marked-katex-extension'
import qrcode from 'qrcode'
import hljs from 'highlight.js'
import { markedHighlight } from 'marked-highlight'
import { chatLunaFetch } from 'koishi-plugin-chatluna/utils/request'
import type { Page } from 'puppeteer-core'
import { runAsyncTimeout } from 'koishi-plugin-chatluna/utils/promise'
import {
    Message,
    Renderer,
    RenderMessage,
    RenderOptions
} from 'koishi-plugin-chatluna'
import type {} from 'koishi-plugin-puppeteer'
import { Config } from '..'
import path from 'path'
import { escapeBrackets, randomArrayItem, renderTemplate } from './image'
import fs from 'fs/promises'

let logger: Logger

export class MixedImageRenderer extends Renderer {
    // eslint-disable-next-line @typescript-eslint/naming-convention
    private __page: Page

    private _marked: Marked

    constructor(
        protected readonly ctx: Context,
        protected readonly config: Config
    ) {
        super(ctx)
        logger = createLogger(ctx)

        this._marked = new Marked(
            markedKatex({
                throwOnError: false,
                displayMode: false,
                nonStandard: true,
                output: 'html'
            }),
            markedHighlight({
                langPrefix: 'hljs language-',
                //  langPrefix: 'hljs language-',
                highlight(code, lang) {
                    if (
                        code.match(/^sequenceDiagram/) ||
                        code.match(/^graph/) ||
                        lang === 'mermaid'
                    ) {
                        return '<pre class="mermaid">' + code + '</pre>'
                    }
                    return `<pre><code class="hljs">${
                        hljs.highlightAuto(code, [lang]).value
                    }</code></pre>`
                }
            })
        )

        ctx.on('dispose', async () => {
            await this.__page?.close()
        })
    }

    private async _page() {
        if (!this.__page) {
            this.__page = await this.ctx.puppeteer.page()
        }

        return this.__page
    }

    async render(
        message: Message,
        options: RenderOptions
    ): Promise<RenderMessage> {
        const elements: h[] = []
        const content = message.content

        // step 1: lex markdown

        const tokens = marked.lexer(content)

        // step 2: match text
        const matchedTexts = this._matchText(tokens)

        // step 3: merge the same type text

        const mergedMatchedTexts: MatchedText[] = []

        for (let i = 0; i < matchedTexts.length; i++) {
            const lastMatchedText =
                mergedMatchedTexts[mergedMatchedTexts.length - 1]

            const currentMatchedText = matchedTexts[i]

            if (lastMatchedText?.type === currentMatchedText.type) {
                mergedMatchedTexts.pop()
                mergedMatchedTexts.push({
                    type: currentMatchedText.type,
                    text: lastMatchedText.text + currentMatchedText.text
                })
            } else {
                mergedMatchedTexts.push(currentMatchedText)
            }
        }

        logger.debug(
            `mergedMatchedTexts: ${JSON.stringify(mergedMatchedTexts)}`
        )

        // step 4: render markdown to image

        for (const matchedText of mergedMatchedTexts) {
            if (matchedText.type === 'markdown') {
                const image = await this._renderMarkdownToImage(
                    matchedText.text
                )

                const element = h.image(image, 'image/png')

                if (options.split) {
                    elements.push(h('message', element))
                } else {
                    elements.push(element)
                }
            } else {
                if (options.split) {
                    // 自分段逻辑
                    matchedText.text.split('\n\n').forEach((text) => {
                        elements.push(h('message', h.text(text)))
                    })
                } else {
                    elements.push(h.text(matchedText.text))
                }
            }
        }

        return {
            element: elements
        }
    }

    private _matchText(tokens: Token[]): MatchedText[] {
        const currentMatchedTexts: MatchedText[] = []

        for (const token of tokens) {
            if (
                token.type === 'text' ||
                token.type === 'del' ||
                token.type === 'br'
            ) {
                currentMatchedTexts.push({
                    type: 'text',
                    text: token.raw
                })
            } else if (
                token.type === 'code' ||
                token.type === 'image' ||
                token.type === 'html' ||
                token.type === 'table'
            ) {
                currentMatchedTexts.push({
                    type: 'markdown',
                    text: token.raw
                })
            } else if (token.type === 'paragraph') {
                const matchedTexts = this._matchText(token.tokens)
                currentMatchedTexts.push(...matchedTexts)
            } else if (token.type === 'space') {
                const currentMatchedText =
                    currentMatchedTexts[currentMatchedTexts.length - 1]
                currentMatchedText.text = currentMatchedText.text + token.raw
            } else {
                currentMatchedTexts.length = 0

                currentMatchedTexts.push({
                    type: 'markdown',
                    text: tokens.map((token) => token.raw).join('')
                })

                break
            }
        }
        return currentMatchedTexts
    }

    private async _renderMarkdownToImage(
        markdownText: string
    ): Promise<Buffer> {
        const page = await this._page()

        const templateDir = path.resolve(
            this.ctx.baseDir,
            'data/chathub/render_template'
        )

        const templateHtmlPath = path.resolve(templateDir, 'template.html')
        const randomId = Math.random().toString(36).substring(2, 15)
        const outTemplateHtmlPath = path.resolve(
            templateDir,
            `${randomId}.html`
        )

        const templateHtml = readFileSync(templateHtmlPath).toString()

        let qrCode = ''

        if (this.config.qrCode) {
            qrCode = await runAsyncTimeout(
                this._textToQrcode(markdownText),
                7500,
                ''
            )
        }

        const content = await this._renderMarkdownToHtml(markdownText)
        // ${content} => markdownText'
        // eslint-disable-next-line no-template-curly-in-string
        const outTemplateHtml = renderTemplate(templateHtml, {
            content,
            qr_data: qrCode,
            background: randomArrayItem(this.config.background),
            blurAmount: this.config.blurAmount.toString(),
            backgroundMaskOpacity: this.config.backgroundMaskOpacity.toString()
        })

        writeFileSync(outTemplateHtmlPath, outTemplateHtml)

        await page.reload()
        await page.goto('file://' + outTemplateHtmlPath, {
            waitUntil: 'networkidle0',
            timeout: 30 * 1000
        })

        const app = await page.$('body')
        // screenshot

        const clip = await app.boundingBox()
        const result = await page.screenshot({ clip })

        this.ctx.setTimeout(async () => {
            // delete temp file
            await fs.unlink(outTemplateHtmlPath)
        }, 1000)

        return result
    }

    private async _renderMarkdownToHtml(text: string) {
        return await this._marked.parse(escapeBrackets(text), {
            gfm: true
        })
    }

    private async _textToQrcode(markdownText: string): Promise<string> {
        const response = await chatLunaFetch(
            'https://prod.pastebin.prod.webservices.mozgcp.net/api/',
            {
                method: 'POST',
                body: new URLSearchParams({
                    expires: '604800',
                    format: 'url',
                    lexer: '_markdown',
                    content: markdownText
                })
            }
        )

        const url = await response.text()
        logger.debug('pastebin url: ' + url)

        const qrcodeDataURL = await new Promise<string>((resolve, reject) => {
            qrcode.toDataURL(url, { errorCorrectionLevel: 'H' }, (err, url) => {
                if (err) {
                    reject(err)
                } else {
                    resolve(url)
                }
            })
        })

        return qrcodeDataURL
    }

    schema = Schema.const('mixed-image').i18n({
        'zh-CN': '同时输出图片和文本',
        'en-US': 'Output both image and text'
    })
}

interface MatchedText {
    type: 'text' | 'markdown'
    text: string
}
