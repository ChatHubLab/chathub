/* eslint-disable no-template-curly-in-string */
import { Marked } from 'marked'
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
import fs from 'fs/promises'

let logger: Logger

export class ImageRenderer extends Renderer {
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
                macros: {
                    '\\ge': '\\geqslant',
                    '\\le': '\\leqslant',
                    '\\geq': '\\geqslant',
                    '\\leq': '\\leqslant'
                },
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
        const markdownText = message.content
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
                this._textToQRCode(markdownText),
                7500,
                ''
            )
        }

        // ${content} => markdownText'
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
            timeout: 40 * 1000
        })

        const app = await page.$('body')
        // screenshot

        const clip = await app.boundingBox()
        const screenshot = await page.screenshot({ clip })

        this.ctx.setTimeout(async () => {
            // delete temp file
            await fs.unlink(outTemplateHtmlPath)
        }, 1000)

        return {
            element: h.image(screenshot, 'image/png')
        }
    }

    private async _renderMarkdownToHtml(text: string) {
        return await this._marked.parse(escapeBrackets(text), {
            gfm: true
        })
    }

    private async _textToQRCode(markdownText: string): Promise<string> {
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

    schema = Schema.const('image').i18n({
        'zh-CN': '将回复渲染为图片',
        'en-US': 'Render as image'
    })
}

export function renderTemplate(template: string, data: Record<string, string>) {
    return template.replace(/\${(.*?)}/g, (_, key) => data[key] || '')
}

export function randomArrayItem<T>(array: T[]): T {
    return array[Math.floor(Math.random() * array.length)]
}

export function escapeBrackets(text: string) {
    // 正则表达式匹配 LaTeX 的行内公式 \(...\) 和块级公式 \[...\]

    const pattern = /\\\[([\s\S]*?[^\\])\\\]|\\\((.*?)\\\)/gm

    const result = text.replace(
        pattern,
        (match: string, squareContent: string, roundContent: string) => {
            if (squareContent) {
                // 块级公式 \[...\] 替换为 $$...$$
                return `$$${squareContent}$$`
            } else if (roundContent) {
                // 行内公式 \(...\) 替换为 $...$
                return `$${roundContent}$`
            }
            return match
        }
    )

    return result
}
