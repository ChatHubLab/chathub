import { Message, RenderMessage, RenderOptions } from '../types'
import { Renderer } from './default'
import { marked, Token } from 'marked'
import { logger } from 'koishi-plugin-chatluna'
import { h, Schema } from 'koishi'
import type {} from 'koishi-plugin-puppeteer'
import { transformAndEscape } from './text'
import type {} from '@initencounter/vits'

export class MixedVoiceRenderer extends Renderer {
    async render(
        message: Message,
        options: RenderOptions
    ): Promise<RenderMessage> {
        const elements: h[] = []

        const renderText = (await this.renderText(message, options)).element

        if (renderText instanceof Array) {
            elements.push(...renderText)
        } else {
            elements.push(renderText)
        }

        const renderVoice = (await this.renderVoice(message, options)).element

        if (renderVoice instanceof Array) {
            elements.push(...renderVoice)
        } else {
            elements.push(renderVoice)
        }

        return {
            element: elements
        }
    }

    async renderText(
        message: Message,
        options: RenderOptions
    ): Promise<RenderMessage> {
        let transformed = transformAndEscape(message.content)

        if (options.split) {
            transformed = transformed.map((element) => {
                return h('message', element)
            })
        }

        return {
            element: transformed
        }
    }

    async renderVoice(
        message: Message,
        options: RenderOptions
    ): Promise<RenderMessage> {
        const splitMessages = this._splitMessage(message.content)
            .flatMap((text) => text.trim().split('\n\n'))
            .filter((text) => text.length > 0)

        logger?.debug(`splitMessages: ${JSON.stringify(splitMessages)}`)

        return {
            element: await this._renderToVoice(splitMessages.join(''), options)
        }
    }

    private _splitMessage(message: string): string[] {
        const tokens = renderTokens(marked.lexer(message))

        if (tokens.length === 0 || tokens[0].length === 0) {
            return [message]
        }

        return tokens
    }

    private _renderToVoice(text: string, options: RenderOptions) {
        return this.ctx.vits.say(
            Object.assign(
                {
                    speaker_id: options?.voice?.speakerId ?? 0,
                    input: text
                },
                {
                    session: options.session
                }
            )
        )
    }

    schema = Schema.const('mixed-voice').i18n({
        'zh-CN': '同时输出语音和文本',
        'en-US': 'Output both voice and text'
    })
}

function renderToken(token: Token): string {
    if (
        token.type === 'text' ||
        //     token.type === "space" ||
        token.type === 'heading' ||
        token.type === 'em' ||
        token.type === 'strong' ||
        token.type === 'del' ||
        token.type === 'codespan' ||
        token.type === 'list_item' ||
        token.type === 'blockquote'
        //   || token.type === "code"
    ) {
        return token.text
    }

    return token.raw
}

function renderTokens(tokens: Token[]): string[] {
    return tokens.map(renderToken)
}
