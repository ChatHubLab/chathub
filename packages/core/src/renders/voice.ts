import { Message, RenderMessage, RenderOptions } from '../types'
import { Renderer } from './default'
import { marked, Token } from 'marked'
import { logger } from 'koishi-plugin-chatluna'
import { h, Schema } from 'koishi'
import type {} from '@initencounter/vits'

export class VoiceRenderer extends Renderer {
    async render(
        message: Message,
        options: RenderOptions
    ): Promise<RenderMessage> {
        const splitMessages = this._splitMessage(message.content)
            .flatMap((text) => text.trim().split('\n\n'))
            .filter((text) => text.length > 0)

        logger?.debug(`splitMessages: ${JSON.stringify(splitMessages)}`)

        if (options.split) {
            return {
                element: await Promise.all(
                    splitMessages.map(async (text) => {
                        return h(
                            'message',
                            await this._renderToVoice(text, options)
                        )
                    })
                )
            }
        } else {
            return {
                element: await this._renderToVoice(
                    splitMessages.join(''),
                    options
                )
            }
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

    schema = Schema.const('voice').i18n({
        'zh-CN': '将回复渲染为语音',
        'en-US': 'Render as voice'
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
        token.type === 'blockquote' ||
        token.type === 'code'
    ) {
        return token.text
    }

    return token.raw
}

function renderTokens(tokens: Token[]): string[] {
    return tokens.map(renderToken)
}
