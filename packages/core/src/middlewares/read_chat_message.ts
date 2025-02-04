import { Context, h } from 'koishi'
import { ChainMiddlewareRunStatus, ChatChain } from '../chains/chain'
import { Config } from '../config'
import { logger } from '../index'
import type {} from '@initencounter/sst'

export function apply(ctx: Context, config: Config, chain: ChatChain) {
    chain
        .middleware('read_chat_message', async (session, context) => {
            let message =
                context.command != null ? context.message : session.elements

            message = message as h[] | string

            if (typeof message === 'string') {
                message = [h.text(message)]
            }

            const transformedMessage =
                await ctx.chatluna.messageTransformer.transform(
                    session,
                    message
                )

            if (transformedMessage.content.length < 1) {
                return ChainMiddlewareRunStatus.STOP
            }

            context.options.inputMessage = transformedMessage

            return ChainMiddlewareRunStatus.CONTINUE
        })
        .after('lifecycle-prepare')
        .before('resolve_room')

    ctx.chatluna.messageTransformer.intercept(
        'text',
        async (session, element, message) => {
            message.content += element.attrs['content']
        }
    )

    ctx.chatluna.messageTransformer.intercept(
        'at',
        async (session, element, message) => {
            const name = element.attrs['name']
            const id = element.attrs['id']

            if (name && id !== session.bot.selfId) {
                message.content += `<at name="${name}" id="${id}"/>`
            }
        }
    )

    ctx.chatluna.messageTransformer.intercept(
        'img',
        async (session, element, message) => {
            const images: string[] = message.additional_kwargs.images ?? []

            const url = (element.attrs.url ?? element.attrs.src) as string

            logger.debug(`image url: ${url}`)

            const readImage = async (url: string) => {
                const response = await ctx.http(url, {
                    responseType: 'arraybuffer',
                    method: 'get',
                    headers: {
                        'User-Agent':
                            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
                    }
                })

                // support any text
                let ext = url.match(/\.([^.]*)$/)?.[1]

                if (!['png', 'jpeg'].includes(ext)) {
                    ext = 'jpeg'
                }

                const buffer = response.data

                const base64 = Buffer.from(buffer).toString('base64')

                images.push(`data:image/${ext ?? 'jpeg'};base64,${base64}`)
            }

            if (url.startsWith('data:image') && url.includes('base64')) {
                images.push(url)
            } else {
                try {
                    await readImage(url)
                } catch (error) {
                    logger.warn(`read image error: ${error}`)
                    return
                }
            }

            message.additional_kwargs.images = images

            if (message.content?.length < 1) {
                message.content = '[image]'
            }
        }
    )

    ctx.inject(['sst'], (ctx) => {
        logger.debug('sst service loaded.')

        ctx.effect(() =>
            ctx.chatluna.messageTransformer.intercept(
                'audio',
                async (session, element, message) => {
                    // The sst service only use session
                    const content = await ctx.sst.audio2text(session)
                    logger.debug(`audio2text: ${content}`)
                    message.content += content
                }
            )
        )
    })
}

declare module '../chains/chain' {
    export interface ChainMiddlewareName {
        read_chat_message: string
    }
}
