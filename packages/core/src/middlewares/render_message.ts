// eslint-disable-next-line @typescript-eslint/no-unused-vars
import { Context, Element } from 'koishi'
import { Config } from '../config'
import { ChainMiddlewareRunStatus, ChatChain } from '../chains/chain'
import { Message, RenderOptions } from '../types'
import { DefaultRenderer } from '../render'

let renderer: DefaultRenderer

export function apply(ctx: Context, config: Config, chain: ChatChain) {
    renderer = new DefaultRenderer(ctx, config)

    chain
        .middleware('render_message', async (session, context) => {
            if (context.options.responseMessage == null) {
                return ChainMiddlewareRunStatus.SKIPPED
            }

            return await renderMessage(
                context.options.responseMessage,
                context.options.renderOptions
            )
        })
        .after('lifecycle-send')
}

export async function renderMessage(message: Message, options?: RenderOptions) {
    return (await renderer.render(message, options)).map((message) => {
        const elements = message.element
        if (elements instanceof Array) {
            return elements
        } else {
            return [elements]
        }
    })
}

export async function markdownRenderMessage(text: string) {
    const elements = await renderMessage(
        {
            content: text
        },
        {
            type: 'text'
        }
    )

    return elements[0]
}

declare module '../chains/chain' {
    interface ChainMiddlewareName {
        render_message: never
    }

    interface ChainMiddlewareContextOptions {
        renderOptions?: RenderOptions
    }
}
