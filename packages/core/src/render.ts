import { Context } from 'koishi'
import {
    ChatLunaError,
    ChatLunaErrorCode
} from 'koishi-plugin-chatluna/utils/error'
import { Config } from './config'
import { Message, RenderMessage, RenderOptions, RenderType } from './types'

export abstract class Renderer {
    constructor(
        protected readonly ctx: Context,
        protected readonly config: Config
    ) {}

    abstract render(
        message: Message,
        options: RenderOptions
    ): Promise<RenderMessage>
}

export class DefaultRenderer {
    defaultOptions: RenderOptions

    private allRenderers: Record<string, Renderer> = {}

    constructor(
        protected readonly ctx: Context,
        protected readonly config: Config
    ) {
        this.defaultOptions = {
            type: config.outputMode as RenderType,
            split: config.splitMessage
        }
    }

    public async render(
        message: Message,
        options: RenderOptions = this.defaultOptions
    ): Promise<RenderMessage[]> {
        try {
            const result: RenderMessage[] = []

            const currentRenderer = await this._getRenderer(options.type)
            const rawRenderer =
                options.type === 'raw'
                    ? currentRenderer
                    : await this._getRenderer('raw')

            result.push(await currentRenderer.render(message, options))

            if (message.additionalReplyMessages) {
                for (const additionalMessage of message.additionalReplyMessages) {
                    result.push(
                        await rawRenderer.render(additionalMessage, options)
                    )
                }
            }

            return result
        } catch (e) {
            throw new ChatLunaError(ChatLunaErrorCode.RENDER_ERROR, e)
        }
    }

    private async _getRenderer(type: string): Promise<Renderer> {
        let renderer = this.allRenderers[type]

        if (renderer) {
            return renderer
        }

        const importRenderer = await import(`./renders/${type}.ts`)
        // eslint-disable-next-line new-cap
        renderer = new importRenderer.default(this.ctx, this.config)

        this.allRenderers[type] = renderer

        return renderer
    }
}
