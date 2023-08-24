import { Context } from "koishi";
import { Config } from "./config";
import { RenderMessage, RenderOptions, RenderType, Message } from "./types";


export abstract class Renderer {
    constructor(protected readonly ctx: Context, protected readonly config: Config) { }

    abstract render(message: Message, options: RenderOptions): Promise<RenderMessage>
}

export class DefaultRenderer {
    defaultOptions: RenderOptions;

    private allRenderers: Record<string, Renderer> = {}

    constructor(protected readonly ctx: Context, protected readonly config: Config) {
        this.defaultOptions = {
            type: config.outputMode as RenderType,
            split: config.splitMessage
        };
    }

    public async render(
        message: Message,
        options: RenderOptions = this.defaultOptions
    ): Promise<RenderMessage[]> {
        const result: RenderMessage[] = [];

        const currentRenderer = await this._getRenderer(options.type);
        const rawRenderer = options.type === "raw" ? currentRenderer : await this._getRenderer("raw");

        result.push(await currentRenderer.render(message, options));

        if (message.additionalReplyMessages) {
            for (const additionalMessage of message.additionalReplyMessages) {
                result.push(await rawRenderer.render(additionalMessage, options));
            }
        }

        return result;
    }

    private async _getRenderer(type: string): Promise<Renderer> {
        let renderer = this.allRenderers[type];

        if (renderer) {
            return renderer;
        }

        const importRenderer = await require(`./renders/${type}.js`)
        renderer = new importRenderer.default(this.ctx, this.config);

        this.allRenderers[type] = renderer;

        return renderer;
    }

}

