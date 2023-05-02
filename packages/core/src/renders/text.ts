import { Message, RenderMessage, RenderOptions, SimpleMessage } from '../types';
import { buildTextElement } from '../chat';
import { Renderer } from '../render';
import { transform } from 'koishi-plugin-markdown';
import { h } from 'koishi';

export default class TextRenderer extends Renderer {

    async render(message: SimpleMessage, options: RenderOptions): Promise<RenderMessage> {

        let transformed = transform(message.content)



        if (options.split) {

            transformed = transformed.map((element) => {
                return h("message", element)
            })
        }

        return {
            element: transformed
        }
    }
}