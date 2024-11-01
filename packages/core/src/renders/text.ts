import { Message, RenderMessage, RenderOptions } from '../types'
import { Renderer } from './default'
import { transform } from 'koishi-plugin-markdown'
import { h, Schema } from 'koishi'
import he from 'he'

export class TextRenderer extends Renderer {
    async render(
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

    schema = Schema.const('text').i18n({
        'zh-CN': '将回复作为 markdown 进行渲染',
        'en-US': 'Render as markdown'
    })
}

function unescape(element: h): h {
    if (element.type === 'text') {
        element.attrs['content'] = he.decode(element.attrs['content'])
    }
    if (element.children && element.children.length > 0) {
        element.children = element.children.map(unescape)
    }
    return element
}

export function transformAndEscape(source: string) {
    const transformed = transform(source).map(unescape)

    return transformed
}
