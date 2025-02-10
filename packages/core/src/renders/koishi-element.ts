import { Message, RenderMessage, RenderOptions } from '../types'
import { Renderer } from './default'
import { h, Schema } from 'koishi'
import he from 'he'
import { logger } from 'koishi-plugin-chatluna'

export class KoishiElementRenderer extends Renderer {
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

    schema = Schema.const('koishi-element').i18n({
        'zh-CN': '将回复作为 koishi 消息元素进行渲染',
        'en-US': 'Render as koishi message element template'
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
    try {
        return h.parse(source).map(unescape)
    } catch (e) {
        logger.error(e)
        return [h.text(source)]
    }
}
