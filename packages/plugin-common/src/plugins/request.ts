import { Tool } from '@langchain/core/tools'
import { Context } from 'koishi'
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'
import { randomUA } from 'koishi-plugin-chatluna/utils/request'
import { getMessageContent } from 'koishi-plugin-chatluna/utils/string'
import { Config } from '..'

export async function apply(
    ctx: Context,
    config: Config,
    plugin: ChatLunaPlugin
) {
    if (config.request !== true) {
        return
    }

    const requestGetTool = new RequestsGetTool(
        plugin,
        {
            'User-Agent': randomUA()
        },
        {
            maxOutputLength: config.requestMaxOutputLength
        }
    )

    const requestPostTool = new RequestsPostTool(
        plugin,
        {
            'User-Agent': randomUA()
        },
        {
            maxOutputLength: config.requestMaxOutputLength
        }
    )

    plugin.registerTool(requestGetTool.name, {
        selector(history) {
            if (config.actions === true) {
                return true
            }

            return history.some((item) => {
                const content = getMessageContent(item.content)
                return (
                    content.includes('url') ||
                    content.includes('http') ||
                    content.includes('request') ||
                    content.includes('请求') ||
                    content.includes('网页') ||
                    content.includes('get')
                )
            })
        },
        createTool: async () => requestGetTool
    })

    plugin.registerTool(requestPostTool.name, {
        selector(history) {
            if (config.actions === true) {
                return true
            }
            return history.some((item) => {
                const content = getMessageContent(item.content)
                return (
                    content.includes('url') ||
                    content.includes('http') ||
                    content.includes('request') ||
                    content.includes('请求') ||
                    content.includes('网页') ||
                    content.includes('post')
                )
            })
        },
        createTool: async () => requestPostTool
    })
}

export interface Headers {
    [key: string]: string
}

export interface RequestTool {
    headers: Headers
    maxOutputLength: number
}

export class RequestsGetTool extends Tool implements RequestTool {
    name = 'requests_get'

    maxOutputLength = 2000

    constructor(
        private _plugin: ChatLunaPlugin,
        public headers: Headers = {},
        { maxOutputLength }: { maxOutputLength?: number } = {}
    ) {
        super({
            ...headers
        })

        this.maxOutputLength = maxOutputLength ?? this.maxOutputLength
    }

    /** @ignore */
    async _call(input: string) {
        const res = await this._plugin.fetch(input, {
            headers: this.headers
        })
        const text = await res.text()
        return text.slice(0, this.maxOutputLength)
    }

    description = `A portal to the internet. Use this when you need to get specific content from a website.
  Input should be a url string (i.e. "https://www.google.com"). The output will be the text response of the GET request.`
}

export class RequestsPostTool extends Tool implements RequestTool {
    name = 'requests_post'

    maxOutputLength = Infinity

    constructor(
        private _plugin: ChatLunaPlugin,
        public headers: Headers = {},
        { maxOutputLength }: { maxOutputLength?: number } = {}
    ) {
        super({
            ...headers
        })

        this.maxOutputLength = maxOutputLength ?? this.maxOutputLength
    }

    /** @ignore */
    async _call(input: string) {
        try {
            const { url, data } = JSON.parse(input)
            const res = await this._plugin.fetch(url, {
                method: 'POST',
                headers: this.headers,
                body: JSON.stringify(data)
            })
            const text = await res.text()
            return text.slice(0, this.maxOutputLength)
        } catch (error) {
            return `${error}`
        }
    }

    description = `Use this when you want to POST to a website.
  Input should be a json string with two keys: "url" and "data".
  The value of "url" should be a string, and the value of "data" should be a dictionary of
  key-value pairs you want to POST to the url as a JSON body.
  Be careful to always use double quotes for strings in the json string
  The output will be the text response of the POST request.`
}
