import { sleep } from 'koishi'

import { request } from "@dingyi222666/koishi-plugin-chathub/lib/llm-core/utils/request"
import { createLogger } from '@dingyi222666/koishi-plugin-chathub/lib/llm-core/utils/logger'

import randomUserAgent from "random-useragent"

import { ClaudeChatResponse, ClaudeCreateConversationResponse, ClaudeOrganizationResponse, ClaudeSendMessageRequest } from './types';
import Claude2ChatPlugin from '.';

const logger = createLogger('@dingyi222666/chathub-claude2-adapter/api')

const STOP_TOKEN = ["\n\nuser:", "\n\nsystem:"]

export class Api {

    private _organizationId: string

    private _ua = randomUserAgent.getRandom((ua: any) => ua.browserName === 'Chrome' && parseFloat(ua.browserVersion) >= 90)

    private _headers: any = {
        "content-type": "application/json",
        Host: 'claude.ai',
        Origin: "https://claude.ai",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        Referer: 'https://claude.ai/chats',
        Connection: 'keep-alive',
        //  "User-Agent": this._ua,
        'Accept': '*/*',
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) chatall/1.37.59 Chrome/114.0.5735.289 Electron/25.5.0 Safari/537.36",
        "sec-ch-ua": '"Not.A/Brand";v="8", "Chromium";v="114"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "Windows"

    }

    constructor(
        private readonly config: Claude2ChatPlugin.Config,
    ) {
        let cookie = config.cookie

        if (!cookie.includes("sessionKey=")) {
            cookie = 'sessionKey=' + cookie
        }

        this._headers.Cookie = cookie
    }

    private _concatUrl(url: string): string {
        return 'https://claude.ai/' + url
    }


    async sendMessage(conversationId: string, message: string): Promise<string> {

        if (this._organizationId == null) {
            await this.init()
        }

        const headers = {
            ...this._headers
        }

        headers.Accept = 'text/event-stream, text/event-stream'
        headers.Referer = `https://claude.ai/chat/`//${conversationId}`

        const requestBody: ClaudeSendMessageRequest = {
            completion: {
                prompt: message,
                //   timezone: "",
                model: "claude-2",
                incremental: true,
            },
            organization_uuid: this._organizationId,
            conversation_uuid: conversationId,
            text: message,
            attachments: []
        }

        const controller = new AbortController();


        const url = this._concatUrl(`/api/append_message`)

        const response = await request.fetch(
            url, {
            headers,
            signal: controller.signal,
            method: 'POST',
            body: JSON.stringify(requestBody)
        })

        const reader = response.body.getReader();

        let result = ''
        let stopTokenFound = false

        try {
            const decoder = new TextDecoder('utf-8')

            while (true) {

                const { value, done } = await reader.read();

                if (done) {
                    logger.debug(`Claude2 Response: ${result}`)
                    break; // 读取完毕
                }

                let rawDecodeValue = decoder.decode(value)

                logger.debug(`Claude2 SSE: ${rawDecodeValue}`)

                let splitted = rawDecodeValue.split('\n\n')

                if (splitted.length === 0 || (splitted.length === 1 && splitted[0].length === 0)) {
                    splitted = [rawDecodeValue]
                }

                for (let i = splitted.length - 1; i >= 0; i--) {
                    const item = splitted[i]

                    if (item.trim().length === 0) {
                        continue
                    } else {
                        rawDecodeValue = item
                        break
                    }
                }

                if (rawDecodeValue.startsWith('data: ')) {
                    rawDecodeValue = rawDecodeValue.substring(6)
                }

                
                try {
                    result += (JSON.parse(rawDecodeValue) as ClaudeChatResponse).completion
                } catch (e) {
                    logger.error(`Claude2 SSE Parse Error: ${rawDecodeValue} `)

                    if (rawDecodeValue.includes('div')) {
                        throw new Error('Claude2 出现了一些问题！可能是被 Claude 官方检测了。请尝试重启 koishi 或更换 Cookie 或者等待一段时间再试。')
                    }
                    continue
                }

                let text = result

                STOP_TOKEN.forEach(token => {
                    if (result != null && result.includes(token)) {
                        const startIndex = result.indexOf(token)
                        text = result.substring(0, startIndex)
                            .replace(token, '')

                        text = result

                        stopTokenFound = true

                        controller.abort()
                    }

                })

                if (!stopTokenFound && text != null) {
                    result = text
                }

            }
        } catch (e) {
            logger.error(e)
            throw e
        }


        return result
    }

    async deleteConversation(conversationId: string): Promise<void> {

        /* if (this._organizationId == null) {
            await this.init()
        } */

        const headers = {
            ...this._headers
        }

       // headers.Accept = 'text/event-stream, text/event-stream'
        headers.Referer = `https://claude.ai/chat/${conversationId}`

        const controller = new AbortController();


        const url = this._concatUrl(`/organizations/${this._organizationId}/chat_conversations/${conversationId}`)

        const response = await request.fetch(
            url, {
            headers,
            signal: controller.signal,
            method: 'delete',
        })


        try {
            logger.debug(`Claude2 deleteConversation: ${await response.text()}`)
        } catch (e) {
            logger.error(e)
            throw e
        }


    }

    async createConversation(conversationId: string) {

        if (this._organizationId == null) {
            await this.init()
        }

        const url = this._concatUrl(`/api/organizations/${this._organizationId}/chat_conversations`)


        const result = await request.fetch(
            url,
            {
                headers: this._headers,
                method: 'POST',
                body: JSON.stringify({
                    uuid: conversationId,
                    name: ""
                })
            })

        const raw = await result.text()

        logger.debug(`Claude2 createConversation: ${raw}`)

        try {
            const data = JSON.parse(raw) as ClaudeCreateConversationResponse

            if (data?.uuid !== conversationId) {
                return Error('Can\'t find create conversation: ' + raw)
            }

            return true
        } catch (e) {
            return Error('Can\'t parse create conversation: ' + raw)
        }
    }


    async getOrganizationsId() {
        const url = this._concatUrl('api/organizations')

        const headers = {
            ...this._headers
        }

        headers.Origin = undefined

        const result = await request.fetch(
            url,
            {
                headers: this._headers,
            }
        )

        const raw = await result.text()

        logger.debug(`Claude2 getOrganizationsId: ${raw}`)

        try {
            const array = JSON.parse(raw) as ClaudeOrganizationResponse[]
            const data = array?.[0]

            if (!data?.uuid) {
                throw new Error('Can\'t find organization id: ' + raw)
            }

            return data.uuid
        } catch (e) {
            throw new Error('Can\'t parse organization id: ' + raw)
        }

    }

    async init() {
        for (let count = 0; count < this.config.maxRetries; count++) {
            try {
                this._organizationId = await this.getOrganizationsId()
                break
            } catch (e) {
                logger.error(e)
                if (e.cause) {
                    logger.error(e.cause)
                }
                await sleep(10000)

                if (count == this.config.maxRetries - 1) {
                    throw e
                }
            }
        }

    }


}


