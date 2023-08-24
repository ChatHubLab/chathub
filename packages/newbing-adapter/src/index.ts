import { createLogger } from '@dingyi222666/koishi-plugin-chathub/lib/llm-core/utils/logger'
import { ChatHubPlugin } from "@dingyi222666/koishi-plugin-chathub/lib/services/chat"
import { Context, Schema } from 'koishi'
import fs from 'fs/promises'
import path from 'path'
import os from 'os'
import { BingChatProvider } from './providers'


const logger = createLogger('@dingyi222666/chathub-newbing-adapter')

class BingChatPlugin extends ChatHubPlugin<BingChatPlugin.Config> {

    name = "@dingyi222666/chathub-newbing-adapter"

    constructor(protected ctx: Context, public readonly config: BingChatPlugin.Config) {
        super(ctx, config)

        this.config.chatConcurrentMaxSize = 0

        setTimeout(async () => {
            await ctx.chathub.registerPlugin(this)

            this.registerModelProvider(new BingChatProvider(config))
        })

    }
}


namespace BingChatPlugin {

    //export const usage = readFileSync(__dirname + '/../README.md', 'utf8')

    export interface Config extends ChatHubPlugin.Config {
        cookie: string,
        showExtraInfo: boolean,

        webSocketApiEndPoint: string,
        createConversationApiEndPoint: string,

        sydney: boolean
    }

    export const Config: Schema<Config> = Schema.intersect([
        ChatHubPlugin.Config,

        Schema.object({
            cookie: Schema.string().role('secret').description('Bing 账号的 cookie').default(""),
            webSocketApiEndPoint: Schema.string().description('New Bing 的WebSocket Api EndPoint').default("wss://sydney.bing.com/sydney/ChatHub"),
            createConversationApiEndPoint:  Schema.string().description('New Bing 的新建会话 Api EndPoint').default("https://edgeservices.bing.com/edgesvc/turing/conversation/create"),
        }).description('请求设置'),


        Schema.object({
            sydney: Schema.boolean().description('是否开启 Sydeny 模式（破解对话20次回复数限制，账号可能会有风险）').default(false),

            showExtraInfo: Schema.boolean().description('是否显示额外信息（如剩余回复数，猜你想问）').default(false),

        }).description('对话设置'),


    ])



    export const using = ['chathub']

}



export default BingChatPlugin