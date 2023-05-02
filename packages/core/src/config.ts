import { Awaitable, Computed, Schema } from 'koishi'

export interface Config {
    botName: string,
    isNickname: boolean,
    allowPrivate: boolean,
    isReplyWithAt: boolean,
    msgCooldown: number,
    randomReplyFrequency: number,
    expireTime: number,
    conversationIsolationGroup: string[],
    injectDataEnenhance: boolean,
    injectData: boolean,
    isLog: boolean,
    configDir: string,
    proxyAddress: string,
    isProxy: boolean,
    outputMode: string,
    sendThinkingMessage: boolean,
    sendThinkingMessageTimeout: number,
    thinkingMessage: string,
    splitMessage: boolean,
    blackList: Computed<Awaitable<boolean>>,
    blockText: string,
    censor: boolean,
}

export const Config: Schema<Config> = Schema.intersect([
    Schema.object({
        botName: Schema.string().description('bot名字').default('香草'),
        isNickname: Schema.boolean().description('是否允许全局设置中的昵称引发回复').default(true),
    }).description('bot相关配置'),

    Schema.object({
        allowPrivate: Schema.boolean().description('是否允许私聊').default(true),
        isReplyWithAt: Schema.boolean().description('是否在回复时引用原消息').default(false),
        msgCooldown: Schema.number().description('全局消息冷却时间，单位为秒，防止适配器调用过于频繁')
            .min(1).max(3600).step(1).default(5),

        outputMode: Schema.union([
            Schema.const('raw').description("原始（直接输出，不做任何处理）"),
            Schema.const('text').description("文本（把回复当成markdown渲染）"),
            Schema.const('image').description("图片（需要puppeteer服务）"),
            Schema.const("mixed",).description("混合（图片和文本）"),
            Schema.const('voice').description("语音（需要vits服务）"),
        ]).default("text").description('Bot回复的模式'),

        splitMessage: Schema.boolean().description('是否分割消息发现（看起来更像普通水友（并且会不支持引用消息），不支持原始模式和图片模式）').default(false),

        sendThinkingMessage: Schema.boolean().description('是否发送思考中的消息').default(true),
        sendThinkingMessageTimeout: Schema.number().description('当请求多少毫秒后适配器没有响应时发送思考中的消息').default(10000),

        thinkingMessage: Schema.string().description('思考中的消息内容').default('我还在思考中呢，稍等一下哦~'),

        randomReplyFrequency: Schema.percent().description('随机回复频率')
            .min(0).max(1).step(0.01).default(0.2),

    }).description('回复选项'),

    Schema.object({
        expireTime: Schema.number().default(1440).description('不活跃对话的保存时间，单位为分钟。'),
        conversationIsolationGroup: Schema.array(Schema.string()).description('对话隔离群组，开启后群组内对话将隔离到个人级别（填入群组在koishi里的ID）')
            .default([]),
        blackList: Schema.union([
            Schema.boolean(),
            Schema.any().hidden(),
        ]).role('computed').description("黑名单列表 (请只对需要拉黑的用户或群开启，其他（如默认）请不要打开，否则会导致全部聊天都会被拉黑无法回复").default(false),
        blockText: Schema.string().description('黑名单回复内容').default('哎呀(ｷ｀ﾟДﾟ´)!!，你怎么被拉入黑名单了呢？要不你去问问我的主人吧。'),
        censor: Schema.boolean().description('是否开启文本审核服务（需要安装censor服务').default(false),
        injectData: Schema.boolean().description('是否注入信息数据以用于模型聊天（增强模型回复，需要安装服务支持并且适配器支持）').default(true),
        injectDataEnenhance: Schema.boolean().description('是否加强注入信息的数据（会尝试把每一条注入的数据也放入聊天记录,并且也要打开注入信息数据选项。）[大量token消耗，只是开发者拿来快速填充上下文的，建议不要打开]').default(false),
    }).description("对话选项"),

    Schema.object({
        isProxy: Schema.boolean().description('是否使用代理，开启后会为相关插件的网络服务使用代理').default(false),
        configDir: Schema.string().description('配置文件目录').default('chathub'),
        isLog: Schema.boolean().description('是否输出Log，调试用').default(false),
    }).description('杂项'),

    Schema.union([
        Schema.object({
            isProxy: Schema.const(true).required(),
            proxyAddress: Schema.string().description('插件网络请求的代理地址，填写后相关插件的网络服务都将使用该代理地址。如不填写会尝试使用koishi的全局配置里的代理设置').default(''),
        }),
        Schema.object({}),
    ]),
  
]) as Schema<Config>
