import { Context, h } from 'koishi';
import { Config } from '../config';
import { ChainMiddlewareContext, ChainMiddlewareRunStatus, ChatChain } from '../chains/chain';
import { v4 as uuidv4 } from 'uuid';
import { getKeysCache } from '..';
import { Preset } from '../preset';
import { resolveModelProvider } from './chat_time_limit_check';
import { Factory } from '../llm-core/chat/factory';
import { createLogger } from '../llm-core/utils/logger';

const logger = createLogger("@dingyi222666/chathub/middlewares/request_model")

export function apply(ctx: Context, config: Config, chain: ChatChain) {
    chain.middleware("resolve_model", async (session, context) => {

        const { room } = context.options

        const { model } = room

        const splitted = model.split(/(?<=^[^\/]+)\//)
        const modelProvider = (await Factory.selectModelProviders(async (name, _) => {
            return name == splitted[0]
        }))[0]

        if (modelProvider == null) {
            throw new Error("无法找到模型适配器，是否设置了默认模型或者没指定模型？")
        }

        const modelList = await modelProvider.listModels()

        if (modelList.length == 0 || modelList.find(x => x == splitted[1]) == null) {

            // 这比较难，强行 fallback 到推荐模型

            const recommendModel = modelProvider.name + "/" + (await modelProvider.recommendModel())

            logger.debug(`[resolve_model] recommendModel: ${recommendModel}`)

            await context.send("检查到您可能更新了某些配置，已无法使用之前设置的旧的模型，已为您自动切换到其他可用模型。")

            room.model = recommendModel

            ctx.database.upsert('chathub_room', [room])

            return ChainMiddlewareRunStatus.CONTINUE
        }


        if (room.model != null) {
            return ChainMiddlewareRunStatus.SKIPPED
        } else {
            throw new Error("无法找到模型，是否设置了默认模型或者没指定模型？")
        }

    }).before("request_model")
    //  .before("lifecycle-request_model")


}

declare module '../chains/chain' {
    interface ChainMiddlewareName {
        "resolve_model": never
    }
}


