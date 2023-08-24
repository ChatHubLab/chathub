import { Context } from 'koishi';
import { Config } from '../config';
import { ChainMiddlewareRunStatus, ChatChain } from '../chains/chain';
import { createLogger } from '../llm-core/utils/logger';
import { Factory } from '../llm-core/chat/factory';
import { dump } from 'js-yaml'
import fs from 'fs/promises'
import { randomUUID } from 'crypto';
import { PresetTemplate } from '../llm-core/prompt';
import { getKeysCache, getPresetInstance } from '..';

const logger = createLogger("@dingyi222666/chathub/middlewares/delete_preset")

export function apply(ctx: Context, config: Config, chain: ChatChain) {
    chain.middleware("delete_preset", async (session, context) => {

        const { command } = context

        if (command !== "delete_preset") return ChainMiddlewareRunStatus.SKIPPED

        const presetName = context.options.deletePreset
        const preset = getPresetInstance()

        let presetTemplate: PresetTemplate

        try {
            presetTemplate = await preset.getPreset(presetName)

            const allPreset = await preset.getAllPreset()

            if (allPreset.length === 1) {
                await context.send("现在只有一个预设了，删除后将无法使用预设功能，所以不允许删除。")
                return ChainMiddlewareRunStatus.STOP
            }

        } catch (e) {
            await context.send("找不到该预设！请检查你是否输入了正确的预设？")

            return ChainMiddlewareRunStatus.STOP
        }

        await context.send(`是否要删除 ${presetName} 预设？输入大写 Y 来确认删除，输入其他字符来取消删除。提示：删除后使用了该预设的会话将会自动删除无法使用。`)

        const result = await session.prompt(1000 * 30)

        if (!result) {
            context.message = `删除预设超时，已取消删除预设: ${presetName}`
            return ChainMiddlewareRunStatus.STOP
        }

        if (result !== "Y") {
            context.message = `已取消删除预设: ${presetName}`
            return ChainMiddlewareRunStatus.STOP
        }

        await fs.rm(presetTemplate.path)

        const defaultPreset = await preset.getDefaultPreset()

        logger.debug(`${context.options.senderInfo} ${defaultPreset.triggerKeyword[0]}`)


        const roomList = await ctx.database.get('chathub_room', {
            preset: presetName
        })

        for (const room of roomList) {
            room.preset = defaultPreset.triggerKeyword[0]
            await ctx.database.upsert('chathub_room', [room])

        }

        context.message = `已删除预设: ${presetName}，即将自动重启完成更改。`

        ctx.scope.update(config, true)

        return ChainMiddlewareRunStatus.STOP
    }).after("lifecycle-handle_command")
}

declare module '../chains/chain' {
    interface ChainMiddlewareName {
        "delete_preset": string
    }

    interface ChainMiddlewareContextOptions {
        deletePreset?: string
    }
}