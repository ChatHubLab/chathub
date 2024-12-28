import { Context, Logger } from 'koishi'
import { Config } from '../config'
import { ChainMiddlewareRunStatus, ChatChain } from '../chains/chain'
import { createLogger } from 'koishi-plugin-chatluna/utils/logger'
import fs from 'fs/promises'
import { PresetTemplate } from '../llm-core/prompt'

let logger: Logger

export function apply(ctx: Context, config: Config, chain: ChatChain) {
    logger = createLogger(ctx)

    chain
        .middleware('delete_preset', async (session, context) => {
            const { command } = context

            if (command !== 'delete_preset')
                return ChainMiddlewareRunStatus.SKIPPED

            const presetName = context.options.deletePreset
            const preset = ctx.chatluna.preset

            let presetTemplate: PresetTemplate

            try {
                presetTemplate = await preset.getPreset(presetName)

                const allPreset = await preset.getAllPreset()

                if (allPreset.length === 1) {
                    await context.send(session.text('.only_one_preset'))
                    return ChainMiddlewareRunStatus.STOP
                }
            } catch (e) {
                logger.error(e)
                await context.send(session.text('.not_found'))
                return ChainMiddlewareRunStatus.STOP
            }

            await context.send(session.text('.confirm_delete', [presetName]))

            const result = await session.prompt(1000 * 30)

            if (!result) {
                context.message = session.text('.timeout', [presetName])
                return ChainMiddlewareRunStatus.STOP
            }

            if (result !== 'Y') {
                context.message = session.text('.cancelled', [presetName])
                return ChainMiddlewareRunStatus.STOP
            }

            await fs.rm(presetTemplate.path)

            const defaultPreset = await preset.getDefaultPreset()

            logger.debug(
                `${context.options.senderInfo} ${defaultPreset.triggerKeyword[0]}`
            )

            const roomList = await ctx.database.get('chathub_room', {
                preset: presetName
            })

            for (const room of roomList) {
                room.preset = defaultPreset.triggerKeyword[0]
            }

            await ctx.database.upsert('chathub_room', roomList)

            context.message = session.text('.success', [presetName])

            ctx.scope.parent.scope.update(config, true)

            return ChainMiddlewareRunStatus.STOP
        })
        .after('lifecycle-handle_command')
}

declare module '../chains/chain' {
    interface ChainMiddlewareName {
        delete_preset: string
    }

    interface ChainMiddlewareContextOptions {
        deletePreset?: string
    }
}
