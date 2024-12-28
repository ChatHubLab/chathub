import { Context, Logger } from 'koishi'
import { Config } from '../config'
import { ChainMiddlewareRunStatus, ChatChain } from '../chains/chain'
import { createLogger } from 'koishi-plugin-chatluna/utils/logger'
import fs from 'fs/promises'

let logger: Logger

export function apply(ctx: Context, config: Config, chain: ChatChain) {
    logger = createLogger(ctx)
    chain
        .middleware('wipe', async (session, context) => {
            const { command } = context

            if (command !== 'wipe') return ChainMiddlewareRunStatus.SKIPPED

            const expression = generateExpression()

            await context.send(
                session.text('.confirm_wipe', [expression.expression])
            )

            const result = await session.prompt(1000 * 30)

            if (!result) {
                context.message = session.text('.timeout')
                return ChainMiddlewareRunStatus.STOP
            }

            if (result !== expression.result.toString()) {
                context.message = session.text('.incorrect_input')
                return ChainMiddlewareRunStatus.STOP
            }

            // drop database tables

            await ctx.database.drop('chathub_room_member')
            await ctx.database.drop('chathub_conversation')
            await ctx.database.drop('chathub_message')
            await ctx.database.drop('chathub_room')
            await ctx.database.drop('chathub_room_group_member')
            await ctx.database.drop('chathub_user')
            await ctx.database.drop('chathub_auth_group')
            await ctx.database.drop('chathub_auth_joined_user')
            await ctx.database.drop('chathub_auth_user')
            // knowledge

            try {
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                await ctx.database.drop('chathub_knowledge' as any)
            } catch (e) {
                logger.warn(`wipe: ${e}`)
            }

            // drop caches

            await ctx.chatluna.cache.clear('chathub/chat_limit')
            await ctx.chatluna.cache.clear('chathub/keys')
            await ctx.chatluna.cache.clear('chathub/client_config')

            // delete local database and temps

            try {
                await fs.rm('data/chathub/vector_store', { recursive: true })
            } catch (e) {
                logger.warn(`wipe: ${e}`)
            }

            try {
                await fs.rm('data/chathub/temp', { recursive: true })
            } catch (e) {
                logger.warn(`wipe: ${e}`)
            }

            context.message = session.text('.success')

            ctx.scope.parent.scope.update(config, true)

            return ChainMiddlewareRunStatus.STOP
        })
        .before('black_list')
}

declare module '../chains/chain' {
    interface ChainMiddlewareName {
        wipe: never
    }
}

// 接下来请你给我写这样的代码：随机生成一个三位数的加，乘，减，除 算式并生成字符以及结果。如 { expression: "111+444", result: 555 }
export function generateExpression() {
    const operators = ['+', '-', '*']

    const operator = operators[Math.floor(Math.random() * operators.length)]

    const a = Math.floor(Math.random() * 1000)

    const b = Math.floor(Math.random() * 1000)

    // eslint-disable-next-line no-eval
    const result = eval(`${a}${operator}${b}`)

    return {
        expression: `${a}${operator}${b}`,
        result
    }
}
