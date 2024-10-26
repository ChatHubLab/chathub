/* eslint-disable max-len */

import { StructuredTool } from '@langchain/core/tools'
import { Context, Element, h, Session } from 'koishi'
import type { Command as CommandType } from '@satorijs/protocol'
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'
import {
    fuzzyQuery,
    getMessageContent
} from 'koishi-plugin-chatluna/utils/string'
import { Config } from '..'
import { z } from 'zod'

export async function apply(
    ctx: Context,
    config: Config,
    plugin: ChatLunaPlugin
) {
    if (config.command !== true) {
        return
    }

    const commandList = getCommandList(ctx, config.commandList)

    for (const command of commandList) {
        const prompt = generateSingleCommandPrompt(command)
        let normalizedName = normalizeCommandName(command.name)

        if (normalizedName.replaceAll('_', '').length < 1) {
            normalizedName = crypto.randomUUID().substring(0, 16)

            // while the normalized name is not start with number

            while (/^[0-9]/.test(normalizedName[0])) {
                normalizedName = crypto.randomUUID().substring(0, 16)
            }
        }

        plugin.registerTool(`command-execute-${normalizedName}`, {
            selector(history) {
                return history.some((item) => {
                    const content = getMessageContent(item.content)

                    return fuzzyQuery(content, [
                        '令',
                        '调用',
                        '获取',
                        'get',
                        'help',
                        'command',
                        '执行',
                        '用',
                        'execute',
                        ...command.name.split('.'),
                        ...(command.selector ?? [])
                    ])
                })
            },

            async createTool(params, session) {
                return new CommandExecuteTool(
                    ctx,
                    session,
                    `${normalizedName}`,
                    prompt,
                    command
                )
            }
        })
    }
}

function normalizeCommandName(name: string): string {
    // Replace non-alphanumeric characters (except underscore and hyphen) with underscores
    return name.replace(/[^a-zA-Z0-9_-]/g, '_')
}

function generateSingleCommandPrompt(command: PickCommandType): string {
    let prompt = `To execute the "${command.name}" tool, use the following input format:

${command.name}`

    if (command.arguments.length > 0) {
        prompt += ` ${command.arguments.map((arg) => `<${arg.name}:${arg.type}>`).join(' ')}`
    }

    if (command.options.length > 0) {
        prompt += ` ${command.options.map((opt) => `[--${opt.name}${opt.type !== 'boolean' ? ` <${opt.type}>` : ''}]`).join(' ')}`
    }

    prompt += '\n\n'
    prompt += `Tool Description: ${command.description || 'No description'}\n\n`

    if (command.arguments.length > 0) {
        prompt += 'Tool Arguments:\n'
        command.arguments.forEach((arg) => {
            prompt += `- ${arg.name}: ${getDescription(arg.description)}${arg.required ? ' (Required)' : ' (Optional)'}\n`
        })
        prompt += '\n'
    }

    if (command.options.length > 0) {
        prompt += 'Tool Options:\n'
        command.options.forEach((opt) => {
            if (opt.name !== 'help') {
                prompt += `- --${opt.name}: ${getDescription(opt.description)}${opt.required ? ' (Required)' : ''}\n`
            }
        })
        prompt += '\n'
    }

    return prompt
}

function getDescription(description: string | Record<string, string>): string {
    if (typeof description === 'string') {
        return description
    }

    return description['zh-CN'] || description[''] || 'No description'
}

function getCommandList(
    ctx: Context,
    rawCommandList: Config['commandList']
): PickCommandType[] {
    return ctx.$commander._commandList
        .filter((item) => !item.name.includes('chatluna'))
        .filter((item) => {
            if (rawCommandList.length < 1) {
                return true
            }
            return rawCommandList.some(
                (command) => command.command === item.name
            )
        })
        .map((item) => item.toJSON())
        .map((item) => {
            const rawCommand = rawCommandList.find(
                (command) => command.command === item.name
            )

            let description: string | CommandType['description'] =
                rawCommand?.description

            if ((rawCommand?.description.length ?? 0) < 1) {
                description = JSON.stringify(item.description)
            }

            return {
                ...item,
                selector: rawCommand?.selector,
                description
            }
        })
}

export class CommandExecuteTool extends StructuredTool {
    schema = z.object({
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
    }) as any

    constructor(
        public ctx: Context,
        public session: Session,
        public name: string,
        public description: string,
        private command: PickCommandType
    ) {
        super()

        this.schema = this.generateSchema()
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    private generateSchema(): z.ZodObject<any> {
        const schemaShape: Record<string, z.ZodTypeAny> = {}

        this.command.arguments.forEach((arg) => {
            const zodType = this.getZodType(arg.type)
            schemaShape[arg.name] = arg.required ? zodType : zodType.optional()
        })

        this.command.options.forEach((opt) => {
            if (opt.name !== 'help') {
                const zodType = this.getZodType(opt.type)
                schemaShape[opt.name] = opt.required
                    ? zodType
                    : zodType.optional()
            }
        })

        if (Object.keys(schemaShape).length < 1) {
            return z.object({
                input: z.string().optional()
            })
        }

        return z.object(schemaShape)
    }

    private getZodType(type: string): z.ZodTypeAny {
        switch (type) {
            case 'text':
            case 'string':
            case 'date':
                return z.string()
            case 'integer':
            case 'posint':
            case 'natural':
            case 'number':
                return z.number()
            case 'boolean':
                return z.boolean()
            default:
                return z.string()
        }
    }

    /** @ignore */
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    async _call(input: any) {
        const koishiCommand = this.parseInput(input)

        const validationString = randomString(8)
        const session = this.session

        await session.send(
            `模型请求执行指令 ${koishiCommand}，如需同意，请输入以下字符：${validationString}`
        )
        const canRun = await this.session.prompt()

        if (canRun !== validationString) {
            await this.session.send('指令执行失败')
            return `The command ${koishiCommand} execution failed, because the user didn't confirm`
        }

        try {
            const result = await this.session.execute(koishiCommand, true)

            await this.session.send(result)

            return `Successfully executed command ${koishiCommand} with result: ${elementToString(result)}`
        } catch (e) {
            this.ctx.logger.error(e)
            return `The command ${koishiCommand} execution failed, because ${e.message}`
        }
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    private parseInput(input: Record<string, any>): string {
        try {
            const args: string[] = []
            const options: string[] = []

            // 处理参数
            this.command.arguments.forEach((arg) => {
                if (arg.name in input) {
                    args.push(String(input[arg.name]))
                }
            })

            // 处理选项
            this.command.options.forEach((opt) => {
                if (opt.name in input && opt.name !== 'help') {
                    if (opt.type === 'boolean') {
                        if (input[opt.name]) {
                            options.push(`--${opt.name}`)
                        }
                    } else {
                        options.push(`--${opt.name}`, String(input[opt.name]))
                    }
                }
            })

            // 构建完整的命令字符串
            const fullCommand = [this.command.name, ...args, ...options]
                .join(' ')
                .trim()

            return fullCommand
        } catch (error) {
            console.error('Failed to parse JSON input:', error)
            throw new Error('Invalid JSON input')
        }
    }
}

export function randomString(size: number) {
    let text = ''
    const possible =
        'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    for (let i = 0; i < size; i++)
        text += possible.charAt(Math.floor(Math.random() * possible.length))
    return text
}

export function elementToString(elements: Element[]) {
    return h
        .select(elements, 'text')
        .map((element) => element.toString())
        .join(' ')
}

type PickCommandType = Omit<CommandType, 'description'> & {
    description?: string
    selector?: string[]
}
