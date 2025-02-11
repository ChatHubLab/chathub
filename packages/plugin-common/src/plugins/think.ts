/* eslint-disable max-len */
import { Tool } from '@langchain/core/tools'
import { Context, Schema, Session } from 'koishi'
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'
import { Config } from '..'
import { ChatLunaChatModel } from 'koishi-plugin-chatluna/llm-core/platform/model'
import { PlatformService } from 'koishi-plugin-chatluna/llm-core/platform/service'
import { ModelType } from 'koishi-plugin-chatluna/llm-core/platform/types'
import { parseRawModelName } from 'koishi-plugin-chatluna/llm-core/utils/count_tokens'

export async function apply(
    ctx: Context,
    config: Config,
    plugin: ChatLunaPlugin
) {
    ctx.on('chatluna/model-added', (service) => {
        ctx.schema.set('model', Schema.union(getModelNames(service)))
    })

    ctx.on('chatluna/model-removed', (service) => {
        ctx.schema.set('model', Schema.union(getModelNames(service)))
    })

    ctx.schema.set('model', Schema.union(getModelNames(ctx.chatluna.platform)))

    if (config.think === true) {
        plugin.registerTool('think', {
            selector(_) {
                return true
            },

            async createTool(params, session) {
                const thinkModel = config.thinkModel

                if (thinkModel != null) {
                    const [platform, model] = parseRawModelName(thinkModel)
                    params.model = await ctx.chatluna.createChatModel(
                        platform,
                        model
                    )
                }

                return new ThinkTool(params.model)
            }
        })
    }

    if (config.chat === true) {
        plugin.registerTool('question', {
            selector(history) {
                return true
            },
            alwaysRecreate: true,
            async createTool(params, session) {
                return new QuestionTool(session)
            }
        })
    }

    if (config.send === true) {
        plugin.registerTool('send', {
            selector(history) {
                return true
            },
            async createTool(params, session) {
                return new SendTool(ctx, session)
            },
            alwaysRecreate: true
        })
    }

    function getModelNames(service: PlatformService) {
        return service.getAllModels(ModelType.llm).map((m) => Schema.const(m))
    }
}

export class ThinkTool extends Tool {
    name = 'think'
    description =
        'A tool for deep analysis, structured thinking, and task planning on complex problems.'

    constructor(private _model: ChatLunaChatModel) {
        super()
    }

    private _thinkPrompt = `Analyze the following input comprehensively and create an action plan:

1. Summarize the main problem or task
2. Define a clear goal
3. Outline potential steps or subtasks to achieve the goal
4. For each step, suggest which tools (think, chat, send, or others) might be needed and why
5. Identify any assumptions, constraints, or potential challenges

Provide a structured response with a clear goal, action plan, and tool suggestions:

{input}

Think critically and creatively. Be specific about which tools to use for each step. Your response should be actionable, allowing for immediate execution of the plan using the suggested tools.`

    private _responsePrompt = `Based on the analysis and action plan provided, proceed with the following steps:

1. Review the analysis and action plan carefully.
2. For each step in the plan:
   a. If a tool is suggested, use that tool by calling it with the appropriate input.
   b. If no specific tool is suggested, decide which tool would be most appropriate and use it.
3. After each tool use, evaluate the result and decide on the next action.
4. If you encounter any challenges or need more information, use the 'think' tool again to refine the plan.
5. Continue until you have completed all steps or achieved the defined goal.

Here's the analysis and action plan:

{analysis}

Proceed with executing this plan, using the suggested tools and your best judgment. Provide updates on your progress and any results obtained from tool usage.`

    /** @ignore */
    async _call(input: string): Promise<string> {
        try {
            const thinkPrompt = this._thinkPrompt.replace('{input}', input)
            const response = await this._model.invoke(thinkPrompt)
            let analysis = response.content as string

            if (response.additional_kwargs?.reasoning_content) {
                analysis = response.additional_kwargs[
                    'reasoning_content'
                ] as string
            }

            const finalResponse = this._responsePrompt.replace(
                '{analysis}',
                analysis
            )
            return finalResponse
        } catch (error) {
            return 'An error occurred while processing your request. Please try again.'
        }
    }
}

export class QuestionTool extends Tool {
    name = 'question'
    description = `A tool for interacting with the user. Use this when you need to ask the user for your task, clarification, or a decision. The input is the message or question you want to send to the user, and the output is the user's response. Only use this tool when absolutely necessary for task completion. If the user requests to stop interactions or if you have sufficient information to proceed, avoid using this tool and provide a direct response or result instead.`

    constructor(private session: Session) {
        super()
    }

    /** @ignore */
    async _call(input: string) {
        await this.session.send(input)

        try {
            const result = await this.session.prompt()
            return result
        } catch (error) {
            return 'An error occurred while requesting user input. Please stop the tool call.'
        }
    }
}

export class SendTool extends Tool {
    name = 'send'
    description =
        'A tool for sending messages to the user. Use this when you want to communicate information, results, or responses directly to the user without expecting a reply. The input is the message you want to send.'

    constructor(
        private ctx: Context,
        private session: Session
    ) {
        super()
    }

    /** @ignore */
    async _call(input: string) {
        try {
            const elements = (
                await this.ctx.chatluna.renderer.render({
                    content: input
                })
            ).flatMap((message) => {
                const elements = message.element
                if (elements instanceof Array) {
                    return elements
                } else {
                    return [elements]
                }
            })

            await this.session.send(elements)
            return 'Message sent successfully. '
        } catch (error) {
            return 'An error occurred while sending your message. Please try again.'
        }
    }
}
