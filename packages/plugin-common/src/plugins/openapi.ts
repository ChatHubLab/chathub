/* eslint-disable max-len */
import { Tool, ToolParams } from '@langchain/core/tools'
import { Context } from 'koishi'
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'
import {
    fuzzyQuery,
    getMessageContent
} from 'koishi-plugin-chatluna/utils/string'
import { Config } from '..'

export async function apply(
    ctx: Context,
    config: Config,
    plugin: ChatLunaPlugin
) {
    if (config.actions !== true) {
        return
    }

    for (const action of config.actionsList) {
        const tool = AIPluginTool.fromAction(action)

        plugin.registerTool(action.name, {
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
                        ...action.name.split('.'),
                        ...(action.selector ?? [])
                    ])
                })
            },

            async createTool(params, session) {
                return tool
            }
        })
    }
}

/**
 * Interface for parameters required to create an instance of
 * AIPluginTool.
 */
export interface AIPluginToolParams extends ToolParams {
    name: string
    description: string
    apiSpec: string
}

/**
 * Class for creating instances of AI tools from plugins. It extends the
 * Tool class and implements the AIPluginToolParams interface.
 */
export class AIPluginTool extends Tool implements AIPluginToolParams {
    static lc_name() {
        return 'AIPluginTool'
    }

    name: string = ''

    apiSpec: string = ''
    description: string = ''

    constructor(params: AIPluginToolParams) {
        super(params)
        this.name = params.name
        this.description = params.description
        this.apiSpec = params.apiSpec
    }

    /** @ignore */
    async _call(input: string) {
        // First return the API spec
        const spec = this.apiSpec

        // Suggest the model to use request_get or request_post tools
        return `${spec}\n\nTo execute this API, please use the request_get or request_post tools with the appropriate endpoint and parameters from the OpenAPI specification above.`
    }

    static fromAction(action: Config['actionsList'][number]) {
        return new AIPluginTool({
            name: action.name,
            description: `Tool: ${action.name}
Purpose: ${action.description}
Usage: Call this tool whenever you need the OpenAPI specification to interact with this API. The response will provide the complete API documentation in OpenAPI format.`,
            apiSpec: `OpenAPI Spec in JSON or YAML format:\n${action.openAPISpec}`
        })
    }
}
