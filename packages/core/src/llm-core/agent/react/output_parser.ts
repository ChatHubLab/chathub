import { AgentAction, AgentFinish } from '@langchain/core/agents'
import { renderTemplate } from '@langchain/core/prompts'
import { OutputParserException } from '@langchain/core/output_parsers'
import { AgentActionOutputParser } from '../types.js'
import { FORMAT_INSTRUCTIONS } from './prompt.js'

/**
 * Parses ReAct-style LLM calls that have a single tool input.
 *
 * Expects output to be in one of two formats.
 *
 * If the output signals that an action should be taken,
 * should be in the below format. This will result in an AgentAction
 * being returned.
 *
 * ```
 * Thought: agent thought here
 * Action: search
 * Action Input: what is the temperature in SF?
 * ```
 *
 * If the output signals that a final answer should be given,
 * should be in the below format. This will result in an AgentFinish
 * being returned.
 *
 * ```
 * Thought: agent thought here
 * Final Answer: The temperature is 100 degrees
 * ```
 * @example
 * ```typescript
 *
 * const runnableAgent = RunnableSequence.from([
 *   ...rest of runnable
 *   new ReActSingleInputOutputParser({ toolNames: ["SerpAPI", "Calculator"] }),
 * ]);
 * const agent = AgentExecutor.fromAgentAndTools({
 *   agent: runnableAgent,
 *   tools: [new SerpAPI(), new Calculator()],
 * });
 * const result = await agent.invoke({
 *   input: "whats the weather in pomfret?",
 * });
 * ```
 */
export class ReActSingleInputOutputParser extends AgentActionOutputParser {
    lc_namespace = ['langchain', 'agents', 'react']

    private toolNames: string[]

    constructor(fields: { toolNames: string[] }) {
        super(fields)
        this.toolNames = fields.toolNames
    }

    /**
     * Parses the given text into an AgentAction or AgentFinish object. If an
     * output fixing parser is defined, uses it to parse the text.
     * @param text Text to parse.
     * @returns Promise that resolves to an AgentAction or AgentFinish object.
     */
    async parse(text: string): Promise<AgentAction | AgentFinish> {
        const actionRegex = /Action:\s*(.*\})/ms
        const thoughtsRegex = /Thought:\s*([^]*?)(?=Action:|$)/ms
        const actionMatch = text.match(actionRegex)
        const thoughtsMatch = text.match(thoughtsRegex)

        if (actionMatch) {
            let [rawAction, action] = actionMatch
            let [, thoughts] = thoughtsMatch

            action = action.trim()

            if (!thoughts) {
                thoughts = text.replace(rawAction, '').trim()
            }

            return this.parseAction(action, thoughts)
        }

        throw new OutputParserException(
            `Could not parse LLM output: ${text}`,
            `Could not parse LLM output: ${text}`,
            `Could not parse LLM output: ${text}`,
            true
        )
    }

    parseAction(action: string, thoughts: string): AgentAction | AgentFinish {
        try {
            const parsedRawAction = JSON.parse(action) as unknown as {
                name: string
                arguments: Record<string, string>
            }

            if (
                parsedRawAction.name == null ||
                parsedRawAction.arguments == null
            ) {
                throw new OutputParserException(
                    `Could not parse action command: ${action}`
                )
            }

            if (parsedRawAction.name === 'final_answer') {
                return {
                    returnValues: { output: parsedRawAction.arguments },
                    log: thoughts
                }
            }

            return {
                tool: parsedRawAction.name,
                toolInput: parsedRawAction.arguments || {},
                log: thoughts
            }
        } catch (e) {
            throw new OutputParserException(`Could not parse action: ${action}`)
        }
    }

    /**
     * Returns the format instructions as a string. If the 'raw' option is
     * true, returns the raw FORMAT_INSTRUCTIONS.
     * @param options Options for getting the format instructions.
     * @returns Format instructions as a string.
     */
    getFormatInstructions(): string {
        return renderTemplate(FORMAT_INSTRUCTIONS, 'f-string', {
            tool_names: this.toolNames.join(', ')
        })
    }
}
