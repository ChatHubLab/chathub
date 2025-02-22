import {
    CallbackManager,
    CallbackManagerForChainRun,
    Callbacks
} from '@langchain/core/callbacks/manager'
import { BaseMessage, HumanMessage } from '@langchain/core/messages'
import { BaseLLMOutputParser } from '@langchain/core/output_parsers'
import { BasePromptTemplate } from '@langchain/core/prompts'
import { ensureConfig, RunnableConfig } from '@langchain/core/runnables'
import { ChainValues } from '@langchain/core/utils/types'
import { Session } from 'koishi'
import { BufferMemory } from 'koishi-plugin-chatluna/llm-core/memory/langchain'
import { ChatEvents } from '../../services/types'
import { ChatLunaChatModel } from 'koishi-plugin-chatluna/llm-core/platform/model'
import {
    ChatLunaError,
    ChatLunaErrorCode
} from 'koishi-plugin-chatluna/utils/error'
import {
    BaseLangChain,
    BaseLangChainParams
} from '@langchain/core/language_models/base'
import { ChatGeneration, RUN_KEY } from '@langchain/core/outputs'
import { BaseMemory } from '@langchain/core/memory'
import type { PostHandler } from '../../utils/types'

export type SystemPrompts = BaseMessage[]

export abstract class ChatLunaLLMChainWrapper {
    abstract call(arg: ChatLunaLLMCallArg): Promise<ChainValues>

    abstract historyMemory: BufferMemory

    abstract get model(): ChatLunaChatModel
}

export interface ChatLunaLLMCallArg {
    message: HumanMessage
    events: ChatEvents
    stream: boolean
    conversationId: string
    session: Session
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    variables?: Record<string, any>
    signal?: AbortSignal
    postHandler?: PostHandler
}

export interface ChatLunaLLMChainInput extends ChainInputs {
    /** Prompt object to use */
    prompt?: BasePromptTemplate
    /** LLM Wrapper to use */
    llm?: ChatLunaChatModel
    /** Kwargs to pass to LLM */
    llmKwargs?: this['llm']['ParsedCallOptions']
    /** OutputParser to use */
    outputParser?: BaseLLMOutputParser<ChatLunaChatModel>
    /** Key to use for output, defaults to `text` */
    outputKey?: string
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type LoadValues = Record<string, any>

export interface ChainInputs extends BaseLangChainParams {
    memory?: BaseMemory

    /**
     * @deprecated Use `callbacks` instead
     */
    callbackManager?: CallbackManager
}

/**
 * Base interface that all chains must implement.
 */
export abstract class BaseChain<
        RunInput extends ChainValues = ChainValues,
        RunOutput extends ChainValues = ChainValues
    >
    extends BaseLangChain<RunInput, RunOutput>
    implements ChainInputs
{
    declare memory?: BaseMemory

    get lc_namespace(): string[] {
        return ['langchain', 'chains', this._chainType()]
    }

    constructor(
        fields?: BaseMemory | ChainInputs,
        /** @deprecated */
        verbose?: boolean,
        /** @deprecated */
        callbacks?: Callbacks
    ) {
        if (
            arguments.length === 1 &&
            typeof fields === 'object' &&
            !('saveContext' in fields)
        ) {
            // fields is not a BaseMemory
            const { memory, callbackManager, ...rest } = fields
            super({ ...rest, callbacks: callbackManager ?? rest.callbacks })
            this.memory = memory
        } else {
            // fields is a BaseMemory
            super({ verbose, callbacks })
            this.memory = fields as BaseMemory
        }
    }

    /** @ignore */
    _selectMemoryInputs(values: ChainValues): ChainValues {
        const valuesForMemory = { ...values }
        if ('signal' in valuesForMemory) {
            delete valuesForMemory.signal
        }
        if ('timeout' in valuesForMemory) {
            delete valuesForMemory.timeout
        }
        return valuesForMemory
    }

    /**
     * Invoke the chain with the provided input and returns the output.
     * @param input Input values for the chain run.
     * @param config Optional configuration for the Runnable.
     * @returns Promise that resolves with the output of the chain run.
     */
    async invoke(
        input: RunInput,
        options?: RunnableConfig
    ): Promise<RunOutput> {
        const config = ensureConfig(options)
        const fullValues = await this._formatValues(input)
        const callbackManager_ = CallbackManager.configure(
            config?.callbacks,
            this.callbacks,
            config?.tags,
            this.tags,
            config?.metadata,
            this.metadata,
            { verbose: this.verbose }
        )
        const runManager = await callbackManager_?.handleChainStart(
            this.toJSON(),
            fullValues,
            undefined,
            undefined,
            undefined,
            undefined,
            config?.runName
        )
        let outputValues: RunOutput
        try {
            outputValues = await (fullValues.signal
                ? (Promise.race([
                      this._call(fullValues as RunInput, runManager, config),
                      // eslint-disable-next-line promise/param-names
                      new Promise((_, reject) => {
                          fullValues.signal?.addEventListener('abort', () => {
                              reject(
                                  new ChatLunaError(ChatLunaErrorCode.ABORTED)
                              )
                          })
                      })
                  ]) as Promise<RunOutput>)
                : this._call(fullValues as RunInput, runManager, config))
        } catch (e) {
            await runManager?.handleChainError(e)
            throw e
        }
        if (!(this.memory == null)) {
            await this.memory.saveContext(
                this._selectMemoryInputs(input),
                outputValues
            )
        }
        await runManager?.handleChainEnd(outputValues)
        // add the runManager's currentRunId to the outputValues
        Object.defineProperty(outputValues, RUN_KEY, {
            value: runManager ? { runId: runManager?.runId } : undefined,
            configurable: true
        })
        return outputValues
    }

    private _validateOutputs(outputs: Record<string, unknown>): void {
        const missingKeys = this.outputKeys.filter((k) => !(k in outputs))
        if (missingKeys.length) {
            throw new Error(
                `Missing output keys: ${missingKeys.join(
                    ', '
                )} from chain ${this._chainType()}`
            )
        }
    }

    async prepOutputs(
        inputs: Record<string, unknown>,
        outputs: Record<string, unknown>,
        returnOnlyOutputs = false
    ) {
        this._validateOutputs(outputs)
        if (this.memory) {
            await this.memory.saveContext(inputs, outputs)
        }
        if (returnOnlyOutputs) {
            return outputs
        }
        return { ...inputs, ...outputs }
    }

    /**
     * Run the core logic of this chain and return the output
     */
    abstract _call(
        values: RunInput,
        runManager?: CallbackManagerForChainRun,
        config?: RunnableConfig
    ): Promise<RunOutput>

    /**
     * Return the string type key uniquely identifying this class of chain.
     */
    abstract _chainType(): string

    /**
     * Return a json-like object representing this chain.
     */
    serialize(): unknown {
        throw new Error('Method not implemented.')
    }

    abstract get inputKeys(): string[]

    abstract get outputKeys(): string[]

    protected async _formatValues(
        values: ChainValues & { signal?: AbortSignal; timeout?: number }
    ) {
        const fullValues = { ...values } as typeof values
        if (fullValues.timeout && !fullValues.signal) {
            fullValues.signal = AbortSignal.timeout(fullValues.timeout)
            delete fullValues.timeout
        }
        if (!(this.memory == null)) {
            const newValues = await this.memory.loadMemoryVariables(
                this._selectMemoryInputs(values)
            )
            for (const [key, value] of Object.entries(newValues)) {
                fullValues[key] = value
            }
        }
        return fullValues
    }

    /**
     * Load a chain from a json-like object describing it.
     */
    static async deserialize(
        data: unknown,
        values: LoadValues = {}
    ): Promise<BaseChain> {
        throw new Error('Method not implemented.')
    }
}

export class ChatLunaLLMChain<
        RunInput extends ChainValues = ChainValues,
        RunOutput extends ChainValues = ChainValues
    >
    extends BaseChain<RunInput, RunOutput>
    implements ChatLunaLLMChainInput
{
    // eslint-disable-next-line @typescript-eslint/naming-convention
    lc_serializable = false

    prompt: BasePromptTemplate

    llm: ChatLunaChatModel

    outputKey = 'text'

    llmKwargs?: this['llm']['ParsedCallOptions']

    get inputKeys() {
        return this.prompt.inputVariables
    }

    get outputKeys() {
        return [this.outputKey]
    }

    constructor(fields: ChatLunaLLMChainInput) {
        super(fields)
        this.prompt = fields.prompt
        this.llm = fields.llm
        this.memory = fields.memory
        this.outputKey = fields.outputKey ?? this.outputKey
        this.llmKwargs = fields.llmKwargs
    }

    /** @ignore */
    async _call(
        values: RunInput & this['llm']['ParsedCallOptions'],
        runManager?: CallbackManagerForChainRun
    ): Promise<RunOutput> {
        const valuesForPrompt = { ...values }
        const valuesForLLM: this['llm']['ParsedCallOptions'] = {
            ...this.llmKwargs
        }

        for (const key of this.llm.callKeys) {
            if (key in values) {
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                valuesForLLM[key as any] = values[key]
                delete valuesForPrompt[key]
            }
        }
        const promptValue = await this.prompt.formatPromptValue(valuesForPrompt)
        const { generations } = await this.llm.generatePrompt(
            [promptValue],
            valuesForLLM,
            runManager?.getChild()
        )

        const generation = generations[0][0]

        return {
            [this.outputKey]: generation.text,
            rawGeneration: generation,
            message: (generation as ChatGeneration).message,
            extra: generation?.generationInfo
        } as unknown as RunOutput
    }

    /** @ignore */
    _selectMemoryInputs(values: ChainValues): ChainValues {
        const valuesForMemory = this._selectMemoryInputs(values)
        for (const key of this.llm.callKeys) {
            if (key in values) {
                delete valuesForMemory[key]
            }
        }
        return valuesForMemory
    }

    _chainType() {
        return 'chatluna_chain' as const
    }

    // eslint-disable-next-line @typescript-eslint/naming-convention
    get lc_namespace(): string[] {
        return ['langchain', 'chains', this._chainType()]
    }
}

export async function callChatLunaChain(
    chain: ChatLunaLLMChain,
    values: ChainValues & ChatLunaLLMChain['llm']['ParsedCallOptions'],
    events: ChatEvents
): Promise<ChainValues> {
    let usedToken = 0

    const response = await chain.invoke(values, {
        callbacks: [
            {
                handleLLMNewToken(token: string) {
                    events?.['llm-new-token']?.(token)
                },
                handleLLMEnd(output, runId, parentRunId, tags) {
                    usedToken += output.llmOutput?.tokenUsage?.totalTokens ?? 0
                }
            }
        ]
    })

    await events?.['llm-used-token-count'](usedToken)

    return response
}
