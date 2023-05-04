
import { BasePromptValue, ChatMessage, InputValues, createAIChatMessage, createChatMessage, createHumanChatMessage, createSystemChatMessage, MessageType, PartialValues } from '../common';
import {
    BasePromptTemplate,
    BasePromptTemplateInput,
    BaseStringPromptTemplate,
} from "./base.js";
import { PromptTemplate } from "./prompt.js";
import {
    SerializedChatPromptTemplate,
    SerializedMessagePromptTemplate,
} from "./serde.js";

export abstract class BaseMessagePromptTemplate {
    abstract inputVariables: string[];

    abstract formatMessages(values: InputValues): Promise<ChatMessage[]>;

    serialize(): SerializedMessagePromptTemplate {
        return {
            _type: this.constructor.name,
            ...JSON.parse(JSON.stringify(this)),
        };
    }
}

export class ChatPromptValue extends BasePromptValue {
    messages: ChatMessage[];

    constructor(messages: ChatMessage[]) {
        super();
        this.messages = messages;
    }

    toString() {
        return JSON.stringify(this.messages);
    }

    toChatMessages() {
        return this.messages;
    }
}

export class MessagesPlaceholder extends BaseMessagePromptTemplate {
    variableName: string;

    constructor(variableName: string) {
        super();
        this.variableName = variableName;
    }

    get inputVariables() {
        return [this.variableName];
    }

    formatMessages(values: InputValues): Promise<ChatMessage[]> {
        return Promise.resolve(values[this.variableName] as ChatMessage[]);
    }
}

export abstract class BaseMessageStringPromptTemplate extends BaseMessagePromptTemplate {
    prompt: BaseStringPromptTemplate;

    protected constructor(prompt: BaseStringPromptTemplate) {
        super();
        this.prompt = prompt;
    }

    get inputVariables() {
        return this.prompt.inputVariables;
    }

    abstract format(values: InputValues): Promise<ChatMessage>;

    async formatMessages(values: InputValues): Promise<ChatMessage[]> {
        return [await this.format(values)];
    }
}

export abstract class BaseChatPromptTemplate extends BasePromptTemplate {
    constructor(input: BasePromptTemplateInput) {
        super(input);
    }

    abstract formatMessages(values: InputValues): Promise<ChatMessage[]>;

    async format(values: InputValues): Promise<string> {
        return (await this.formatPromptValue(values)).toString();
    }

    async formatPromptValue(values: InputValues): Promise<BasePromptValue> {
        const resultMessages = await this.formatMessages(values);
        return new ChatPromptValue(resultMessages);
    }
}

export class ChatMessagePromptTemplate extends BaseMessageStringPromptTemplate {
    role: string;

    async format(values: InputValues): Promise<ChatMessage> {
        return createChatMessage(await this.prompt.format(values), "generic", this.role)
    }

    constructor(prompt: BaseStringPromptTemplate, role: string) {
        super(prompt);
        this.role = role;
    }

    static fromTemplate(template: string, role: string) {
        return new this(PromptTemplate.fromTemplate(template), role);
    }
}

export class HumanMessagePromptTemplate extends BaseMessageStringPromptTemplate {
    async format(values: InputValues): Promise<ChatMessage> {
        return createHumanChatMessage(await this.prompt.format(values));
    }

    constructor(prompt: BaseStringPromptTemplate) {
        super(prompt);
    }

    static fromTemplate(template: string) {
        return new this(PromptTemplate.fromTemplate(template));
    }
}

export class AIMessagePromptTemplate extends BaseMessageStringPromptTemplate {
    async format(values: InputValues): Promise<ChatMessage> {
        return createAIChatMessage(await this.prompt.format(values));
    }

    constructor(prompt: BaseStringPromptTemplate) {
        super(prompt);
    }

    static fromTemplate(template: string) {
        return new this(PromptTemplate.fromTemplate(template));
    }
}

export class SystemMessagePromptTemplate extends BaseMessageStringPromptTemplate {
    async format(values: InputValues): Promise<ChatMessage> {
        return createSystemChatMessage(await this.prompt.format(values));
    }

    constructor(prompt: BaseStringPromptTemplate) {
        super(prompt);
    }

    static fromTemplate(template: string) {
        return new this(PromptTemplate.fromTemplate(template));
    }
}

export interface ChatPromptTemplateInput extends BasePromptTemplateInput {
    /**
     * The prompt messages
     */
    promptMessages: BaseMessagePromptTemplate[];

    /**
     * Whether to try validating the template on initialization
     *
     * @defaultValue `true`
     */
    validateTemplate?: boolean;
}

export class ChatPromptTemplate
    extends BaseChatPromptTemplate
    implements ChatPromptTemplateInput {
    promptMessages: BaseMessagePromptTemplate[];

    validateTemplate = true;

    constructor(input: ChatPromptTemplateInput) {
        super(input);
        Object.assign(this, input);

        if (this.validateTemplate) {
            const inputVariablesMessages = new Set<string>();
            for (const promptMessage of this.promptMessages) {
                for (const inputVariable of promptMessage.inputVariables) {
                    inputVariablesMessages.add(inputVariable);
                }
            }
            const inputVariablesInstance = new Set(
                this.partialVariables
                    ? this.inputVariables.concat(Object.keys(this.partialVariables))
                    : this.inputVariables
            );
            const difference = new Set(
                [...inputVariablesInstance].filter(
                    (x) => !inputVariablesMessages.has(x)
                )
            );
            if (difference.size > 0) {
                throw new Error(
                    `Input variables \`${[
                        ...difference,
                    ]}\` are not used in any of the prompt messages.`
                );
            }
            const otherDifference = new Set(
                [...inputVariablesMessages].filter(
                    (x) => !inputVariablesInstance.has(x)
                )
            );
            if (otherDifference.size > 0) {
                throw new Error(
                    `Input variables \`${[
                        ...otherDifference,
                    ]}\` are used in prompt messages but not in the prompt template.`
                );
            }
        }
    }

    _getPromptType(): "chat" {
        return "chat";
    }

    async formatMessages(values: InputValues): Promise<ChatMessage[]> {
        const allValues = await this.mergePartialAndUserVariables(values);

        let resultMessages: ChatMessage[] = [];

        for (const promptMessage of this.promptMessages) {
            const inputValues = promptMessage.inputVariables.reduce(
                (acc, inputVariable) => {
                    if (!(inputVariable in allValues)) {
                        throw new Error(
                            `Missing value for input variable \`${inputVariable}\``
                        );
                    }
                    acc[inputVariable] = allValues[inputVariable];
                    return acc;
                },
                {} as InputValues
            );
            const message = await promptMessage.formatMessages(inputValues);
            resultMessages = resultMessages.concat(message);
        }
        return resultMessages;
    }

    serialize(): SerializedChatPromptTemplate {
        if (this.outputParser !== undefined) {
            throw new Error(
                "ChatPromptTemplate cannot be serialized if outputParser is set"
            );
        }
        return {
            input_variables: this.inputVariables,
            prompt_messages: this.promptMessages.map((m) => m.serialize()),
        };
    }

    async partial(values: PartialValues): Promise<ChatPromptTemplate> {
        // This is implemented in a way it doesn't require making
        // BaseMessagePromptTemplate aware of .partial()
        const promptDict: ChatPromptTemplateInput = { ...this };
        promptDict.inputVariables = this.inputVariables.filter(
            (iv) => !(iv in values)
        );
        promptDict.partialVariables = {
            ...(this.partialVariables ?? {}),
            ...values,
        };
        return new ChatPromptTemplate(promptDict);
    }

    static fromPromptMessages(
        promptMessages: (BaseMessagePromptTemplate | ChatPromptTemplate)[]
    ): ChatPromptTemplate {
        const flattenedMessages = promptMessages.reduce(
            (acc, promptMessage) =>
                acc.concat(
                    // eslint-disable-next-line no-instanceof/no-instanceof
                    promptMessage instanceof ChatPromptTemplate
                        ? promptMessage.promptMessages
                        : [promptMessage]
                ),
            [] as BaseMessagePromptTemplate[]
        );
        const flattenedPartialVariables = promptMessages.reduce(
            (acc, promptMessage) =>
                // eslint-disable-next-line no-instanceof/no-instanceof
                promptMessage instanceof ChatPromptTemplate
                    ? Object.assign(acc, promptMessage.partialVariables)
                    : acc,
            Object.create(null) as PartialValues
        );

        const inputVariables = new Set<string>();
        for (const promptMessage of flattenedMessages) {
            for (const inputVariable of promptMessage.inputVariables) {
                if (inputVariable in flattenedPartialVariables) {
                    continue;
                }
                inputVariables.add(inputVariable);
            }
        }
        return new ChatPromptTemplate({
            inputVariables: [...inputVariables],
            promptMessages: flattenedMessages,
            partialVariables: flattenedPartialVariables,
        });
    }
}