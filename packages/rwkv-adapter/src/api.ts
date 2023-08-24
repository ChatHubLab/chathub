import { Dict } from 'koishi'
import { request } from '@dingyi222666/koishi-plugin-chathub/lib/llm-core/utils/request'
import { createLogger } from '@dingyi222666/koishi-plugin-chathub/lib/llm-core/utils/logger'
import { BaseMessage, MessageType } from 'langchain/schema'
import RWKVPlugin from './index'

const logger = createLogger('@dingyi222666/chathub-rwkv-adapter/api')

export class Api {

    constructor(
        private readonly config: RWKVPlugin.Config
    ) { }

    private _buildHeaders() {
        return {
            Authorization: `Bearer ${this.config.apiKey}`,
            "Content-Type": "application/json"
        }
    }

    private _concatUrl(url: string): string {
        const apiEndPoint = this.config.apiEndPoint


        // match the apiEndPoint ends with '/v1' or '/v1/' using regex
        if (!apiEndPoint.match(/\/v1\/?$/)) {
            if (apiEndPoint.endsWith('/')) {
                return apiEndPoint + 'v1/' + url
            }

            return apiEndPoint + '/v1/' + url
        }

        if (apiEndPoint.endsWith('/')) {
            return apiEndPoint + url
        }

        return apiEndPoint + '/' + url

    }

    private _get(url: string) {
        const requestUrl = this._concatUrl(url)

        return request.fetch(requestUrl, {
            method: 'GET',
            headers: this._buildHeaders()
        })
    }

    private _post(urL: string, data: any, params: Record<string, any> = {}) {
        const requestUrl = this._concatUrl(urL)

        return request.fetch(requestUrl, {
            body: JSON.stringify(data),
            headers: this._buildHeaders(),
            method: 'POST',
            ...params
        })
    }


    async listModels(): Promise<string[] | null> {
        try {
            const response = await this._get("models")
            const data = (<any>(await response.json()))

            // logger.debug(JSON.stringify(data))

            return (<Dict<string, any>[]>(data.data)).map((model) => model.id)
        } catch (e) {

            logger.error(
                "Error when listing openai models, Result: " + e.response
                    ? (e.response ? e.response.data : e)
                    : e
            );

            // return fake empty models
            return null
        }
    }


    async chatTrubo(
        model: string,
        messages: BaseMessage[],
        signal?: AbortSignal
    ) {
        let data: {
            choices: Array<{
                index: number;
                finish_reason: string | null;
                delta: { content?: string; role?: string };
                message: { role: string, content: string }
            }>; id: string; object: string; created: number; model: string; usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number }
        }
        try {
            const response = await this._post("chat/completions", {
                model: model,
                messages: messages.map((message) => {
                    return {
                        role: messageTypeToOpenAIRole(message._getType()),
                        content: message.content
                    }
                }),
                max_tokens: this.config.maxTokens,

                user: "user"
            }, {
                signal: signal
            })

            data = (await response.json()) as {
                id: string;
                object: string;
                created: number;
                model: string;
                choices: Array<{
                    index: number;
                    finish_reason: string | null;
                    delta: { content?: string; role?: string };
                    message: { role: string, content: string }
                }>;
                usage: {
                    prompt_tokens: number,
                    completion_tokens: number,
                    total_tokens: number
                }
            };


            if (data.choices && data.choices.length > 0) {
                return data
            }

            throw new Error("error when calling rwkv chat, Result: " + JSON.stringify(data))

        } catch (e) {

            logger.error(data)
            logger.error(
                "Error when calling rwkv chat, Result: " + e.response
                    ? (e.response ? e.response.data : e)
                    : e
            );


            return null
        }
    }


    async embeddings({
        model,
        input
    }: { model: string, input: string[] | string }) {
        let data: {
            id: string;
            object: string;
            created: number;
            model: string;
            data: Array<{
                embedding: number[];
                object: string | null;
                index: number
            }>;
            usage: {
                prompt_tokens: number,
                completion_tokens: number,
                total_tokens: number
            }
        };

        try {
            const response = await this._post("embeddings", {
                input,
                model
            })

            data = (await response.json()) as {
                id: string;
                object: string;
                created: number;
                model: string;
                data: Array<{
                    embedding: number[];
                    object: string | null;
                    index: number
                }>;
                usage: {
                    prompt_tokens: number,
                    completion_tokens: number,
                    total_tokens: number
                }
            };

            if (data.data && data.data.length > 0) {
                return data
            }

            throw new Error("error when calling rwkv embeddings, Result: " + JSON.stringify(data))

        } catch (e) {

            logger.error(data)
            logger.error(
                "Error when calling rwkv embeddings, Result: " + e.response
                    ? (e.response ? e.response.data : e)
                    : e
            );

            return null
        }

    }
}

export function messageTypeToOpenAIRole(
    type: MessageType
): string {
    switch (type) {
        case "system":
            return "system";
        case "ai":
            return "assistant";
        case "human":
            return "user";
        default:
            throw new Error(`Unknown message type: ${type}`);
    }
}