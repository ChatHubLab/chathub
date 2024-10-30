import {
    getEncodingNameForModel,
    Tiktoken,
    TiktokenBPE,
    TiktokenEncoding,
    TiktokenModel
} from 'js-tiktoken/lite'
import { chatLunaFetch } from 'koishi-plugin-chatluna/utils/request'

const cache: Record<string, TiktokenBPE> = {}

export async function getEncoding(
    encoding: TiktokenEncoding,
    options?: {
        signal?: AbortSignal
        extendedSpecialTokens?: Record<string, number>
    }
) {
    if (!(encoding in cache)) {
        cache[encoding] = await chatLunaFetch(
            `https://tiktoken.pages.dev/js/${encoding}.json`,
            {
                signal: options?.signal
            }
        )
            .then((res) => res.json() as unknown as TiktokenBPE)
            .catch((e) => {
                delete cache[encoding]
                throw e
            })
    }

    return new Tiktoken(cache[encoding], options?.extendedSpecialTokens)
}

export async function encodingForModel(
    model: TiktokenModel,
    options?: {
        signal?: AbortSignal
        extendedSpecialTokens?: Record<string, number>
    }
) {
    options = options ?? {}

    let timeout: NodeJS.Timeout

    if (options.signal == null) {
        const abortController = new AbortController()

        options.signal = abortController.signal

        timeout = setTimeout(() => abortController.abort(), 1000 * 5)
    }

    const result = await getEncoding(getEncodingNameForModel(model), options)

    if (timeout != null) {
        clearTimeout(timeout)
    }

    return result
}
