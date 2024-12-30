import {
    getEncodingNameForModel,
    Tiktoken,
    TiktokenBPE,
    TiktokenEncoding,
    TiktokenModel
} from 'js-tiktoken/lite'
import {
    chatLunaFetch,
    globalProxyAddress
} from 'koishi-plugin-chatluna/utils/request'

const cache: Record<string, TiktokenBPE> = {}

export async function getEncoding(
    encoding: TiktokenEncoding,
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

    if (!(encoding in cache)) {
        const url =
            globalProxyAddress.length > 0
                ? `https://tiktoken.pages.dev/js/${encoding}.json`
                : `https://jsd.onmicrosoft.cn/npm/tiktoken@latest/encoders/${encoding}.json`

        cache[encoding] = await chatLunaFetch(url, {
            signal: options?.signal
        })
            .then((res) => res.json() as unknown as TiktokenBPE)
            .catch((e) => {
                delete cache[encoding]
                throw e
            })
    }

    if (timeout != null) {
        clearTimeout(timeout)
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
    const result = await getEncoding(getEncodingNameForModel(model), options)

    return result
}
