import path from 'path'
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
import os from 'os'
import fs from 'fs/promises'

const cache: Record<string, TiktokenBPE> = {}

export async function getEncoding(
    encoding: TiktokenEncoding,
    options?: {
        signal?: AbortSignal
        extendedSpecialTokens?: Record<string, number>
    }
) {
    options = options ?? {}

    // pwd + data/chathub/tmps
    const cacheDir = path.resolve(os.tmpdir(), 'chatluna', 'tiktoken')
    const cachePath = path.join(cacheDir, `${encoding}.json`)

    if (cache[encoding]) {
        return new Tiktoken(cache[encoding], options?.extendedSpecialTokens)
    }

    await fs.mkdir(cacheDir, { recursive: true })

    try {
        const cacheContent = await fs.readFile(cachePath, 'utf-8')
        cache[encoding] = JSON.parse(cacheContent)
        return new Tiktoken(cache[encoding], options?.extendedSpecialTokens)
    } catch (e) {
        // ignore
    }

    const url =
        globalProxyAddress.length > 0
            ? `https://tiktoken.pages.dev/js/${encoding}.json`
            : `https://jsd.onmicrosoft.cn/npm/tiktoken@latest/encoders/${encoding}.json`

    cache[encoding] = await chatLunaFetch(url)
        .then((res) => res.json() as unknown as TiktokenBPE)
        .catch((e) => {
            delete cache[encoding]
            throw e
        })

    await fs.writeFile(cachePath, JSON.stringify(cache[encoding]))

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
