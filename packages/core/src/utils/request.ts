import { HttpsProxyAgent } from 'https-proxy-agent'
import { logger } from 'koishi-plugin-chatluna'
import { lookup } from 'node:dns/promises'
import {
    ChatLunaError,
    ChatLunaErrorCode
} from 'koishi-plugin-chatluna/utils/error'
import { SocksProxyAgent } from 'socks-proxy-agent'
import unidci, { Agent, buildConnector, FormData, ProxyAgent } from 'undici'
import * as fetchType from 'undici/types/fetch'
import { ClientRequestArgs } from 'http'
import { ClientOptions, WebSocket } from 'ws'
import { SocksClient, SocksClientOptions, SocksProxy } from 'socks'
import Connector = buildConnector.connector
import TLSOptions = buildConnector.BuildOptions

export { FormData }

function createProxyAgentForFetch(
    init: fetchType.RequestInit,
    proxyAddress: string
): fetchType.RequestInit {
    if (init.dispatcher || globalProxyAddress == null) {
        return init
    }

    let proxyAddressURL: URL

    try {
        proxyAddressURL = new URL(proxyAddress)
    } catch (e) {
        logger?.error(
            'Unable to parse your proxy address, please check if your proxy address is correct! (e.g., did you add http://)'
        )
        logger?.error(e)
        throw e
    }

    if (proxyAddress.startsWith('socks')) {
        init.dispatcher = socksDispatcher(proxyAddressURL)
        // match http/https
    } else if (proxyAddress.match(/^https?:\/\//)) {
        init.dispatcher = new ProxyAgent({
            uri: proxyAddress
        })
    } else {
        throw new ChatLunaError(
            ChatLunaErrorCode.UNSUPPORTED_PROXY_PROTOCOL,
            new Error('Unsupported proxy protocol')
        )
    }

    // koishi now use undici, never set the global scheduler!!!

    // global[Symbol.for('undici.globalDispatcher.1')] = init.dispatcher
    // setGlobalDispatcher(init.dispatcher)

    return init
}

function createProxyAgent(
    proxyAddress: string
): HttpsProxyAgent<string> | SocksProxyAgent {
    // sock[4/5]
    if (proxyAddress.match(/^socks/)) {
        return new SocksProxyAgent(proxyAddress)
    } else if (proxyAddress.match(/^https?:\/\//)) {
        return new HttpsProxyAgent(proxyAddress)
    } else {
        throw new ChatLunaError(
            ChatLunaErrorCode.UNSUPPORTED_PROXY_PROTOCOL,
            new Error('Unsupported proxy protocol')
        )
    }
}

export let globalProxyAddress: string | null = global['globalProxyAddress']

export function setGlobalProxyAddress(address: string) {
    if (address.match(/^socks/) || address.match(/^https?:\/\//)) {
        globalProxyAddress = address
        global['globalProxyAddress'] = address
    } else {
        throw new ChatLunaError(
            ChatLunaErrorCode.UNSUPPORTED_PROXY_PROTOCOL,
            new Error('Unsupported proxy protocol')
        )
    }
}

/**
 * package undici, and with proxy support
 * @returns
 */
export function chatLunaFetch(
    info: fetchType.RequestInfo,
    init?: fetchType.RequestInit,
    proxyAddress: string = globalProxyAddress
) {
    if (proxyAddress !== 'null' && proxyAddress != null && !init?.dispatcher) {
        init = createProxyAgentForFetch(init || {}, proxyAddress)
    }

    try {
        return unidci.fetch(info, init)
    } catch (e) {
        if (e.cause) {
            logger.error(e.cause)
        }
        throw e
    }
}

/**
 * package ws, and with proxy support
 */
export function ws(
    url: string,
    options?: ClientOptions | ClientRequestArgs,
    proxyAddress: string = globalProxyAddress
) {
    if (proxyAddress !== 'null' && proxyAddress != null && !options?.agent) {
        options = options || {}
        options.agent = createProxyAgent(proxyAddress)
    }
    return new WebSocket(url, options)
}

export function randomUA() {
    const browsers = ['Chrome', 'Edg']
    const browser = browsers[Math.floor(Math.random() * browsers.length)]

    const chromeVersions = [
        '90',
        '91',
        '92',
        '93',
        '94',
        '95',
        '96',
        '97',
        '98',
        '99',
        '100',
        '101',
        '102',
        '103'
    ]
    const edgeVersions = [
        '90',
        '91',
        '92',
        '93',
        '94',
        '95',
        '96',
        '97',
        '98',
        '99',
        '100',
        '101',
        '102',
        '103'
    ]

    const version =
        browser === 'Chrome'
            ? chromeVersions[Math.floor(Math.random() * chromeVersions.length)]
            : edgeVersions[Math.floor(Math.random() * edgeVersions.length)]

    const osVersions = ['10.0', '11.0']
    const osVersion = osVersions[Math.floor(Math.random() * osVersions.length)]

    return `Mozilla/5.0 (Windows NT ${osVersion}; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) ${browser}/${version}.0.0.0 Safari/537.36`
}

// See https://github.com/TooTallNate/proxy-agents/blob/main/packages/socks-proxy-agent/src/index.ts

function parseSocksURL(url: URL): { shouldLookup: boolean; proxy: SocksProxy } {
    let lookup = false
    let type: SocksProxy['type'] = 5
    const host = url.hostname

    // From RFC 1928, Section 3: https://tools.ietf.org/html/rfc1928#section-3
    // "The SOCKS service is conventionally located on TCP port 1080"
    const port = parseInt(url.port, 10) || 1080

    // figure out if we want socks v4 or v5, based on the "protocol" used.
    // Defaults to 5.
    switch (url.protocol.replace(':', '')) {
        case 'socks4':
            lookup = true
            type = 4
            break
        // pass through
        case 'socks4a':
            type = 4
            break
        case 'socks5':
            lookup = true
            type = 5
            break
        // pass through
        case 'socks': // no version specified, default to 5h
            type = 5
            break
        case 'socks5h':
            type = 5
            break
        default:
            throw new TypeError(
                `A "socks" protocol must be specified! Got: ${String(
                    url.protocol
                )}`
            )
    }

    const proxy: SocksProxy = {
        host,
        port,
        type
    }

    if (url.username) {
        Object.defineProperty(proxy, 'userId', {
            value: decodeURIComponent(url.username),
            enumerable: false
        })
    }

    if (url.password != null) {
        Object.defineProperty(proxy, 'password', {
            value: decodeURIComponent(url.password),
            enumerable: false
        })
    }

    return { shouldLookup: lookup, proxy }
}

// see https://github.com/Kaciras/fetch-socks/blob/master/index.ts

export type SocksProxies = URL

/**
 * Since socks does not guess HTTP ports, we need to do that.
 *
 * @param protocol Upper layer protocol, "http:" or "https:"
 * @param port A string containing the port number of the URL, maybe empty.
 */
function resolvePort(protocol: string, port: string) {
    return port ? Number.parseInt(port) : protocol === 'http:' ? 80 : 443
}

/**
 * Create an Undici connector which establish the connection through socks proxies.
 *
 * If the proxies is an empty array, it will connect directly.
 *
 * @param proxies The proxy server to use or the list of proxy servers to chain.
 * @param tlsOpts TLS upgrade options.
 */
export function socksConnector(url: URL, tlsOpts: TLSOptions = {}): Connector {
    const { timeout = 1e4 } = tlsOpts
    const undiciConnect = buildConnector(tlsOpts)

    return async (options, callback) => {
        let { protocol, hostname, port, httpSocket } = options

        const { proxy, shouldLookup } = parseSocksURL(url)

        if (shouldLookup) {
            hostname = (await lookup(hostname)).address
        }

        const destination = {
            host: hostname,
            port: resolvePort(protocol, port)
        }

        const socksOpts: SocksClientOptions = {
            proxy,
            destination,
            command: 'connect',
            timeout,
            existing_socket: httpSocket
        }

        try {
            const r = await SocksClient.createConnection(socksOpts)
            httpSocket = r.socket
        } catch (error) {
            return callback(error, null)
        }

        // httpSocket may not exist when the chain is empty.
        if (httpSocket && protocol !== 'https:') {
            return callback(null, httpSocket.setNoDelay())
        }

        /*
         * There are 2 cases here:
         * If httpSocket doesn't exist, let Undici make a connection.
         * If httpSocket exists & protocol is HTTPS, do TLS upgrade.
         */
        return undiciConnect({ ...options, httpSocket }, callback)
    }
}

export interface SocksDispatcherOptions extends Agent.Options {
    /**
     * TLS upgrade options, see:
     * https://undici.nodejs.org/#/docs/api/Client?id=parameter-connectoptions
     *
     * The connect function is not supported.
     * If you want to create a custom connector, you can use `socksConnector`.
     */
    connect?: TLSOptions
}

/**
 * Create a Undici Agent with socks connector.
 *
 * If the proxies is an empty array, it will connect directly.
 *
 * @param proxies The proxy server to use or the list of proxy servers to chain.
 * @param options Additional options passed to the Agent constructor.
 */
export function socksDispatcher(
    proxies: SocksProxies,
    options: SocksDispatcherOptions = {}
) {
    const { connect, ...rest } = options
    return new Agent({ ...rest, connect: socksConnector(proxies, connect) })
}
