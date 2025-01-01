import { logger as koishiLogger } from 'koishi-plugin-chatluna'

// eslint-disable-next-line prefer-const
export let ERROR_FORMAT_TEMPLATE =
    '使用 ChatLuna 时出现错误，错误码为 %s。请联系开发者以解决此问题。'

export const setErrorFormatTemplate = (template: string | null) => {
    if (template) {
        ERROR_FORMAT_TEMPLATE = template
    }
}

export class ChatLunaError extends Error {
    constructor(
        public errorCode: ChatLunaErrorCode = ChatLunaErrorCode.UNKNOWN_ERROR,
        public originError?: Error
    ) {
        super(ERROR_FORMAT_TEMPLATE.replace('%s', errorCode.toString()))

        this.name = 'ChatLunaError'
        const logger = koishiLogger ?? console
        logger.error(
            '='.repeat(20) + 'ChatLunaError:' + errorCode + '='.repeat(20)
        )
        if (originError) {
            logger.error(originError)
            if (originError.cause) {
                logger.error(originError.cause)
            }
        } else {
            logger.error(this)
        }
    }

    public toString() {
        return this.message
    }
}

export enum ChatLunaErrorCode {
    NETWORK_ERROR = 1,
    UNSUPPORTED_PROXY_PROTOCOL = 2,
    QUEUE_OVERFLOW = 3,
    RENDER_ERROR = 4,
    ABORTED = 5,
    PLUGIN_ALREADY_REGISTERED = 6,
    API_KEY_UNAVAILABLE = 100,
    API_REQUEST_RESOLVE_CAPTCHA = 101,
    API_REQUEST_TIMEOUT = 102,
    API_REQUEST_FAILED = 103,
    API_UNSAFE_CONTENT = 104,
    MODEL_ADAPTER_NOT_FOUND = 300,
    MODEL_NOT_FOUND = 301,
    PREST_NOT_FOUND = 302,
    MODEL_INIT_ERROR = 303,
    EMBEDDINGS_INIT_ERROR = 304,
    VECTOR_STORE_INIT_ERROR = 305,
    CHAT_HISTORY_INIT_ERROR = 306,
    NOT_AVAILABLE_CONFIG = 307,
    MODEL_CONVERSION_INIT_ERROR = 308,
    MODEL_RESPONSE_IS_EMPTY = 309,
    PRESET_LOAD_ERROR = 311,
    MODEL_DEPOSE_ERROR = 310,
    LONG_MEMORY_INIT_ERROR = 312,
    MEMBER_NOT_IN_ROOM = 400,
    ROOM_NOT_JOINED = 401,
    ROOM_NOT_FOUND_MASTER = 402,
    ROOM_TEMPLATE_INVALID = 403,
    THE_NAME_FIND_IN_MULTIPLE_ROOMS = 404,
    ROOM_NOT_FOUND = 405,
    INIT_ROOM = 406,
    KNOWLEDGE_CONFIG_INVALID = 500,
    KNOWLEDGE_DOC_NOT_FOUND = 501,
    KNOWLEDGE_LOOP_INCLUDE = 502,
    KNOWLEDGE_UNSUPPORTED_FILE_TYPE = 503,
    KNOWLEDGE_EXIST_FILE = 504,
    KNOWLEDGE_VECTOR_NOT_FOUND = 505,
    USER_NOT_FOUND = 600,
    AUTH_GROUP_NOT_FOUND = 601,
    AUTH_GROUP_NOT_JOINED = 602,
    AUTH_GROUP_ALREADY_JOINED = 603,
    UNKNOWN_ERROR = 999
}
