import { h, Session } from 'koishi'

export interface ConversationRoom {
    visibility: 'public' | 'private' | 'template_clone'
    roomMasterId: string
    roomName: string
    roomId: number
    conversationId?: string
    preset: string
    model: string
    chatMode: string
    password?: string
    autoUpdate?: boolean
    updatedTime: Date

    // allowGroups?: string[]
    // allowUsers?: string[]
}

export interface ConversationRoomMemberInfo {
    userId: string
    roomId: number
    mute?: boolean
    roomPermission: 'owner' | 'admin' | 'member'
}

export interface ConversationRoomGroupInfo {
    groupId: string
    roomId: number
    roomVisibility: 'public' | 'private' | 'template_clone'
}

export interface ConversationRoomUserInfo {
    groupId?: string
    defaultRoomId: number
    userId: string
}

/**
 * 渲染参数
 */
export interface RenderOptions {
    // 如果type为voice，那么这个值不可为空
    voice?: {
        speakerId?: number
    }
    split?: boolean
    type: RenderType
    session?: Session
}

export interface Message {
    content: string

    conversationId?: string

    name?: string

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    additional_kwargs?: Record<string, any>

    /**
     * 附加消息回复
     */
    additionalReplyMessages?: Message[]
}

export interface RenderMessage {
    element: h | h[]
}

export type RenderType = 'raw' | 'voice' | 'text' | 'image' | 'mixed'
