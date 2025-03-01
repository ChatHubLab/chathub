// 记忆类型枚举
export enum MemoryType {
    FACTUAL = 'factual', // 事实性知识（长期有效）
    PREFERENCE = 'preference', // 用户偏好（长期有效）
    PERSONAL = 'personal', // 个人信息（长期有效）
    CONTEXTUAL = 'contextual', // 上下文相关（中期有效）
    TEMPORAL = 'temporal', // 时间相关（短期有效）
    TASK = 'task', // 任务相关（中期有效）
    SKILL = 'skill', // 技能相关（长期有效）
    INTEREST = 'interest', // 兴趣爱好（长期有效）
    HABIT = 'habit', // 习惯相关（长期有效）
    EVENT = 'event', // 事件相关（短期有效）
    LOCATION = 'location', // 位置相关（中期有效）
    RELATIONSHIP = 'relationship' // 关系相关（长期有效）
}

// 记忆结构接口
export interface EnhancedMemory {
    content: string // 记忆内容
    type: MemoryType // 记忆类型
    importance: number // 重要性 (1-10)
    expirationDate?: Date // 过期时间（可选）
    rawId?: string // 原始ID
}

export enum MemoryRetrievalLayerType {
    GLOBAL = 'global',
    PRESET = 'preset',
    PRESET_USER = 'preset-user',
    USER = 'user'
}

export interface MemoryRetrievalLayerInfo<
    T extends MemoryRetrievalLayerType = MemoryRetrievalLayerType
> {
    userId: T extends MemoryRetrievalLayerType.PRESET_USER
        ? string
        : T extends MemoryRetrievalLayerType.PRESET
          ? string
          : never
    presetId: T extends MemoryRetrievalLayerType.PRESET
        ? string
        : T extends MemoryRetrievalLayerType.PRESET_USER
          ? string
          : never
    memoryId: string
    type: T
}
