import { Time } from 'koishi'
import {
    ChatLunaError,
    ChatLunaErrorCode
} from 'koishi-plugin-chatluna/utils/error'
import { ObjectLock } from 'koishi-plugin-chatluna/utils/lock'
import { withResolver } from 'koishi-plugin-chatluna/utils/promise'

interface QueueItem {
    requestId: string
    timestamp: number
    notifyPromise: {
        promise: Promise<void>
        resolve: () => void
        reject: (error: Error) => void
    }
}

export class RequestIdQueue {
    private _queue: Record<string, QueueItem[]> = {}
    private _lock = new ObjectLock(Time.minute * 3)
    private readonly _maxQueueSize = 50
    private readonly _queueTimeout: number

    constructor(queueTimeout = Time.minute * 3) {
        this._queueTimeout = queueTimeout
        setInterval(() => this.cleanup(), queueTimeout)
    }

    public async add(key: string, requestId: string) {
        await this._lock.runLocked(async () => {
            if (!this._queue[key]) {
                this._queue[key] = []
            }

            if (this._queue[key].length >= this._maxQueueSize) {
                throw new ChatLunaError(ChatLunaErrorCode.QUEUE_OVERFLOW)
            }

            const { promise, resolve, reject } = withResolver<void>()

            const queueItem: QueueItem = {
                requestId,
                timestamp: Date.now(),
                notifyPromise: { promise, resolve, reject }
            }

            this._queue[key].push(queueItem)

            // 如果是第一个请求，立即解决
            if (this._queue[key].length === 1) {
                resolve()
            }
        })
    }

    public async remove(key: string, requestId: string) {
        await this._lock.runLocked(async () => {
            if (!this._queue[key]) return

            const index = this._queue[key].findIndex(
                (item) => item.requestId === requestId
            )

            if (index !== -1) {
                this._queue[key].splice(index, 1)

                // 通知下一个请求（如果存在）
                if (index === 0 && this._queue[key].length > 0) {
                    this._queue[key][0].notifyPromise.resolve()
                }
            }
        })
    }

    public async wait(key: string, requestId: string, maxConcurrent: number) {
        if (!this._queue[key]) {
            await this.add(key, requestId)
        }

        const startTime = Date.now()

        try {
            await this._lock.runLocked(async () => {
                const index = this._queue[key].findIndex(
                    (item) => item.requestId === requestId
                )

                if (index === -1 || index < maxConcurrent) {
                    return // 已经可以执行
                }

                const item = this._queue[key][index]

                // 使用 add 时创建的 promise
                await Promise.race([
                    item.notifyPromise.promise,
                    // eslint-disable-next-line promise/param-names
                    new Promise((_, reject) =>
                        setTimeout(
                            () =>
                                reject(
                                    new Error(
                                        `Queue wait timeout after ${this._queueTimeout}ms`
                                    )
                                ),
                            Math.max(
                                0,
                                this._queueTimeout - (Date.now() - startTime)
                            )
                        )
                    )
                ])
            })
        } catch (error) {
            await this.remove(key, requestId)
            throw error
        }
    }

    private async cleanup() {
        const now = Date.now()
        await this._lock.runLocked(async () => {
            for (const key in this._queue) {
                const expiredItems = this._queue[key].filter(
                    (item) => now - item.timestamp >= this._queueTimeout
                )

                // 通知所有过期项
                expiredItems.forEach((item) => {
                    item.notifyPromise.reject(
                        new Error(
                            `Queue wait timeout after ${this._queueTimeout}ms`
                        )
                    )
                })

                // 移除过期项
                this._queue[key] = this._queue[key].filter(
                    (item) => now - item.timestamp < this._queueTimeout
                )

                // 如果队列头部被清理，通知新的头部
                if (this._queue[key].length > 0) {
                    this._queue[key][0].notifyPromise.resolve()
                }
            }
        })
    }

    public async getQueueLength(key: string) {
        return await this._lock.runLocked(
            async () => this._queue[key]?.length ?? 0
        )
    }

    public async getQueueStatus(key: string) {
        return await this._lock.runLocked(async () => ({
            length: this._queue[key]?.length ?? 0,
            items:
                this._queue[key]?.map((item) => ({
                    requestId: item.requestId,
                    age: Date.now() - item.timestamp
                })) ?? []
        }))
    }
}
