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

            // If it is the first request, resolve immediately
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

                // If an item is removed from any position in the queue, the subsequent requests need to be notified to move forward
                this._queue[key].slice(index).forEach((item, newIndex) => {
                    // If the new position is less than maxConcurrent (determined in wait), resolve the promise
                    // Note: The specific logic here may need to be determined based on maxConcurrent in wait
                    // However, since maxConcurrent is passed in wait, it cannot be accessed here,
                    // so we only handle the most explicit case: the head of the queue
                    if (index + newIndex === 0) {
                        item.notifyPromise.resolve()
                    }
                })
            }
        })
    }

    public async wait(key: string, requestId: string, maxConcurrent: number) {
        if (!this._queue[key]) {
            await this.add(key, requestId)
        }

        let item: QueueItem | undefined

        // First, get the project information within the lock
        await this._lock.runLocked(async () => {
            const index = this._queue[key].findIndex(
                (item) => item.requestId === requestId
            )

            if (index === -1) return // Request not in queue
            if (index < maxConcurrent) {
                // If within the concurrency limit, resolve immediately
                this._queue[key][index].notifyPromise.resolve()
                return
            }

            item = this._queue[key][index]
        })

        // If you need to wait, wait outside the lock
        if (item) {
            try {
                const timeoutId = setTimeout(
                    () =>
                        item.notifyPromise.reject(
                            new Error(
                                `Queue wait timeout after ${this._queueTimeout}ms`
                            )
                        ),
                    this._queueTimeout
                )

                try {
                    await item.notifyPromise.promise
                } finally {
                    clearTimeout(timeoutId)
                }
            } catch (error) {
                await this.remove(key, requestId)
                throw error
            }
        }
    }

    private async cleanup() {
        const now = Date.now()
        await this._lock.runLocked(async () => {
            for (const key in this._queue) {
                const expiredItems = this._queue[key].filter(
                    (item) => now - item.timestamp >= this._queueTimeout
                )

                // Notify all expired items
                expiredItems.forEach((item) => {
                    item.notifyPromise.reject(
                        new Error(
                            `Queue wait timeout after ${this._queueTimeout}ms`
                        )
                    )
                })

                // Remove expired items
                this._queue[key] = this._queue[key].filter(
                    (item) => now - item.timestamp < this._queueTimeout
                )

                // If the head of the queue is cleaned up, notify the new head
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
