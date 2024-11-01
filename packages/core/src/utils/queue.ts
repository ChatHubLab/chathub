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
    private _queueLocks: Record<string, ObjectLock> = {}
    private readonly _maxQueueSize = 50
    private readonly _queueTimeout: number

    constructor(queueTimeout = Time.minute * 3) {
        this._queueTimeout = queueTimeout
        setInterval(() => this.cleanup(), queueTimeout)
    }

    public async add(key: string, requestId: string) {
        // Fast path: check queue size without lock first
        const currentLength = this._queue[key]?.length ?? 0
        if (currentLength >= this._maxQueueSize) {
            throw new ChatLunaError(ChatLunaErrorCode.QUEUE_OVERFLOW)
        }

        // Get or create lock for this specific queue
        if (!this._queueLocks[key]) {
            this._queueLocks[key] = new ObjectLock(this._queueTimeout)
        }

        // Prepare the queue item outside the lock
        const { promise, resolve, reject } = withResolver<void>()
        const queueItem: QueueItem = {
            requestId,
            timestamp: Date.now(),
            notifyPromise: { promise, resolve, reject }
        }

        let isFirst = false

        try {
            await this._queueLocks[key].runLocked(async () => {
                // Initialize queue if needed
                if (!this._queue[key]) {
                    this._queue[key] = []
                }

                // Check if requestId already exists
                const existingIndex = this._queue[key].findIndex(
                    (item) => item.requestId === requestId
                )
                if (existingIndex !== -1) {
                    return // Skip if already exists
                }

                // Double check size under lock
                if (this._queue[key].length >= this._maxQueueSize) {
                    throw new ChatLunaError(ChatLunaErrorCode.QUEUE_OVERFLOW)
                }

                this._queue[key].push(queueItem)
                isFirst = this._queue[key].length === 1
            })

            // Resolve immediately if it's the first item (outside lock)
            if (isFirst) {
                resolve()
            }
        } catch (error) {
            reject(error)
            throw error
        }
    }

    public async remove(key: string, requestId: string) {
        // Skip if queue doesn't exist
        if (!this._queue[key]) return

        // Get or create lock for this specific queue
        if (!this._queueLocks[key]) {
            this._queueLocks[key] = new ObjectLock(this._queueTimeout)
        }

        let nextItem: QueueItem | undefined
        let shouldCleanup = false

        try {
            await this._queueLocks[key].runLocked(async () => {
                if (!this._queue[key]) return

                const index = this._queue[key].findIndex(
                    (item) => item.requestId === requestId
                )

                if (index === -1) return

                // Remove the item
                this._queue[key].splice(index, 1)

                // Check if we need to cleanup
                if (this._queue[key].length === 0) {
                    shouldCleanup = true
                    return
                }

                // Get next item if we removed the first item
                if (index === 0 && this._queue[key].length > 0) {
                    nextItem = this._queue[key][0]
                }
            })

            // Perform cleanup outside the lock if needed
            if (shouldCleanup) {
                delete this._queue[key]
                delete this._queueLocks[key]
                return
            }

            // Notify next item outside the lock
            if (nextItem) {
                nextItem.notifyPromise.resolve()
            }
        } catch (error) {
            console.error('Error in remove operation:', error)
            // Don't throw here to prevent queue from getting stuck
        }
    }

    public async wait(key: string, requestId: string, maxConcurrent: number) {
        // Fast path: if queue doesn't exist, add directly
        if (!this._queue[key]) {
            await this.add(key, requestId)
            return
        }

        // Get or create lock for this specific queue
        if (!this._queueLocks[key]) {
            this._queueLocks[key] = new ObjectLock(this._queueTimeout)
        }

        let item: QueueItem | undefined
        let shouldExecute = false

        // Get queue item information within the lock
        await this._queueLocks[key].runLocked(async () => {
            if (!this._queue[key]) return

            const index = this._queue[key].findIndex(
                (item) => item.requestId === requestId
            )

            if (index === -1) return

            // Execute immediately if it's the first item or within concurrent limit
            if (index === 0 || index < maxConcurrent) {
                shouldExecute = true
                return
            }

            item = this._queue[key][index]
        })

        // If should execute immediately, return
        if (shouldExecute) {
            item?.notifyPromise.resolve()
            return
        }

        // Wait for turn
        if (item) {
            let timeoutId: NodeJS.Timeout
            try {
                // eslint-disable-next-line promise/param-names
                const timeoutPromise = new Promise<never>((_, reject) => {
                    timeoutId = setTimeout(() => {
                        reject(
                            new Error(
                                `Queue wait timeout after ${this._queueTimeout}ms`
                            )
                        )
                    }, this._queueTimeout)
                })

                await Promise.race([item.notifyPromise.promise, timeoutPromise])
            } catch (error) {
                await this.remove(key, requestId).catch(() => {
                    /* ignore */
                })
                throw error
            } finally {
                clearTimeout(timeoutId)
            }
        }
    }

    private async cleanup() {
        const now = Date.now()
        const keys = Object.keys(this._queue)

        // Process each queue separately with its own lock
        for (const key of keys) {
            if (!this._queueLocks[key]) {
                this._queueLocks[key] = new ObjectLock(this._queueTimeout)
            }

            await this._queueLocks[key].runLocked(async () => {
                if (!this._queue[key]) return

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

                // If queue becomes empty after cleanup, remove the lock
                if (this._queue[key].length === 0) {
                    delete this._queue[key]
                    delete this._queueLocks[key]
                    return
                }

                // If the head of the queue was cleaned up, notify the new head
                if (this._queue[key].length > 0) {
                    this._queue[key][0].notifyPromise.resolve()
                }
            })
        }
    }

    public async getQueueLength(key: string) {
        // Get or create lock for this specific queue
        if (!this._queueLocks[key]) {
            this._queueLocks[key] = new ObjectLock(this._queueTimeout)
        }

        return await this._queueLocks[key].runLocked(
            async () => this._queue[key]?.length ?? 0
        )
    }

    public async getQueueStatus(key: string) {
        // Get or create lock for this specific queue
        if (!this._queueLocks[key]) {
            this._queueLocks[key] = new ObjectLock(this._queueTimeout)
        }

        return await this._queueLocks[key].runLocked(async () => ({
            length: this._queue[key]?.length ?? 0,
            items:
                this._queue[key]?.map((item) => ({
                    requestId: item.requestId,
                    age: Date.now() - item.timestamp
                })) ?? []
        }))
    }
}
