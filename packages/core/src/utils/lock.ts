import { Time } from 'koishi'
import { withResolver } from 'koishi-plugin-chatluna/utils/promise'

export class ObjectLock {
    private _lock: boolean = false
    private _queue: {
        id: number
        resolve: () => void
    }[] = []

    private _currentId = 0
    private _currentLockId?: number
    private _lockCount: number = 0
    private readonly _timeout: number
    private readonly _maxQueueLength: number

    constructor(timeout = Time.minute * 3, maxQueueLength = 100) {
        this._timeout = timeout
        this._maxQueueLength = maxQueueLength
    }

    async lock() {
        const id = this._currentId++

        if (this._currentLockId === id || this._currentLockId === id - 1) {
            this._lockCount = Math.max(0, this._lockCount + 1)
            return this._currentLockId!
        }

        if (this._lock) {
            if (this._queue.length >= this._maxQueueLength) {
                throw new Error('Lock queue is full')
            }

            const { promise, resolve } = withResolver<void>()
            this._queue.push({ id, resolve })

            let timeoutId: NodeJS.Timeout | undefined
            try {
                await Promise.race([
                    promise,
                    // eslint-disable-next-line promise/param-names
                    new Promise<never>((_, reject) => {
                        timeoutId = setTimeout(() => {
                            reject(
                                new Error(
                                    `Lock timeout after ${this._timeout}ms`
                                )
                            )
                        }, this._timeout)
                    })
                ])
            } catch (error) {
                this._queue = this._queue.filter((item) => item.id !== id)
                throw error instanceof Error
                    ? error
                    : new Error('Unknown lock error')
            } finally {
                if (timeoutId) clearTimeout(timeoutId)
            }
        }

        this._lock = true
        this._currentLockId = id
        this._lockCount = 1
        return id
    }

    async runLocked<T>(func: () => Promise<T>): Promise<T> {
        const id = await this.lock()
        const result = await func()
        await this.unlock(id)
        return result
    }

    async unlock(id: number) {
        if (!this._lock) {
            throw new Error('Unlock called without active lock')
        }

        if (this._currentLockId !== id) {
            throw new Error('Invalid unlock attempt: lock not owned by caller')
        }

        this._lockCount = Math.max(0, this._lockCount - 1)

        if (this._lockCount === 0) {
            this._lock = false
            this._currentLockId = undefined

            if (this._queue.length > 0) {
                const nextRequest = this._queue.shift()!
                this._lock = true
                this._currentLockId = nextRequest.id
                this._lockCount = 1
                nextRequest.resolve()
            }
        }
    }

    get isLocked() {
        return this._lock
    }
}
