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
    private readonly _timeout: number

    constructor(timeout = Time.minute * 3) {
        this._timeout = timeout
    }

    async lock() {
        const id = this._currentId++

        if (this._lock) {
            const { promise, resolve } = withResolver()
            this._queue.push({ id, resolve })

            try {
                await Promise.race([
                    promise,
                    // eslint-disable-next-line promise/param-names
                    new Promise((_, reject) =>
                        setTimeout(
                            () => reject(new Error('Lock timeout')),
                            this._timeout
                        )
                    )
                ])
            } catch (error) {
                // Remove from queue on timeout
                this._queue = this._queue.filter((item) => item.id !== id)
                throw error
            }
        }

        this._lock = true
        this._currentLockId = id
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

        this._lock = false
        this._currentLockId = undefined

        if (this._queue.length > 0) {
            const nextRequest = this._queue.shift()!
            this._lock = true
            this._currentLockId = nextRequest.id
            nextRequest.resolve()
        }
    }

    get isLocked() {
        return this._lock
    }
}
