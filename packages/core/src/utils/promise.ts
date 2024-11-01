export interface Resolver<R = void, E = unknown> {
    promise: Promise<R>
    resolve: (res: R) => void
    reject: (err: E) => void
}

// https://github.com/tc39/proposal-promise-with-resolvers

export function withResolver<R = void, E = unknown>(): Resolver<R, E> {
    let resolve: (res: R) => void
    let reject: (err: E) => void
    const promise = new Promise<R>((_resolve, _reject) => {
        resolve = _resolve
        reject = _reject
    })
    return { promise, resolve, reject }
}

export function runAsync(func: () => Promise<void>): void {
    func().then(
        () => {},
        (err) => {
            throw err
        }
    )
}

export function runAsyncTimeout<T>(
    func: Promise<T>,
    timeout: number,
    defaultValue: T | null = null
): Promise<T> {
    const timeoutPromise = new Promise<T>((resolve, reject) => {
        setTimeout(() => {
            if (defaultValue != null) {
                resolve(defaultValue)
            } else {
                reject(new Error('timeout'))
            }
        }, timeout)
    })

    const wrappedPromise = func.catch((error) => {
        console.error(error)
        if (defaultValue != null) {
            return defaultValue
        }
        throw new Error('timeout')
    })

    return Promise.race([wrappedPromise, timeoutPromise])
}
