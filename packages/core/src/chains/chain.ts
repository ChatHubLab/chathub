import { EventEmitter } from 'events'
import { Context, h, Logger, Session } from 'koishi'
import {
    ChatLunaError,
    ChatLunaErrorCode,
    setErrorFormatTemplate
} from 'koishi-plugin-chatluna/utils/error'
import { createLogger } from 'koishi-plugin-chatluna/utils/logger'
import { Config } from '../config'
import { lifecycleNames } from '../middlewares/lifecycle'

let logger: Logger

/**
 * ChatChain为消息的发送和接收提供了一个统一的中间提供交互
 */
export class ChatChain {
    public readonly _graph: ChatChainDependencyGraph
    private readonly _senders: ChatChainSender[]
    private isSetErrorMessage = false

    constructor(
        private readonly ctx: Context,
        private readonly config: Config
    ) {
        logger = createLogger(ctx)
        this._graph = new ChatChainDependencyGraph()
        this._senders = []

        const defaultChatChainSender = new DefaultChatChainSender(config)

        this._senders.push((session, messages) =>
            defaultChatChainSender.send(session, messages)
        )
    }

    async receiveMessage(session: Session, ctx?: Context) {
        const context: ChainMiddlewareContext = {
            config: this.config,
            message: session.content,
            ctx: ctx ?? this.ctx,
            session,
            options: {},
            send: (message) => this.sendMessage(session, message),
            recallThinkingMessage: async () => {}
        }

        context.recallThinkingMessage = async () => {
            if (context.options?.thinkingTimeoutObject) {
                clearTimeout(context.options.thinkingTimeoutObject.timeout!)

                if (context.options.thinkingTimeoutObject.autoRecallTimeout) {
                    clearTimeout(
                        context.options.thinkingTimeoutObject.autoRecallTimeout!
                    )
                }

                if (context.options.thinkingTimeoutObject.recallFunc) {
                    await context.options.thinkingTimeoutObject.recallFunc()
                }
                if (context.options?.thinkingTimeoutObject?.timeout) {
                    context.options.thinkingTimeoutObject.timeout = null
                }
                context.options.thinkingTimeoutObject = undefined
            }
        }

        const result = await this._runMiddleware(session, context)

        await context.recallThinkingMessage()

        return result
    }

    async receiveCommand(
        session: Session,
        command: string,
        options: ChainMiddlewareContextOptions = {}
    ) {
        const context: ChainMiddlewareContext = {
            config: this.config,
            message: options?.message ?? session.content,
            ctx: this.ctx,
            session,
            command,
            send: (message) => this.sendMessage(session, message),
            recallThinkingMessage: async () => {},
            options
        }

        context.recallThinkingMessage = async () => {
            if (context.options.thinkingTimeoutObject) {
                clearTimeout(context.options.thinkingTimeoutObject.timeout!)
                if (context.options.thinkingTimeoutObject.recallFunc) {
                    await context.options.thinkingTimeoutObject.recallFunc()
                }
                if (context.options?.thinkingTimeoutObject?.timeout) {
                    context.options.thinkingTimeoutObject.timeout = null
                }
                context.options.thinkingTimeoutObject = undefined
            }
        }

        const result = await this._runMiddleware(session, context)

        await context.recallThinkingMessage()

        return result
    }

    middleware<T extends keyof ChainMiddlewareName>(
        name: T,
        middleware: ChainMiddlewareFunction,
        ctx: Context = this.ctx
    ): ChainMiddleware {
        const result = new ChainMiddleware(name, middleware, this._graph)

        this._graph.addNode(result)

        ctx.on('dispose', () => {
            this._graph.removeNode(name)
        })

        return result
    }

    sender(sender: ChatChainSender) {
        this._senders.push(sender)
    }

    private async _runMiddleware(
        session: Session,
        context: ChainMiddlewareContext
    ) {
        // 手动 polyfill，呃呃呃呃呃
        if (session.isDirect == null) {
            session.isDirect = session.subtype === 'private'
        }

        if (!this.isSetErrorMessage) {
            setErrorFormatTemplate(session.text('chatluna.error_message'))
            this.isSetErrorMessage = true
        }

        const originMessage = context.message

        const runList = this._graph.build()

        if (runList.length === 0) {
            return false
        }

        let isOutputLog = false

        for (const middleware of runList) {
            let result: ChainMiddlewareRunStatus | h[] | h | h[][] | string

            let executedTime = Date.now()

            try {
                result = await middleware.run(session, context)

                executedTime = Date.now() - executedTime
            } catch (error) {
                if (error instanceof ChatLunaError) {
                    if (error.errorCode === ChatLunaErrorCode.ABORTED) {
                        await this.sendMessage(
                            session,
                            session.text('chatluna.aborted')
                        )
                        return false
                    }

                    await this.sendMessage(session, error.message)
                    return false
                }

                logger.error(`chat-chain: ${middleware.name} error ${error}`)

                logger.error(error)

                if (error.cause) {
                    logger.error(error.cause)
                }
                logger.debug('-'.repeat(20) + '\n')

                await this.sendMessage(
                    session,
                    session.text('chatluna.middleware_error', [
                        middleware.name,
                        error.message
                    ])
                )

                return false
            }

            if (
                !middleware.name.startsWith('lifecycle-') &&
                ChainMiddlewareRunStatus.SKIPPED !== result &&
                middleware.name !== 'allow_reply' &&
                executedTime > 10
            ) {
                logger.debug(
                    `middleware %c executed in %d ms`,
                    middleware.name,
                    executedTime
                )
                isOutputLog = true
            }

            if (result === ChainMiddlewareRunStatus.STOP) {
                // 中间件说这里不要继续执行了
                if (
                    context.message != null &&
                    context.message !== originMessage
                ) {
                    // 消息被修改了
                    await this.sendMessage(session, context.message)
                }

                if (isOutputLog) {
                    logger.debug('-'.repeat(20) + '\n')
                }

                return false
            } else if (result instanceof Array || typeof result === 'string') {
                context.message = result
            }
        }

        if (isOutputLog) {
            logger.debug('-'.repeat(20) + '\n')
        }

        if (context.message != null && context.message !== originMessage) {
            // 消息被修改了
            await this.sendMessage(session, context.message)
        }

        return true
    }

    private async sendMessage(
        session: Session,
        message: h[] | h[][] | h | string
    ) {
        // check if message is a two-dimensional array

        const messages: (h[] | h | string)[] =
            message instanceof Array ? message : [message]

        for (const sender of this._senders) {
            await sender(session, messages)
        }
    }
}

class ChatChainDependencyGraph {
    private _tasks = new Map<string, ChainDependencyGraphNode>()
    private _dependencies = new Map<string, Set<string>>()
    private _eventEmitter = new EventEmitter()
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    private _listeners = new Map<string, Set<(...args: any[]) => void>>()
    private _cachedOrder: ChainMiddleware[] | null = null

    constructor() {
        this._eventEmitter.on('build_node', () => {
            for (const [name, listeners] of this._listeners) {
                for (const listener of listeners) {
                    listener(name)
                }
                listeners.clear()
            }
            // Invalidate cache when nodes change
            this._cachedOrder = null
        })
    }

    // Add a task to the DAG.
    public addNode(middleware: ChainMiddleware): void {
        this._tasks.set(middleware.name, {
            name: middleware.name,
            middleware
        })
        this._cachedOrder = null // Invalidate cache
    }

    removeNode(name: string): void {
        this._tasks.delete(name)

        // Efficiently remove dependencies
        this._dependencies.delete(name)
        for (const deps of this._dependencies.values()) {
            deps.delete(name)
        }

        this._cachedOrder = null // Invalidate cache
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    once(name: string, listener: (...args: any[]) => void) {
        const listeners = this._listeners.get(name) ?? new Set()
        listeners.add(listener)
        this._listeners.set(name, listeners)
    }

    // Set a dependency between two tasks
    before(
        taskA: ChainMiddleware | string,
        taskB: ChainMiddleware | string
    ): void {
        if (taskA instanceof ChainMiddleware) {
            taskA = taskA.name
        }
        if (taskB instanceof ChainMiddleware) {
            taskB = taskB.name
        }
        if (taskA && taskB) {
            // Add taskB to the dependencies of taskA
            const dependencies = this._dependencies.get(taskA) ?? new Set()
            dependencies.add(taskB)
            this._dependencies.set(taskA, dependencies)
        } else {
            throw new Error('Invalid tasks')
        }
    }

    // Set a reverse dependency between two tasks
    after(
        taskA: ChainMiddleware | string,
        taskB: ChainMiddleware | string
    ): void {
        if (taskA instanceof ChainMiddleware) {
            taskA = taskA.name
        }
        if (taskB instanceof ChainMiddleware) {
            taskB = taskB.name
        }
        if (taskA && taskB) {
            // Add taskB to the dependencies of taskA
            const dependencies = this._dependencies.get(taskB) ?? new Set()
            dependencies.add(taskA)
            this._dependencies.set(taskB, dependencies)
        } else {
            throw new Error('Invalid tasks')
        }
    }

    // Get dependencies of a task
    getDependencies(task: string) {
        return this._dependencies.get(task)
    }

    // Get dependents of a task
    getDependents(task: string): string[] {
        const dependents: string[] = []
        for (const [key, value] of this._dependencies.entries()) {
            if ([...value].includes(task)) {
                dependents.push(key)
            }
        }
        return dependents
    }

    // Build a two-dimensional array of tasks based on their dependencies
    build(): ChainMiddleware[] {
        // Return cached order if available
        if (this._cachedOrder) {
            return this._cachedOrder
        }

        this._eventEmitter.emit('build_node')
        // Create in-degree table and temporary graph
        const indegree = new Map<string, number>()
        const tempGraph = new Map<string, Set<string>>()

        // Initialize in-degree and temporary graph
        for (const taskName of this._tasks.keys()) {
            indegree.set(taskName, 0)
            tempGraph.set(taskName, new Set())
        }

        // Build temporary graph and calculate in-degree
        for (const [from, deps] of this._dependencies.entries()) {
            const depsSet = tempGraph.get(from) || new Set()
            for (const to of deps) {
                depsSet.add(to)
                indegree.set(to, (indegree.get(to) || 0) + 1)
            }
            tempGraph.set(from, depsSet)
        }

        const queue: string[] = []
        const result: ChainMiddleware[] = []
        const visited = new Set<string>()

        // Find nodes with in-degree of 0
        for (const [task, degree] of indegree.entries()) {
            if (degree === 0) {
                queue.push(task)
            }
        }

        // Topological sorting
        while (queue.length > 0) {
            const current = queue.shift()!

            if (visited.has(current)) {
                continue
            }
            visited.add(current)

            const node = this._tasks.get(current)
            if (node?.middleware) {
                result.push(node.middleware)
            }

            // Process all successors of the current node
            const successors = tempGraph.get(current) || new Set()
            for (const next of successors) {
                const newDegree = indegree.get(next)! - 1
                indegree.set(next, newDegree)

                if (newDegree === 0) {
                    queue.push(next)
                }
            }
        }

        // Check for circular dependencies
        for (const [node, degree] of indegree.entries()) {
            if (degree > 0) {
                throw new Error(
                    `Circular dependency detected involving node: ${node}`
                )
            }
        }

        // Check if all nodes have been visited
        if (visited.size !== this._tasks.size) {
            throw new Error(
                'Some nodes are unreachable in the dependency graph'
            )
        }

        this._cachedOrder = result
        return result
    }
}

interface ChainDependencyGraphNode {
    middleware?: ChainMiddleware
    name: string
}

export class ChainMiddleware {
    constructor(
        readonly name: string,
        private readonly execute: ChainMiddlewareFunction,
        private readonly graph: ChatChainDependencyGraph
    ) {}

    before<T extends keyof ChainMiddlewareName>(name: T) {
        this.graph.before(this.name, name)

        if (this.name.startsWith('lifecycle-')) {
            return this
        }

        const lifecycleName = lifecycleNames

        // 现在我们需要基于当前添加的依赖，去寻找这个依赖锚定的生命周期

        // 如果当前添加的依赖是生命周期，那么我们需要找到这个生命周期的下一个生命周期
        if (lifecycleName.includes(name)) {
            const lastLifecycleName =
                lifecycleName[lifecycleName.indexOf(name) - 1]

            if (lastLifecycleName) {
                this.graph.after(this.name, lastLifecycleName)
            }

            return this
        }

        // 如果不是的话，我们就需要寻找依赖锚定的生命周期

        this.graph.once('build_node', () => {
            const beforeMiddlewares = [
                ...this.graph.getDependencies(name)
            ].filter((name) => name.startsWith('lifecycle-'))

            const afterMiddlewares = this.graph
                .getDependents(name)
                .filter((name) => name.startsWith('lifecycle-'))

            for (const before of beforeMiddlewares) {
                this.graph.before(this.name, before)
            }

            for (const after of afterMiddlewares) {
                this.graph.after(this.name, after)
            }
        })

        return this
    }

    after<T extends keyof ChainMiddlewareName>(name: T) {
        this.graph.after(this.name, name)

        if (this.name.startsWith('lifecycle-')) {
            return this
        }

        const lifecycleName = lifecycleNames

        // 现在我们需要基于当前添加的依赖，去寻找这个依赖锚定的生命周期

        // 如果当前添加的依赖是生命周期，那么我们需要找到这个生命周期的下一个生命周期
        if (lifecycleName.includes(name)) {
            const nextLifecycleName =
                lifecycleName[lifecycleName.indexOf(name) + 1]

            if (nextLifecycleName) {
                this.graph.before(this.name, nextLifecycleName)
            }

            return this
        }

        // 如果不是的话，我们就需要寻找依赖锚定的生命周期
        this.graph.once('build_node', () => {
            const beforeMiddlewares = [
                ...this.graph.getDependencies(name)
            ].filter((name) => name.startsWith('lifecycle-'))

            const afterMiddlewares = this.graph
                .getDependents(name)
                .filter((name) => name.startsWith('lifecycle-'))

            for (const before of beforeMiddlewares) {
                this.graph.before(this.name, before)
            }

            for (const after of afterMiddlewares) {
                this.graph.after(this.name, after)
            }
        })

        return this
    }

    run(session: Session, options: ChainMiddlewareContext) {
        return this.execute(session, options)
    }
}

class DefaultChatChainSender {
    constructor(private readonly config: Config) {}

    private processElements(elements: h[]): h[] {
        return elements
            .filter((element): element is h => {
                if (!element) return false

                if (element.type === 'img') {
                    const src = element.attrs?.['src']
                    return !(
                        typeof src === 'string' && src.startsWith('attachment')
                    )
                }
                return true
            })
            .map((element) => {
                if (element.children?.length) {
                    element.children = this.processElements(element.children)
                }
                return element
            })
    }

    async send(
        session: Session,
        messages: (h[] | h | string)[]
    ): Promise<void> {
        if (!messages?.length) return

        if (this.config.isForwardMsg) {
            await this.sendAsForward(session, messages)
            return
        }

        await this.sendAsNormal(session, messages)
    }

    private async sendAsForward(
        session: Session,
        messages: (h[] | h | string)[]
    ): Promise<void> {
        const sendMessages = this.convertToForwardMessages(messages)

        await session.sendQueued(
            h('message', { forward: true }, ...sendMessages)
        )
    }

    private convertToForwardMessages(messages: (h[] | h | string)[]): h[] {
        const firstMsg = messages[0]

        if (Array.isArray(firstMsg)) {
            // h[][]
            return messages.map((msg) => h('message', ...(msg as h[])))
        }

        if (typeof firstMsg === 'object') {
            // h | h[]
            return [h('message', ...(messages as h[]))]
        }

        if (typeof firstMsg === 'string') {
            // string
            return [h.text(firstMsg)]
        }

        throw new Error(`Unsupported message type: ${typeof firstMsg}`)
    }

    private async sendAsNormal(
        session: Session,
        messages: (h[] | h | string)[]
    ): Promise<void> {
        for (const message of messages) {
            const messageFragment = await this.buildMessageFragment(
                session,
                message
            )

            if (!messageFragment?.length) continue

            const processedFragment = this.processElements(messageFragment)
            await session.sendQueued(processedFragment)
        }
    }

    private async buildMessageFragment(
        session: Session,
        message: h[] | h | string
    ): Promise<h[]> {
        const shouldAddQuote =
            this.config.isReplyWithAt &&
            session.isDirect === false &&
            session.messageId

        const messageContent = this.convertMessageToArray(message)

        if (!shouldAddQuote) {
            return messageContent
        }

        // Check if quote should be removed (for audio or message types)
        const quote = h('quote', { id: session.messageId })
        const hasIncompatibleType = messageContent.some(
            (element) => element.type === 'audio' || element.type === 'message'
        )

        return hasIncompatibleType ? messageContent : [quote, ...messageContent]
    }

    private convertMessageToArray(message: h[] | h | string): h[] {
        if (Array.isArray(message)) {
            return message
        }
        if (typeof message === 'string') {
            return [h.text(message)]
        }
        return [message]
    }
}

export interface ChainMiddlewareContext {
    config: Config
    ctx: Context
    session: Session
    message: string | h[] | h[][]
    options?: ChainMiddlewareContextOptions
    command?: string
    recallThinkingMessage?: () => Promise<void>
    send: (message: h[][] | h[] | h | string) => Promise<void>
}

export interface ChainMiddlewareContextOptions {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    [key: string]: any
}

export interface ChainMiddlewareName {}

export type ChainMiddlewareFunction = (
    session: Session,
    context: ChainMiddlewareContext
) => Promise<string | h[] | h[][] | ChainMiddlewareRunStatus | null>

export type ChatChainSender = (
    session: Session,
    message: (h[] | h | string)[]
) => Promise<void>

export enum ChainMiddlewareRunStatus {
    SKIPPED = 0,
    STOP = 1,
    CONTINUE = 2
}
