import { sleep } from 'koishi'
import { ObjectLock } from 'koishi-plugin-chatluna/utils/lock'

export class BufferText {
    private queue: string[] = []
    private currentText = ''

    private lock = new ObjectLock()
    private isEnd = false
    private isTextStarted = false
    private rawText = ''

    constructor(
        private readonly isStreaming: boolean,
        private readonly sleepTime = 3,
        private readonly prefix?: string,
        private readonly postfix?: string
    ) {}

    async addText(text: string) {
        if (this.isEnd) {
            return
        }

        const unlock = await this.lock.lock()

        if (this.isStreaming) {
            this.queue.push(...text.split(''))
        } else {
            const diffText = text.substring(
                Math.min(text.length, this.currentText.length)
            )

            this.queue.push(...diffText.split(''))
        }

        this.currentText = text

        unlock()
    }

    private async *getText() {
        while (this.queue.length > 0 || !this.isEnd) {
            const text = await this.processChar()

            if (text == null) {
                await sleep(this.sleepTime)
                continue
            }

            this.rawText += text

            yield text
        }
    }

    private async processChar() {
        if (this.queue.length < 1) {
            return undefined
        }

        const unlock = await this.lock.lock()

        const text = this.queue.shift()

        unlock()

        if (!this.isEnd) {
            await sleep(this.sleepTime)
        }

        return text
    }

    async *get() {
        let bufferText = ''
        let inContent = this.prefix == null

        for await (const char of this.getText()) {
            bufferText += char

            if (!inContent) {
                // Looking for prefix
                if (this.prefix && bufferText.endsWith(this.prefix)) {
                    inContent = true
                    bufferText = ''
                }
            } else {
                // In content, looking for postfix
                if (this.postfix && bufferText.endsWith(this.postfix)) {
                    // Found postfix, yield content without postfix
                    yield* bufferText.slice(0, -this.postfix.length)
                    this.isEnd = true
                    break
                } else if (
                    this.postfix == null ||
                    bufferText.length > this.postfix.length
                ) {
                    // No postfix or buffer exceeds postfix length, yield safely
                    yield bufferText[0]
                    bufferText = bufferText.slice(1)
                }
            }
        }

        // If no postfix or reached end without finding postfix, yield remaining content
        if (inContent && !this.isEnd) {
            yield* bufferText
        }
    }

    async *splitByMarkdown() {
        let bufferText = ''

        for await (const char of this.get()) {
            bufferText += char
            const lastTwoChars = bufferText.slice(-2)

            if (lastTwoChars === '\n\n') {
                yield bufferText.slice(0, -2)
                bufferText = ''
            }
        }

        if (bufferText.length > 0) {
            yield bufferText
        }
    }

    async *splitByPunctuations() {
        const punctuations = ['，', '.', '。', '!', '！', '?', '？']
        const sendTogglePunctuations = ['.', '!', '！', '?', '？']

        let bufferText = ''

        for await (const char of this.get()) {
            const inPunctuation = punctuations.includes(char)
            const includeSendPunctuation = sendTogglePunctuations.includes(char)

            if (includeSendPunctuation) {
                bufferText += char
            }

            if (inPunctuation) {
                yield bufferText
                bufferText = ''
                continue
            }

            bufferText += char
        }

        if (bufferText.length > 0) {
            yield bufferText
        }
    }

    async *getCached(endText: string = '●') {
        let bufferText = ''

        for await (const char of this.get()) {
            bufferText += char

            yield bufferText + endText
        }

        yield bufferText
    }

    end() {
        this.isEnd = true
    }
}
