import fs from 'fs/promises'
import { watch } from 'fs'
import { Context, Logger, Schema } from 'koishi'
import {
    loadPreset,
    PresetTemplate
} from 'koishi-plugin-chatluna/llm-core/prompt'
import {
    ChatLunaError,
    ChatLunaErrorCode
} from 'koishi-plugin-chatluna/utils/error'
import { createLogger } from 'koishi-plugin-chatluna/utils/logger'
import path from 'path'
import { fileURLToPath } from 'url'
import { Cache } from './cache'
import { Config } from './config'
import md5 from 'md5'

let logger: Logger

export class PresetService {
    private readonly _presets: PresetTemplate[] = []

    private _aborter: AbortController

    constructor(
        private readonly ctx: Context,
        private readonly config: Config,
        private readonly cache: Cache<'chathub/keys', string>
    ) {
        logger = createLogger(ctx)

        ctx.on('dispose', () => {
            this._aborter?.abort()
        })
    }

    async loadPreset(file: string) {
        const rawText = await fs.readFile(file, 'utf-8')
        try {
            const preset = loadPreset(rawText)

            preset.path = file
            this._presets.push(preset)
        } catch (e) {
            logger.error(`error when load preset ${file}`, e)
        }
    }

    async loadAllPreset() {
        await this._checkPresetDir()

        const presetDir = this.resolvePresetDir()
        const files = await fs.readdir(presetDir)

        this._presets.length = 0

        for (const file of files) {
            // use file
            const extension = path.extname(file)
            if (extension !== '.txt' && extension !== '.yml') {
                continue
            }
            await this.loadPreset(path.join(presetDir, file))
        }

        this._updateSchema()
    }

    watchPreset() {
        let fsWait: NodeJS.Timeout | boolean = false
        const md5Cache = new Map<string, string>()

        if (this._aborter != null) {
            this._aborter.abort()
        }

        this._aborter = new AbortController()

        watch(
            this.resolvePresetDir(),
            {
                signal: this._aborter.signal
            },
            async (event, filename) => {
                if (!filename) {
                    await this.loadAllPreset()
                    logger.debug(`trigger full reload preset`)
                    return
                }

                if (fsWait) return
                fsWait = setTimeout(() => {
                    fsWait = false
                }, 100)

                const filePath = path.join(this.resolvePresetDir(), filename)

                try {
                    const fileStat = await fs.stat(filePath)
                    if (fileStat.isDirectory()) return

                    // Handle file deletion
                    if (event === 'rename' && !fileStat) {
                        const index = this._presets.findIndex(
                            (p) => p.path === filePath
                        )
                        if (index !== -1) {
                            this._presets.splice(index, 1)
                            md5Cache.delete(filePath)
                            logger.debug(`removed preset: ${filename}`)
                            return
                        }
                    }

                    // Check if file content changed
                    const md5Current = md5(await fs.readFile(filePath))
                    if (md5Current === md5Cache.get(filePath)) return

                    md5Cache.set(filePath, md5Current)

                    // Update or add the preset
                    const index = this._presets.findIndex(
                        (p) => p.path === filePath
                    )
                    if (index !== -1) {
                        // Update existing preset
                        const preset = loadPreset(
                            await fs.readFile(filePath, 'utf-8')
                        )
                        preset.path = filePath
                        this._presets[index] = preset
                        logger.debug(`updated preset: ${filename}`)
                    } else {
                        // Add new preset
                        await this.loadPreset(filePath)
                        logger.debug(`added new preset: ${filename}`)
                    }

                    // Update schema after changes
                    this._updateSchema()
                } catch (e) {
                    logger.error(
                        `error when watching preset file ${filePath}`,
                        e
                    )

                    // trigger full reload
                    await this.loadAllPreset()
                }
            }
        )
    }

    async init() {
        await this.loadAllPreset()
        this.watchPreset()
    }

    async getPreset(
        triggerKeyword: string,
        loadForDisk: boolean = false,
        throwError: boolean = true
    ): Promise<PresetTemplate> {
        if (loadForDisk) {
            // always load for disk
            await this.loadAllPreset()
        }

        const preset = this._presets.find((preset) =>
            preset.triggerKeyword.includes(triggerKeyword)
        )

        if (preset) {
            return preset
        }

        if (throwError) {
            throw new ChatLunaError(
                ChatLunaErrorCode.PREST_NOT_FOUND,
                new Error(`No preset found for keyword ${triggerKeyword}`)
            )
        }

        return undefined
    }

    async getDefaultPreset(): Promise<PresetTemplate> {
        if (this._presets.length === 0) {
            await this.loadAllPreset()
        }

        const preset = this._presets.find((preset) =>
            preset.triggerKeyword.includes('chatgpt')
        )

        if (preset) {
            // await this.cache.set('default-preset', 'chatgpt')
            return preset
        } else {
            await this._copyDefaultPresets()
            return this.getDefaultPreset()
        }

        // throw new Error("No default preset found")
    }

    async getAllPreset(concatKeyword: boolean = true): Promise<string[]> {
        await this.loadAllPreset()

        return this._presets.map((preset) =>
            concatKeyword
                ? preset.triggerKeyword.join(', ')
                : preset.triggerKeyword[0]
        )
    }

    async addPreset(preset: PresetTemplate): Promise<void> {
        this._presets.push(preset)

        this._updateSchema()
    }

    private _updateSchema() {
        if (!this.ctx.scope.isActive) {
            return
        }

        this.ctx.schema.set(
            'preset',
            Schema.union(
                this._presets.map((preset) =>
                    Schema.const(preset.triggerKeyword[0])
                )
            )
        )
    }

    async resetDefaultPreset(): Promise<void> {
        await this.cache.delete('default-preset')

        await this._copyDefaultPresets()
    }

    public resolvePresetDir() {
        return path.resolve(this.ctx.baseDir, 'data/chathub/presets')
    }

    private async _checkPresetDir() {
        const presetDir = path.join(this.resolvePresetDir())

        // check if preset dir exists
        try {
            await fs.access(presetDir)
        } catch (err) {
            if (err.code === 'ENOENT') {
                await fs.mkdir(presetDir, { recursive: true })
                await this._copyDefaultPresets()
            } else {
                throw err
            }
        }
    }

    private async _copyDefaultPresets() {
        const currentPresetDir = path.join(this.resolvePresetDir())

        const dirname =
            __dirname?.length > 0 ? __dirname : fileURLToPath(import.meta.url)

        const defaultPresetDir = path.join(dirname, '../resources/presets')

        const files = await fs.readdir(defaultPresetDir)

        for (const file of files) {
            const filePath = path.join(defaultPresetDir, file)
            const fileStat = await fs.stat(filePath)
            if (fileStat.isFile()) {
                await fs.mkdir(currentPresetDir, { recursive: true })
                logger.debug(
                    `copy preset file ${filePath} to ${currentPresetDir}`
                )
                await fs.copyFile(filePath, path.join(currentPresetDir, file))
            }
        }
    }
}
