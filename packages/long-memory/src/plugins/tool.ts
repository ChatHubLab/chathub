import { StructuredTool } from '@langchain/core/tools'
import { Context } from 'koishi'
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'
import { Config } from '../index'
import { CreateToolParams } from 'koishi-plugin-chatluna/llm-core/platform/types'
import { z } from 'zod'
import { EnhancedMemory, MemoryRetrievalLayerType, MemoryType } from '../types'
import { calculateExpirationDate } from '../utils/memory'

export async function apply(
    ctx: Context,
    config: Config,
    plugin: ChatLunaPlugin
) {
    plugin.registerTool('memory_search', {
        selector(history) {
            return true
        },
        alwaysRecreate: true,
        async createTool(params, session) {
            return new MemorySearchTool(ctx, params)
        }
    })

    plugin.registerTool('memory_add', {
        selector(history) {
            return true
        },
        alwaysRecreate: true,
        async createTool(params, session) {
            return new MemoryAddTool(ctx, params)
        }
    })

    plugin.registerTool('memory_delete', {
        selector(history) {
            return true
        },
        alwaysRecreate: true,
        async createTool(params, session) {
            return new MemoryDeleteTool(ctx, params)
        }
    })
}

export class MemorySearchTool extends StructuredTool {
    name = 'memory_search'

    schema = z.object({
        content: z.string().describe('The search content of the memory'),
        layer: z
            .array(
                z.union([
                    z.literal('user'),
                    z.literal('preset_user'),
                    z.literal('preset'),
                    z.literal('global')
                ])
            )
            .describe('The layer of the memory')
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
    })

    constructor(
        private ctx: Context,
        private params: CreateToolParams
    ) {
        super({})
    }

    /** @ignore */
    async _call(input: z.infer<typeof this.schema>) {
        try {
            const result = await this.ctx.chatluna_long_memory.retrieveMemory(
                this.params.conversationId,
                input.content,
                input.layer != null
                    ? input.layer.map(
                          (layer) =>
                              MemoryRetrievalLayerType[layer.toUpperCase()]
                      )
                    : MemoryRetrievalLayerType.PRESET_USER
            )

            return JSON.stringify(result)
        } catch (error) {
            return 'An error occurred while searching for memories.'
        }
    }

    // eslint-disable-next-line max-len
    description = `Searches user-related memories based on keywords or phrases. Usage guidelines:

    - content: Specify search keywords or phrases (e.g., "birthday", "favorite food") to retrieve relevant memories
    - layer: Specify which memory layers to search in as an array. Available layers:
      * preset_user: (Recommended, Default) User-specific memories for the current preset. This is the primary retrieval layer where chat memories are stored by default
      * user: User-specific memories shared across all presets
      * preset: Memories shared by all users using the same preset
      * global: Memories shared across all users and presets

    For best results, prioritize searching in the 'preset_user' layer as it contains the most relevant user-specific memories.`
}

export class MemoryAddTool extends StructuredTool {
    name = 'memory_add'

    schema = z.object({
        memories: z
            .array(
                z.object({
                    content: z.string().describe('The content of the memory'),
                    type: z
                        .nativeEnum(MemoryType)
                        .describe('The type of the memory'),
                    importance: z
                        .number()
                        .min(1)
                        .max(10)
                        .describe('The importance of the memory (1-10)')
                })
            )
            .describe('Array of memories to add'),
        layer: z
            .array(
                z.union([
                    z.literal('user'),
                    z.literal('preset_user'),
                    z.literal('preset'),
                    z.literal('global')
                ])
            )
            .describe('The layer of the memory')
    })

    constructor(
        private ctx: Context,
        private params: CreateToolParams
    ) {
        super({})
    }

    /** @ignore */
    async _call(input: z.infer<typeof this.schema>) {
        try {
            // Convert input memories to EnhancedMemory objects
            const enhancedMemories = input.memories.map((memory) => {
                return {
                    content: memory.content,
                    type: memory.type,
                    importance: memory.importance,
                    expirationDate: calculateExpirationDate(
                        memory.type,
                        memory.importance
                    )
                } as EnhancedMemory
            })

            // Add memories to the specified layers
            await this.ctx.chatluna_long_memory.addMemories(
                this.params.conversationId,
                enhancedMemories,
                input.layer != null
                    ? input.layer.map(
                          (layer) =>
                              MemoryRetrievalLayerType[layer.toUpperCase()]
                      )
                    : MemoryRetrievalLayerType.PRESET_USER
            )

            return `Successfully added ${enhancedMemories.length} memories.`
        } catch (error) {
            return 'An error occurred while adding memories.'
        }
    }

    description = `Adds user-related memories to specified layers. Each memory requires:

    - memories: Array of memory objects with:
      * content: Memory text (e.g., "Likes pizza")
      * type: Memory category - Options include:
        Long-term: factual, preference, personal, skill, interest, habit, relationship
        Medium-term: contextual, task, location
        Short-term: temporal, event
      * importance: Rating 1-10 (higher = longer retention)

    - layer: Target memory layers (array):
      * preset_user: (Default) User memories for current preset
      * user: User memories across all presets
      * preset: Shared memories for all users of this preset
      * global: Shared across all users and presets

    System auto-calculates expiration dates based on type and importance.`
}

export class MemoryDeleteTool extends StructuredTool {
    name = 'memory_delete'

    schema = z.object({
        memoryIds: z
            .array(z.string())
            .describe('Array of memory IDs to delete'),
        layer: z
            .array(
                z.union([
                    z.literal('user'),
                    z.literal('preset_user'),
                    z.literal('preset'),
                    z.literal('global')
                ])
            )
            .describe('The layer of the memory')
    })

    constructor(
        private ctx: Context,
        private params: CreateToolParams
    ) {
        super({})
    }

    /** @ignore */
    async _call(input: z.infer<typeof this.schema>) {
        try {
            // Delete memories from the specified layers
            await this.ctx.chatluna_long_memory.deleteMemories(
                this.params.conversationId,
                input.memoryIds,
                input.layer != null
                    ? input.layer.map(
                          (layer) =>
                              MemoryRetrievalLayerType[layer.toUpperCase()]
                      )
                    : MemoryRetrievalLayerType.PRESET_USER
            )

            return `Successfully deleted ${input.memoryIds.length} memories.`
        } catch (error) {
            return 'An error occurred while deleting memories.'
        }
    }

    // eslint-disable-next-line max-len
    description = `Deletes user-related memories by their IDs. Usage guidelines:

    - memoryIds: Array of memory IDs to delete
    - layer: Specify which memory layers to delete from as an array. Available layers:
      * preset_user: (Recommended, Default) User-specific memories for the current preset
      * user: User-specific memories shared across all presets
      * preset: Memories shared by all users using the same preset
      * global: Memories shared across all users and presets

    Please search for memory IDs using the 'memory_search' tool before deleting.

    Returns the number of successfully deleted memories.`
}
