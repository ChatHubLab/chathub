import { LunaVDB as LunaDB } from '@chatluna/luna-vdb'
import type { EmbeddingsInterface } from '@langchain/core/embeddings'
import { SaveableVectorStore } from '@langchain/core/vectorstores'
import { Document } from '@langchain/core/documents'
import { SynchronousInMemoryDocstore } from '@langchain/community/stores/doc/in_memory'
import crypto from 'crypto'
import { Context, Logger } from 'koishi'
import { ChatLunaSaveableVectorStore } from 'koishi-plugin-chatluna/llm-core/model/base'
import path from 'path'
import fs from 'fs/promises'
import { createLogger } from 'koishi-plugin-chatluna/utils/logger'
import { ChatLunaPlugin } from 'koishi-plugin-chatluna/services/chat'
import { Config } from '..'

let logger: Logger

export async function apply(
    ctx: Context,
    config: Config,
    plugin: ChatLunaPlugin
) {
    logger = createLogger(ctx, 'chatluna-vector-store-service')

    if (!config.vectorStore.includes('luna-vdb')) {
        return
    }

    plugin.registerVectorStore('luna-vdb', async (params) => {
        const embeddings = params.embeddings
        let lunaDBStore: LunaDBVectorStore

        const directory = path.join(
            'data/chathub/vector_store/luna_vdb',
            params.key ?? 'chatluna'
        )

        try {
            await fs.access(directory)
        } catch {
            await fs.mkdir(directory, { recursive: true })
        }

        const jsonFile = path.join(directory, 'docstore.index')

        logger.debug(`Loading luna vdb store from %c`, directory)

        try {
            await fs.access(jsonFile)
            lunaDBStore = await LunaDBVectorStore.load(directory, embeddings)

            // test the embeddings dimension
            const testEmbedding = await embeddings.embedQuery('test')
            if (testEmbedding.length !== lunaDBStore.numDimensions) {
                logger.error(
                    `embeddings dimension mismatch: ${testEmbedding.length} !== ${lunaDBStore.numDimensions}. Please check the embeddings.`
                )
                // voyStore = undefined
            }
        } catch (e) {
            lunaDBStore = new LunaDBVectorStore(new LunaDB(), embeddings)

            try {
                await lunaDBStore.save(directory)
            } catch (e) {
                logger.error(e)
            }
        }

        if (lunaDBStore == null) {
            throw new Error('failed to load voy store')
        }

        const wrapperStore = new ChatLunaSaveableVectorStore<LunaDBVectorStore>(
            lunaDBStore,
            {
                async saveableFunction(store) {
                    await store.save(directory)
                },
                async deletableFunction(store, options) {
                    if (options.deleteAll) {
                        await fs.rm(directory, { recursive: true })
                        await lunaDBStore.delete({ deleteAll: true })
                        return
                    }

                    const ids: string[] = []
                    if (options.ids) {
                        ids.push(...options.ids)
                    }

                    if (options.documents) {
                        const ids = options.documents
                            ?.map((document) => {
                                return document.metadata?.raw_id as
                                    | string
                                    | undefined
                            })
                            .filter((id) => id != null)

                        ids.push(...ids)
                    }

                    if (ids.length > 0) {
                        await store.delete({ ids })
                    }
                },
                async addDocumentsFunction(store, documents, options) {
                    let ids = options?.ids ?? []

                    ids = documents.map((document, i) => {
                        const id = ids[i] ?? crypto.randomUUID()

                        document.metadata = {
                            ...document.metadata,
                            raw_id: id
                        }

                        return id
                    })

                    await store.addDocuments(documents, {
                        ids
                    })
                },
                async freeFunction() {
                    lunaDBStore.client.free()
                    lunaDBStore = undefined
                }
            }
        )

        return wrapperStore
    })
}

export interface LunaVDBLibArgs {
    docstore?: SynchronousInMemoryDocstore
}

/**
 * Class that extends `VectorStore`. It allows to perform similarity search using
 * Voi similarity search engine. The class requires passing Voy Client as an input parameter.
 */
export class LunaDBVectorStore extends SaveableVectorStore {
    client: LunaDB

    numDimensions: number | null = null

    docstore: SynchronousInMemoryDocstore

    getDocstore(): SynchronousInMemoryDocstore {
        return this.docstore
    }

    _vectorstoreType(): string {
        return 'luna-db'
    }

    constructor(
        client: LunaDB,
        embeddings: EmbeddingsInterface,
        args?: LunaVDBLibArgs
    ) {
        super(embeddings, {})
        this.client = client
        this.embeddings = embeddings
        this.docstore = args?.docstore ?? new SynchronousInMemoryDocstore()
    }

    /**
     * Adds documents to the Voy database. The documents are embedded using embeddings provided while instantiating the class.
     * @param documents An array of `Document` instances associated with the vectors.
     */
    async addDocuments(
        documents: Document[],
        options?: { ids?: string[] }
    ): Promise<void> {
        const texts = documents.map(({ pageContent }) => pageContent)
        if (documents.length === 0) {
            return
        }

        const firstVector = (
            await this.embeddings.embedDocuments(texts.slice(0, 1))
        )[0]
        if (this.numDimensions === null) {
            this.numDimensions = firstVector.length
        } else if (this.numDimensions !== firstVector.length) {
            throw new Error(
                `Vectors must have the same length as the number of dimensions (${this.numDimensions})`
            )
        }
        const restResults = await this.embeddings.embedDocuments(texts.slice(1))
        await this.addVectors([firstVector, ...restResults], documents, options)
    }

    /**
     * Adds vectors to the Voy database. The vectors are associated with
     * the provided documents.
     * @param vectors An array of vectors to be added to the database.
     * @param documents An array of `Document` instances associated with the vectors.
     */
    async addVectors(
        vectors: number[][],
        documents: Document[],
        options?: { ids?: string[] }
    ): Promise<string[]> {
        if (vectors.length === 0) {
            return
        }
        if (this.numDimensions === null) {
            this.numDimensions = vectors[0].length
        }

        if (vectors.length !== documents.length) {
            throw new Error(`Vectors and metadata must have the same length`)
        }
        if (!vectors.every((v) => v.length === this.numDimensions)) {
            throw new Error(
                `Vectors must have the same length as the number of dimensions (${this.numDimensions})`
            )
        }

        const documentIds =
            options?.ids ?? documents.map(() => crypto.randomUUID())

        const embeddings = documentIds.map((documentId, idx) => {
            const resource = {
                id: documentId,
                embeddings: vectors[idx]
            }
            this.docstore.add({ [documentId]: documents[idx] })
            return resource
        })
        this.client.index({ embeddings })
        return documentIds
    }

    /**
     * Searches for vectors in the Voy database that are similar to the
     * provided query vector.
     * @param query The query vector.
     * @param k The number of similar vectors to return.
     * @returns A promise that resolves with an array of tuples, each containing a `Document` instance and a similarity score.
     */
    async similaritySearchVectorWithScore(query: number[], k: number) {
        if (this.numDimensions === null) {
            throw new Error("There aren't any elements in the index yet.")
        }
        if (query.length !== this.numDimensions) {
            throw new Error(
                `Query vector must have the same length as the number of dimensions (${this.numDimensions})`
            )
        }
        const docStoreSize = this.docstore._docs.size
        const itemsToQuery = Math.min(docStoreSize, k)
        if (itemsToQuery > docStoreSize) {
            console.warn(
                `k (${k}) is greater than the number of elements in the index (${docStoreSize}), setting k to ${itemsToQuery}`
            )
        }
        const results = this.client.search(
            new Float32Array(query),
            itemsToQuery
        )
        return results.neighbors.map(({ id, distance }) => {
            return [this.docstore.search(id), distance] as [Document, number]
        })
    }

    /**
     * Method to delete data from the Voy index. It can delete data based
     * on specific IDs or a filter.
     * @param params Object that includes either an array of IDs or a filter for the data to be deleted.
     * @returns Promise that resolves when the deletion is complete.
     */
    async delete(params: {
        deleteAll?: boolean
        ids?: string[]
    }): Promise<void> {
        if (params.deleteAll === true) {
            this.client.clear()
            this.docstore._docs.clear()
            return
        }

        const documentIds = params.ids

        if (documentIds == null || documentIds.length < 1) {
            throw new Error('No documentIds provided to delete.')
        }

        const mappingIds = Array.from(this.docstore._docs.keys())

        const missingIds = documentIds.filter((id) => !mappingIds.includes(id))

        if (missingIds.length > 0) {
            throw new Error(
                `Some specified documentIds do not exist in the current store. DocumentIds not found: ${Array.from(
                    missingIds
                ).join(', ')}`
            )
        }

        const embeddings = documentIds.map((id) => {
            this.docstore._docs.delete(id)
            return id
        })

        this.client.remove(embeddings)
    }

    /**
     * Loads a VoyStore from a specified directory.
     * @param directory The directory to load the VoyStore from.
     * @param embeddings An Embeddings object.
     * @returns A Promise that resolves with a new VoyStore instance.
     */
    static async load(directory: string, embeddings: EmbeddingsInterface) {
        const fs = await import('node:fs/promises')
        const path = await import('node:path')
        const readStore = (directory: string) =>
            fs
                .readFile(path.join(directory, 'docstore.index'), 'utf-8')
                .then(JSON.parse) as Promise<Map<string, Document>>

        const readIndex = async (directory: string) => {
            const data = await fs.readFile(
                path.join(directory, 'luna_db.index')
            )

            return LunaDB.deserialize(data)
        }
        const [docstoreFiles, index] = await Promise.all([
            readStore(directory),
            readIndex(directory)
        ])
        const docstore = new SynchronousInMemoryDocstore(docstoreFiles)
        return new this(index, embeddings, { docstore })
    }

    /**
     * Saves the current state of the VoyStore to a specified directory.
     * @param directory The directory to save the state to.
     * @returns A Promise that resolves when the state has been saved.
     */
    async save(directory: string) {
        const fs = await import('node:fs/promises')
        const path = await import('node:path')
        await fs.mkdir(directory, { recursive: true })
        await Promise.all([
            await fs.writeFile(
                path.join(directory, 'luna_db.index'),
                this.client.serialize()
            ),
            await fs.writeFile(
                path.join(directory, 'docstore.index'),
                JSON.stringify(this.docstore._docs)
            )
        ])
    }

    /**
     * Creates a new `VoyVectorStore` instance from an array of text strings. The text
     * strings are converted to `Document` instances and added to the Voy
     * database.
     * @param texts An array of text strings.
     * @param metadatas An array of metadata objects or a single metadata object. If an array is provided, it must have the same length as the `texts` array.
     * @param embeddings An `Embeddings` instance used to generate embeddings for the documents.
     * @param client An instance of Voy client to use in the underlying operations.
     * @returns A promise that resolves with a new `VoyVectorStore` instance.
     */
    static async fromTexts(
        texts: string[],
        metadatas: object[] | object,
        embeddings: EmbeddingsInterface,
        client: LunaDB,
        options?: LunaVDBLibArgs
    ): Promise<LunaDBVectorStore> {
        const docs: Document[] = []
        for (let i = 0; i < texts.length; i += 1) {
            const metadata = Array.isArray(metadatas) ? metadatas[i] : metadatas
            const newDoc = new Document({
                pageContent: texts[i],
                metadata
            })
            docs.push(newDoc)
        }
        return LunaDBVectorStore.fromDocuments(
            docs,
            embeddings,
            client,
            options
        )
    }

    /**
     * Creates a new `VoyVectorStore` instance from an array of `Document` instances.
     * The documents are added to the Voy database.
     * @param docs An array of `Document` instances.
     * @param embeddings An `Embeddings` instance used to generate embeddings for the documents.
     * @param client An instance of Voy client to use in the underlying operations.
     * @returns A promise that resolves with a new `VoyVectorStore` instance.
     */
    static async fromDocuments(
        docs: Document[],
        embeddings: EmbeddingsInterface,
        client: LunaDB,
        options?: LunaVDBLibArgs
    ): Promise<LunaDBVectorStore> {
        const instance = new LunaDBVectorStore(client, embeddings, options)
        await instance.addDocuments(docs)
        return instance
    }
}
