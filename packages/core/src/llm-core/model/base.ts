import { VectorStore } from '@langchain/core/vectorstores'
import { Document } from '@langchain/core/documents'

export class ChatLunaSaveableVectorStore<T extends VectorStore = VectorStore>
    extends VectorStore
    implements ChatLunaSaveableVectorStoreInput<T>
{
    saveableFunction: (store: T) => Promise<void>
    deletableFunction?: (
        store: T,
        input: ChatLunaSaveableVectorDelete
    ) => Promise<void>

    addDocumentsFunction?: (
        store: T,
        ...args: Parameters<T['addDocuments']>
    ) => Promise<void>

    getDocumentsByIdsFunction?: (store: T, ids: string[]) => Promise<Document[]>

    constructor(
        private _store: T,
        input: ChatLunaSaveableVectorStoreInput<T>
    ) {
        super(_store.embeddings, {})
        this.saveableFunction = input.saveableFunction ?? (async () => {})
        this.deletableFunction = input.deletableFunction
        this.addDocumentsFunction = input.addDocumentsFunction

        this.getDocumentsByIdsFunction = input.getDocumentsByIdsFunction
    }

    addVectors(...args: Parameters<typeof this._store.addVectors>) {
        return this._store.addVectors(...args)
    }

    addDocuments(...args: Parameters<T['addDocuments']>) {
        if (this.addDocumentsFunction) {
            return this.addDocumentsFunction(this._store, ...args)
        }
        return this._store.addDocuments(args[0], args[1])
    }

    similaritySearchVectorWithScore(
        query: number[],
        k: number,
        filter?: T['FilterType']
    ) {
        return this._store.similaritySearchVectorWithScore(query, k, filter)
    }

    async editDocument(oldDocumentId: string, newDocument: Document) {
        const documentId = await this.getDocumentsByIds([oldDocumentId])
        if (documentId.length === 0) {
            throw new Error('Document not found')
        }

        // delete
        await this.delete({ ids: [oldDocumentId] })

        // add
        await (this as ChatLunaSaveableVectorStore<VectorStore>).addDocuments([
            newDocument
        ])
    }

    async getDocumentsByIds(ids: string[]) {
        if (this.getDocumentsByIdsFunction) {
            return await this.getDocumentsByIdsFunction(this._store, ids)
        }
        return []
    }

    save() {
        return this?.saveableFunction(this._store)
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    delete(input: ChatLunaSaveableVectorDelete) {
        return this.deletableFunction(this._store, input)
    }

    _vectorstoreType(): string {
        return this._store?._vectorstoreType() ?? '?'
    }
}

export interface ChatLunaSaveableVectorStoreInput<T extends VectorStore> {
    saveableFunction?: (store: T) => Promise<void>
    deletableFunction?: (
        store: T,
        input: ChatLunaSaveableVectorDelete
    ) => Promise<void>
    addDocumentsFunction?: (
        store: T,
        ...args: Parameters<T['addDocuments']>
    ) => Promise<void>
    getDocumentsByIdsFunction?: (store: T, ids: string[]) => Promise<Document[]>
}

export interface ChatLunaSaveableVectorDelete
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    extends Record<string, any> {
    deleteAll?: boolean
    documents?: Document[]
    ids?: string[]
}
