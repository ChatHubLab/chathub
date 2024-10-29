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

    addDocumentsFunction: (
        ...args: Parameters<typeof this._store.addDocuments>
    ) => Promise<void>

    constructor(
        private _store: T,
        input: ChatLunaSaveableVectorStoreInput<T>
    ) {
        super(_store.embeddings, {})
        this.saveableFunction = input.saveableFunction ?? (async () => {})
        this.deletableFunction = input.deletableFunction
        this.addDocumentsFunction = input.addDocumentsFunction
    }

    addVectors(...args: Parameters<typeof this._store.addVectors>) {
        return this._store.addVectors(...args)
    }

    addDocuments(...args: Parameters<typeof this._store.addDocuments>) {
        if (this.addDocumentsFunction) {
            return this.addDocumentsFunction(...args)
        }
        return this._store.addDocuments(...args)
    }

    similaritySearchVectorWithScore(
        query: number[],
        k: number,
        filter?: T['FilterType']
    ) {
        return this._store.similaritySearchVectorWithScore(query, k, filter)
    }

    save() {
        return this.saveableFunction(this._store)
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
        ...args: Parameters<typeof this._store.addDocuments>
    ) => Promise<void>
}

export interface ChatLunaSaveableVectorDelete
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    extends Record<string, any> {
    deleteAll?: boolean
    documents?: Document[]
    ids?: string[]
}
