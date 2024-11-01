export class Pagination<T> {
    private _cacheMap: Record<string, T[]> = {}

    constructor(private input: PaginationInput<T>) {
        input.page = input.page ?? 1
        input.limit = input.limit ?? 5
    }

    async push(items: T[], key: string = 'default') {
        this._cacheMap[key] = items
    }

    async getPage(
        page: number = this.input.page,
        limit: number = this.input.limit,
        key: string = 'default'
    ) {
        const items = this._cacheMap[key]

        if (!items) {
            return []
        }

        const normalizedPage = Math.max(1, page)
        const normalizedLimit = Math.max(1, limit)

        return items.slice(
            (normalizedPage - 1) * normalizedLimit,
            Math.min(items.length, normalizedPage * normalizedLimit)
        )
    }

    async formatItems(
        items: T[],
        page: number = this.input.page,
        limit: number = this.input.limit,
        total?: number
    ) {
        const buffer = [this.input.formatString.top]
        const actualTotal = total ?? Math.ceil(items.length / limit)

        const formatPromises = items.map((item) => {
            const result = this.input.formatItem(item)
            return result instanceof Promise ? result : Promise.resolve(result)
        })
        const formattedItems = await Promise.all(formatPromises)
        buffer.push(...formattedItems)

        buffer.push(this.input.formatString.bottom)

        const formattedPageString = this.input.formatString.pages
            .replaceAll('[page]', Math.min(actualTotal, page).toString())
            .replaceAll('[total]', actualTotal.toString())

        buffer.push(formattedPageString)

        return buffer.join('\n')
    }

    async getFormattedPage(
        page: number = this.input.page,
        limit: number = this.input.limit,
        key: string = 'default'
    ) {
        const sliceItems = await this.getPage(page, limit, key)

        return this.formatItems(
            sliceItems,
            page,
            limit,
            Math.ceil(this._cacheMap[key].length / limit)
        )
    }

    async searchPage(
        find: (value: T) => boolean,
        page: number = this.input.page,
        limit: number = this.input.limit,
        key: string = 'default'
    ) {
        const items = this._cacheMap[key]?.filter(find) ?? []

        return this.formatItems(items, page, limit)
    }

    updateFormatString(formatString: PaginationInput<T>['formatString']) {
        this.input.formatString = formatString
    }

    updateFormatItem(formatItem: PaginationInput<T>['formatItem']) {
        this.input.formatItem = formatItem
    }

    getTotalPages(key: string = 'default'): number {
        const items = this._cacheMap[key]
        if (!items) return 0
        return Math.ceil(items.length / (this.input.limit ?? 5))
    }

    hasPage(page: number, key: string = 'default'): boolean {
        return page > 0 && page <= this.getTotalPages(key)
    }
}

export interface PaginationInput<T> {
    page?: number
    limit?: number

    formatItem(item: T): Promise<string> | string
    formatString: {
        top: string
        bottom: string
        pages: string
    }
}
