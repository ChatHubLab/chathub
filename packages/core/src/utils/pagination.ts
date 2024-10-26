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

        return items.slice(
            (page - 1) * limit,
            Math.min(items.length, page * limit)
        )
    }

    async formatItems(
        items: T[],
        page: number = this.input.page,
        limit: number = this.input.limit,
        total: number = Math.ceil(items.length / limit)
    ) {
        const buffer = [this.input.formatString.top]

        for (const item of items) {
            const itemLikePromise = this.input.formatItem(item)

            if (typeof itemLikePromise === 'string') {
                buffer.push(itemLikePromise)
            } else {
                buffer.push(await itemLikePromise)
            }
        }

        buffer.push(this.input.formatString.bottom)

        const formattedPageString = this.input.formatString.pages
            .replaceAll('[page]', Math.min(total, page).toString())
            .replaceAll('[total]', total.toString())

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
