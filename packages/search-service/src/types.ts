export interface SearchResult {
    title: string
    url: string
    description: string
    image?: string
}

export enum SummaryType {
    Speed = 'speed',
    Balanced = 'balanced',
    Quality = 'quality'
}
