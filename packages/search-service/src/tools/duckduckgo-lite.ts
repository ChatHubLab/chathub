import SearchServicePlugin, { SearchTool, randomUA } from '..';
import { z } from "zod";
import { request } from "@dingyi222666/chathub-llm-core/lib/utils/request"
import { JSDOM } from "jsdom"

export default class DuckDuckGoSearchTool extends SearchTool {

    constructor(protected config: SearchServicePlugin.Config) {
        super()
    }

    async _call(arg: z.infer<typeof this.schema>): Promise<string> {

        const query = JSON.parse(arg).keyword as string

        const res = await request.fetch(`https://lite.duckduckgo.com/lite?q=${query}`, {
            headers: {
                // random ua
                "User-Agent": randomUA(),
            }
        })

        const html = await res.text()

        const doc = new JSDOM(html, {
            url: res.url
        })

        const result: ({
            title: string,
            url: string,
            description: string
        })[] = []

        const main = doc.window.document.querySelector("div.filters")

        let current = {
            title: "",
            url: "",
            description: "",
        }
        for (const tr of main.querySelectorAll("tbody tr")) {

            const link = tr.querySelector(".result-link")
            const description = tr.querySelector(".result-snippet")

            if (link) {
                current = {
                    title: link.textContent.trim(),
                    url: link.getAttribute("href"),
                    description: ""
                }
            } else if (description) {
                current.description = description.textContent.trim()
            }

            // if all data is ready(not empty), push to result

            if (current.title && current.url && current.description) {
                current.url = matchUrl("https:" + current.url)
                result.push(current)
                current = {
                    title: "",
                    url: "",
                    description: "",
                }
            }
        }

        return JSON.stringify(result.slice(0, this.config.topK))
    }

}

const matchUrl = (url: string) => {
    const match = url.match(/uddg=(.+?)&/)
    if (match) {
        return decodeURIComponent(match[1])
    }
    return url
}