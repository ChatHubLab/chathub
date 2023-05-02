import { Context } from 'koishi'

import { request, createLogger } from '@dingyi222666/koishi-plugin-chathub'
import CopilotHubAdapter from './index'
import randomUserAgent from "random-useragent"
import { CopilotResponse } from './types'

const logger = createLogger('@dingyi222666/chathub-copilothub-adapter/api')

//TODO: 后续支持更多平台，我没API KEY，等后续吧。
export class Api {

  
    private headers: Record<string,string> = {
        "content-type": "application/json",
      
        "User-Agent": randomUserAgent.getRandom(),
      
    }


    constructor(
        private readonly config: CopilotHubAdapter.Config,
    ) {
       
    }

   

    async request(prompt: string): Promise<string | Error> {
        const url = "https://api.copilothub.ai/openapi/v1/query"

        const body = JSON.stringify({
            "api_key": this.config.apiKey,
            "query": prompt,
        })

        const response = await request.fetch(url, {
            method: "POST",
            headers: this.headers,
            body
        })

        if (response instanceof Error) {
            return response
        }

        const json = (await response.json()) as CopilotResponse

        logger.debug(`copilot response: ${JSON.stringify(json)}`)

        if (json.result == null) {
            logger.error(`copilot response error: ${JSON.stringify(json)}`)
            return new Error("copilot response error")
        }

        return json.result

    }

  
}
