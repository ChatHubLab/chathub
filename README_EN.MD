<div align="center">

<img src="./logo.png" alt="Project Logo" title="Project Logo" width="300"/>

[中文](./README.MD) | English

#

_A bot plugin for LLM chat services with multi-model integration, extensibility, and various output formats._

[![npm](https://img.shields.io/npm/v/koishi-plugin-chatluna)](https://www.npmjs.com/package/koishi-plugin-chatluna) [![npm](https://img.shields.io/npm/dm/koishi-plugin-chatluna)](https://www.npmjs.com/package/koishi-plugin-chatluna) ![node version](https://img.shields.io/badge/node-%3E=18-green) ![github top language](https://img.shields.io/github/languages/top/ChatLunaLab/chatluna?logo=github)

[![Telegram](https://img.shields.io/badge/Join-Telegram_Group-blue)](https://t.me/koishi_chatluna) [![QQ](https://img.shields.io/badge/Join-QQ_Group-ff69b4)](http://qm.qq.com/cgi-bin/qm/qr?_wv=1027&k=eEBVq6GK7HYX2y61x55WD6hnXTIRop-0&authKey=i4pG5%2BJ%2FY8auWprBubhremTkn3vroPigQq5m9RENGBLrLmlj%2BSu3G%2BqllK7Wts2M&noverify=0&group_code=282381753) [![doc](https://img.shields.io/badge/See-Document(WIP)-green)](https://chatluna.chat/)

**Project Status: Steadily iterating towards the official 1.0 release (currently in Release Candidate stage)**

</div>

## Screenshots

| Preset | Plugin Mode & Streaming Output | Image Rendering  |
|---------------------|--------------------------|--------------------|
| ![preset](https://s2.loli.net/2024/10/17/hfyoKXtJC5dV2Lg.webp) | ![plugin](https://s2.loli.net/2024/10/17/pcDjSIVMkB1aO6L.webp) | ![code](https://s2.loli.net/2024/10/17/7TBlbDsPM95iwdr.gif) |

## Features

- 🚀 Highly extensible (LangChain & Koishi APIs)
- 🎭 Custom conversation presets
- 🛡️ Rate limiting & blacklist system
- 🎨 Multi-format output (text, voice, image, mixed)
- 🧠 Context-aware with [long-term memory](./packages/long-memory/README.md)
- 🔀 Three modes: chat, browsing, plugin
- 🔒 Content moderation via [Koishi censor](https://censor.koishi.chat/)

## Roadmap

- [x] Room-based conversation system
- [x] Content moderation
- [x] TTS support (via [vits service](https://github.com/initialencounter/2022-12-24/blob/neat/plugins/Tool/vits/readme.md))
- [x] Image rendering for replies
- [x] Multi-model integration
- [x] Preset system
- [x] ~~Conversation import/export~~ (abandoned)
- [x] v1 refactoring
- [x] Streaming responses
- [x] i18n support

## Quick Start

Install the plugin in Koishi for basic functionality. For detailed setup, see our [docs](https://chatluna.chat/guide/getting-started.html).

## Supported Models

| Model/Platform | Integration | Features | Notes |
|:---------------|:------------|:---------|:------|
| [OpenAI](./packages/openai-adapter/README.md) | Official API | Customizable, plugin/browsing modes | Paid API |
| [Azure OpenAI](./packages/azure-openai-adapter/README.md) | Official API | Similar to OpenAI | Paid API |
| [Google Gemini](./packages/gemini-adapter/README.md) | Official API | Fast, outperforms GPT-3.5 | Requires account, may be charged |
| [Claude API](./packages/claude-adapter) | Official API | Large context, often beats GPT-3.5 | Paid, no Function Call |
| [Deepseek](./packages/deepseek-adapter/) | Official API | Domestic renowned model | Performance close to or exceeding GPT-4o |
| [Tongyi Qianwen](./packages/qwen-adapter/README.md) | Official API | Free quota available | Slightly better than Zhipu |
| [Zhipu](./packages/zhipu-adapter/README.md) | Official API | Free tokens for new users | Better than Xunfei Spark |
| [Xunfei Spark](./packages/spark-adapter/README.md) | Official API | Free quota for new users | Similar to GPT-3.5 |
| [Wenxin Yiyan](./packages/wenxin-adapter/README.md) | Official API | Baidu's model | Slightly worse than Xunfei Spark |
| [Hunyuan](./packages/hunyuan-adapter/README.md) | Official API | Tencent's model | Better than Wenxin Yiyan |
| [Ollama](./packages/ollama-adapter/README.md) | Self-hosted | Open-source, CPU/GPU support | Requires backend setup |
| [GPT Free](./packages/gptfree-adapter/README.md) | Unofficial | Uses other websites' GPT models | Unstable, may fail |
| [ChatGLM](./packages/chatglm-adapter/README.md) | Self-hosted | Can be self-hosted | Requires backend, suboptimal performance |
| [RWKV](./packages/rwkv-adapter/README.md) | Self-hosted | Open-source model | Requires backend setup |

[Web search support](/packages/search-service/README.md):

- Google (API)
- Bing (API & Web)
- DuckDuckGO (Lite)
- Tavily (API)

## Presets

From `1.0.0-alpha.10`, we use YAML for more customizable presets. Default preset: [catgirl.yml](/packages/core/resources/presets/catgirl.yml)

Preset folder: `<koishi_dir>/data/chathub/presets`

For more info, see [preset system docs](https://chatluna.chat/guide/preset-system/introduction.html).

## Development

Clone the repo:

```bash
# yarn
yarn clone ChatLunaLab/chatluna
# npm
npm run clone ChatLunaLab/chatluna
```

Update `tsconfig.json`:

``` json
{
  "extends": "./tsconfig.base",
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "koishi-plugin-chatluna-*": ["external/chatluna/packages/*/src"]
    }
  }
}
```

Build the project:

``` bash
# yarn
yarn workspace @root/chatluna-koishi build
# npm
npm run build -w @root/chatluna-koishi
```

Start development with `yarn dev` or `npm run dev`.

> Note: HMR may not be fully compatible. If issues occur, rebuild and restart.

## Contributing

We need help with:

- [ ] Web UI
- [ ] HTTP Server
- [ ] Project Documentation

PRs and discussions are welcome!

## Contributors

<a href="https://github.com/ChatLunaLab/chatluna/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ChatLunaLab/chatluna" />
</a>

![Alt](https://repobeats.axiom.co/api/embed/d92b5a9398ee02a797355ed9b866aa93231c25b3.svg "Repobeats analytics image")

[![Star History Chart](https://api.star-history.com/svg?repos=ChatLunaLab/chatluna&type=Date)](https://star-history.com/#ChatLunaLab/chatluna)

## Usage Notice

Developed by [ChatLunaLab](https://github.com/ChatLunaLab).

ChatLuna is an LLM-based chatbot framework. We collaborate with the open-source community to advance LLM technology. Users must comply with open-source agreements and avoid using this project for potentially harmful purposes or unevaluated services.

This project doesn't provide AI services directly. Users must obtain API access from AI service providers.

Users are responsible for complying with local laws and using locally available AI services.

The project isn't responsible for algorithm-generated results. All results and operations are the user's responsibility.

Users configure their own data storage. The project doesn't provide direct data storage.

The project isn't liable for user-caused data security issues, public opinion risks, or model misuse.

## Acknowledgments

Thanks to these projects:

[koishi-plugin-openai](https://github.com/TomLBZ/koishi-plugin-openai)

[node-chatgpt-api](https://github.com/waylaidwanderer/node-chatgpt-api)

[poe-api](https://github.com/ading2210/poe-api)

[Bard](https://github.com/muhiris/wgpt)

[chathub](https://github.com/chathub-dev/chathub)

Thanks to [JetBrains](https://www.jetbrains.com/?from=ChatLuna) for open-source [WebStorm](https://www.jetbrains.com/webstorm/?from=ChatLuna) licenses.

[<img src=".github/jetbrains-variant-3.png" width="200"/>](https://www.jetbrains.com/?from=ChatLuna)
