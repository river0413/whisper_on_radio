# 日语广播 → 中日双语字幕

四个可独立执行的脚本与 Windows PowerShell 一键流程（Python 3.10+），均提供 tqdm 进度条：

| 脚本 | 作用 |
|---|---|
| `scripts\01_get_audio.py <音频>` | ffmpeg 转 16kHz 单声道 WAV（输出到 `work\`） |
| `scripts\02_transcribe_srt.py <wav>` | 本地 faster-whisper（默认 `large-v3-turbo`，模型存 `models\`）→ `.ja.srt`；相邻识别片段会合并为句子，但每条字幕最长 5 秒（句末标点或超过 `--sentence-gap` 的停顿也会提前切开，默认 1 秒）；Whisper 单个片段超过 5 秒时会按文本长度近似分段并均分时间（Whisper 未提供词级时间戳）；GPU 失败自动回退 CPU。可用 `--initial-prompt`、`--hotwords` 注入节目名/专有名词，`--condition-on-previous-text off`、`--hallucination-silence-threshold` 抑制重复幻觉，`--vad-min-silence` 调整切分灵敏度 |
| `scripts\03_translate_srt.py <ja.srt>` | Ollama 混元 Hy-MT（默认 `hy-mt1.5:7b`）本地初译 → `.zh.srt`（中文初译，时间轴与日文一致）；可选 JSON 术语表和 Markdown 背景参考 |
| `scripts\04_proofread_srt.py <ja.srt> --zh-srt <zh.srt> --suspects-only` | 对本地初译进行质量门控，只将可疑/疑难行交给关闭推理的 DeepSeek（`deepseek-v4-pro`）精修；校对请求附日文、罗马音和中文初译；正常行不发送 API |

校对完成后会生成两份字幕：`.zh-ja.srt` 是最终日中双语字幕（每条依次为日文、中文）；`.zh-ja-romaji.srt` 是校对稿（每条依次为日文、罗马音、中文）。罗马音仅用于读音校对，DeepSeek 不会改写或输出罗马音。

转录后会清理疑似幻觉片段：重复率 `compression_ratio > 2.4` 的片段会被舍弃；「ご視聴ありがとうございました」「チャンネル登録」「提供」等黑名单短语仅在 `no_speech_prob > 0.3` 时舍弃。时长不超过 1 秒的短片段，只有在 `no_speech_prob > 0.6` 且 `avg_logprob < -1.0` 时才按低置信度舍弃；其余短片段会尝试与前后相邻字幕合并，单侧间隔最多 300 毫秒，不跨句末标点，合并后仍受 5 秒上限约束。没有合并邻居时保留为单独字幕。`no_speech_prob` 表示模型认为片段没有语音的概率，并非转录准确率。

混元使用 Ollama 的编号输入格式和推荐采样参数（`top_k=20`、`top_p=0.6`、`temperature=0.7`、`repeat_penalty=1.05`）；术语表使用 UTF-8 JSON，例如 `{"番組名":"节目名称","山田さん":"山田"}`，运行时通过 `--terms-file terms.json` 注入。也可用 `--reference-file coconna.md` 加载 UTF-8 Markdown 节目/人物资料，内容作为 system 背景参考，不会作为待翻译条目发送；参考内容只用于核对人名、节目设定和专有词，不替代翻译指令。`--batch-size` 默认 50 条；模型交互分组为 40 条，漏编号时只重试缺失编号。请先在 Ollama 安装并确认所用 Hy-MT 模型标签可运行。

DeepSeek 质量门控将长度比例异常、残留假名、数字/大写缩写不一致、术语表译法缺失以及连续重复行首语气词标为 `suspect`；明显双关提示、日语指代词密集或含大写缩写的疑难行标为 `hard`。两类都交给 `DEEPSEEK_CHAT_MODEL`（默认 `deepseek-v4-pro`，全流程统一使用该模型），精修请求关闭 thinking。请求包含目标行的日文和中文初译，并可附前后各一条日文作只读上下文；上下文行不要求模型输出，也不修改其译文。未命中行不调用 API，最终仍保留全部行的日中双语格式。门控是启发式筛选，不代表自动判断翻译正确；需要时仍可不加 `--suspects-only`，使用既有全量 DeepSeek 校对、日文修正和合并/删除能力。

DeepSeek API 会统计实际返回的输入/输出 token，并按模型汇总。`deepseek-v4-pro` 沿用脚本内配置的参考费率；其他模型只统计 token、不显示费用估算，避免在费率不匹配时误报。若 API 未返回用量，会注明统计不完整。可用 `--no-token-usage` 关闭统计。

使用 `--zh-srt` 校对时，模型还可修正日文转录、合并连续编号的字幕，或用 `<DELETE>` 标记删除确认属于幻听/无效噪声的条目。合并行格式为 `起始编号-结束编号<Tab>修正日文<Tab>校对后中文`；删除行格式为 `编号<Tab><DELETE><Tab><DELETE>`。合并字幕采用首条开始时间和末条结束时间，删除条目不会写入输出 SRT。直接翻译模式不执行这些校对结构操作。

校对默认关闭 DeepSeek 推理（`--deepseek-thinking disabled`，即 `DEEPSEEK_THINKING=disabled`）。`deepseek-v4-pro` 是推理模型，长上下文时推理会占满输出额度导致回复正文为空；若换成不支持该参数的模型，脚本会自动去掉该参数重试，也可用 `--deepseek-thinking auto` 恢复默认推理。

最终输出默认会将相邻短字幕合并为同一字幕条目，保留首尾原时间戳；可用 `--no-merge-short` 关闭，或通过 `--short-cue-chars`、`--merge-gap` 调整阈值。

可用 `--proofread-prompt-file` 指定 UTF-8 Markdown 文件，为 DeepSeek 校对提示词追加专名、语气等自定义要求，例如：`python scripts\04_proofread_srt.py 广播.ja.srt --proofread-prompt-file proofreading.md`。自定义内容只影响 DeepSeek 校对，不替代脚本的字幕编号和输出格式要求。

一键运行（推荐）：`.\run_all.ps1 -Audio 广播.mp3`。该流程依次转 WAV、转录、Hy-MT 本地初译、DeepSeek 全量强校对（默认开启推理，可改日文识别错误、合并/整条删除字幕，并带入术语表与背景资料 Markdown 对照）。可通过 `-TermsFile terms.json` 加载术语表、`-ReferenceFile coconna.md` 同时用于 Hy-MT 与校对背景参考，或通过 `-PromptFile proofreading.md` 添加校对要求。

`run_all.ps1` 会读取 `.env` 中的 `PROOFREAD_THINKING`、`PROOFREAD_LINES`、`PROOFREAD_MAX_TOKENS`：当前配置为推理 `auto`（开启）、200 条/段、输出上限 384000 tokens，推理 token 与正文共用这一上限。可用 `-ProofreadThinking disabled` 关闭推理，或设环境变量 `PROOFREAD_THINKING=disabled` 覆盖。关闭推理时建议把 `PROOFREAD_MAX_TOKENS` 调回 8192 并把分批降到 50 条，避免长回复被截断。如需省 token 的可疑行精修，可手动给 04 脚本加 `--suspects-only`。

校对时会向模型提供每条的时长，并要求合并后的字幕总时长不超过 8 秒、只在同一句话内合并，避免推理模型把整段对话并成一条超长字幕。

也可分步运行：
```powershell
python scripts\03_translate_srt.py 广播.ja.srt --terms-file terms.json --reference-file coconna.md
python scripts\04_proofread_srt.py 广播.ja.srt --zh-srt 广播.zh.srt --suspects-only
```

## PDF 学术翻译（混元 Hy-MT2）

`translate_pdf_hy_mt2.ps1` 把 PDF 论文正文翻译为学术汉语 Markdown。PDF 文本抽取由内嵌在脚本里的 Python 代码完成（运行时写入系统临时目录，结束后自动删除），翻译、缺号重试和 Markdown 拼装都在 PowerShell 中完成，因此工作区只需保留这一个脚本。

```powershell
.\translate_pdf_hy_mt2.ps1 .\paper.pdf
.\translate_pdf_hy_mt2.ps1 .\paper.pdf -Output .\out\paper.zh.md -TermsFile .\terms.json
.\translate_pdf_hy_mt2.ps1 .\papers\*.pdf -OutputDirectory .\out -TermsFile .\terms.json
.\translate_pdf_hy_mt2.ps1 .\a.pdf, .\b.pdf, .\c.pdf -OutputDirectory .\out
```

多个 PDF 可一次传入：用逗号分隔多个路径，或使用通配符；重复路径会自动去重并按出现顺序处理。每个文件独立翻译，某个文件失败不会中断其余文件，结束时打印成功/失败统计，存在失败项时退出码为 1。

| 参数 | 说明 |
|---|---|
| `-Pdf` | 输入 PDF（位置参数，必填），可传多个或用通配符 |
| `-Output` | 输出 Markdown，默认 `<输入文件名>.academic.zh.md`；仅适用于单个输入 |
| `-OutputDirectory` | 输出目录，默认输出到各 PDF 所在目录；不存在时自动创建；输出文件名冲突会报错而不覆盖 |
| `-TermsFile` | 术语表 JSON，格式与 `03_translate_srt.py` 一致：`{"attention":"注意力机制"}` |
| `-Model` | Ollama 模型标签，默认取 `.env` 的 `HY_MT_MODEL`，回退 `hf.co/tencent/Hy-MT2-7B-GGUF:Q4_K_M` |
| `-OllamaHost` | Ollama 地址，默认取 `.env` 的 `OLLAMA_HOST_URL`，回退 `http://127.0.0.1:11434` |
| `-ChunkSize` | 每批送翻译的字符数，默认 1800 |
| `-TimeoutSec` | 单次请求超时秒数，默认 900 |
| `-NumCtx` | Ollama 上下文长度，默认 8192 |
| `-NoPageMarkers` | 不输出 `<!-- page: N -->` 页码注释 |

处理要点：

- PDF 抽取优先使用 `pymupdf`，也可回退 `fitz`（PyMuPDF）；未安装时提示 `pip install PyMuPDF`。
- 双栏页面按中缝识别栏位，先左栏后右栏；跨栏的通栏标题、图注作为分隔带处理。
- 依据字号与粗体识别标题并映射为 `#`、`##`、`###`；摘要、引言、方法、结果、结论等常见章节名即使同字号也会识别为标题。
- 识别到 `References`、`Bibliography`、`参考文献` 等标题后，后续条目保留原文不送模型；遇到 `Appendix`/`附录` 等标题恢复翻译。
- 长段落按句末标点切分后逐段翻译，同一段落的多段译文按中文习惯直接拼接。
- 每批按“编号<Tab>原文”送入模型；模型漏掉编号时只重试缺失条目，仍缺失则报错退出，不写出半成品。
- 无文本层的扫描件会直接报错并提示先做 OCR，不会生成空文件。
- 推理参数为 `top_k=20`、`top_p=0.6`、`temperature=0.3`、`repeat_penalty=1.05`；较低的 temperature 用于减少数字与术语漂移。
- 读取 Ollama 返回的 `prompt_eval_count`、`eval_count` 与 `eval_duration`，每个批次、每个文件以及多文件合计都会显示输入/输出 token 数和输出速度（输出 token ÷ 生成耗时）。速度只统计模型生成阶段，不含 PDF 抽取、请求排队和模型加载；若 Ollama 未返回这些字段则显示 `n/a`。

局限：纯文本抽取无法还原表格结构、公式排版和图注位置；公式若以文本形式存在会保留符号，图片中的内容不会被识别。

## 准备
1. `pip install -r requirements.txt`（包含 tqdm、PyMuPDF、pykakasi），并安装 ffmpeg、Ollama。
2. 运行 `ollama pull hy-mt1.5:7b` 安装 Hy-MT（如使用自建 Modelfile/GGUF 或其他 Ollama 标签，在 `.env` 中将 `HY_MT_MODEL` 设为对应模型名）。
3. 复制 `.env.example` 为 `.env`，填写 `DEEPSEEK_API_KEY`（官方 API 密钥，仅本地保存）。根据账号可用模型调整 `DEEPSEEK_CHAT_MODEL`。

一键流程要求本地 Ollama 服务及 DeepSeek API 密钥可用；精修调用失败会报错退出，不会把失败精修伪装为成功结果。只有质量门控命中的行才会产生 DeepSeek 请求。

## 测试
`python -m pytest tests`
