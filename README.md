# 日语广播转录与中日双语字幕
一套面向日语广播音频的 Windows 命令行工具：将音频转成日文字幕，使用本地模型生成中文初译，再按需调用 DeepSeek 校对，输出中日双语 SRT。脚本可单独运行，也可通过 PowerShell 一键完成整个流程。

## 工作流程

```text
音频
  -> 16 kHz 单声道 WAV
  -> faster-whisper 或 Qwen3-ASR + ForcedAligner 日文转录（.ja.srt）
  -> Ollama / Hy-MT 中文初译（.zh.srt）
  -> DeepSeek 校对（.zh-ja.srt）
```

音频转 WAV、ASR 转录和 Hy-MT 初译都在本机运行。默认 ASR 是 faster-whisper；可选安装 Qwen3-ASR-0.6B 与 Qwen3-ForcedAligner-0.6B 后切换后端。Whisper 模式默认会把低置信度片段交给 Qwen 重新识别；校对步骤会调用 DeepSeek API；一键流程默认校对全部字幕，因此会产生 API 用量。也可以选择只把质量门控标记为可疑或疑难的字幕发送给 API。

## 环境要求

- Windows 10/11 和 PowerShell
- Python 3.10 或更高版本
- [FFmpeg](https://ffmpeg.org/)；安装后确认 `ffmpeg` 可在终端中运行
- [Ollama](https://ollama.com/)；用于运行 Hy-MT 中文初译模型
- DeepSeek API 密钥；用于字幕校对

首次转录时，faster-whisper 会下载所选模型。默认模型为 `large-v3-turbo`，缓存到项目的 `models\` 目录；模型文件较大，不会提交到 Git。

Qwen3 是可选依赖，不会随默认依赖安装。Qwen 的 PyTorch/Transformers 依赖可能与已有环境冲突；建议在独立的 Python 3.12 虚拟环境中安装基础依赖和 Qwen 依赖：

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-qwen-asr.txt
```

Qwen 模型在首次运行时自动下载到 Hugging Face 缓存。默认使用 `Qwen/Qwen3-ASR-0.6B` 和 `Qwen/Qwen3-ForcedAligner-0.6B`；两者均可在 `.env` 中通过 `QWEN3_ASR_MODEL`、`QWEN3_ALIGNER_MODEL` 指定本地目录或其他模型标签。Qwen 后端默认自动选择设备和精度，可通过 `QWEN3_DTYPE`、`WHISPER_DEVICE` 调整。完整 Qwen 流程需要 FFmpeg/ffprobe；长音频会拆成不超过 4 分钟的片段并使用少量重叠，以适配对齐器时长限制。

## 安装与配置

在 PowerShell 中克隆项目并安装 Python 依赖：

```powershell
git clone https://github.com/river0413/whisper-on-radio.git
cd whisper-on-radio
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

复制环境变量模板并编辑 `.env`，至少填写 `DEEPSEEK_API_KEY`：

```powershell
Copy-Item .env.example .env
notepad .env
```

`.env` 包含密钥，只保存在本机，不要提交或分享。其他常用设置包括 Whisper 模型、Ollama 服务地址和 Hy-MT 模型标签；默认值见 [.env.example](.env.example)。

翻译时脚本会先检查 Ollama 服务；本机默认服务未运行时，会自动执行 `ollama serve` 并等待服务就绪。请先安装 Ollama、将 `ollama` 命令加入 `PATH`，并确保 `.env` 中指定的模型已下载。默认单次上下文上限为 1800 tokens，低于 2048；可用 `.env` 的 `OLLAMA_NUM_CTX` 或 `--num-ctx` 调整，范围为 1–2047。

初译请求会复用脚本 05 的字符启发式估算实际系统提示词、术语/参考资料和字幕内容，默认每次请求最多 900 个估算输入 tokens（`OLLAMA_MAX_INPUT_TOKENS` 或 `--max-input-tokens`）。超过上限时自动拆成更小的字幕组；每次请求的 `num_ctx` 则按估算输入、预期输出和 64-token 余量动态设置，且不超过 `OLLAMA_NUM_CTX`。估算不是模型 tokenizer 的精确值；减小上下文可以降低显存需求，但如果模型权重本身无法放入显存，仍可能由 Ollama 部分分配到 CPU。

```powershell
ollama pull hf.co/tencent/Hy-MT2-7B-GGUF:Q4_K_M
```

如果你使用其他 Ollama 模型，请将 `.env` 中的 `HY_MT_MODEL` 改为对应标签。

## 一键生成双语字幕

```powershell
.\run_all.ps1 -Audio ".\radio.mp3"
```

脚本按顺序执行音频转换、日语转录、本地中文初译和 DeepSeek 全量校对。可选参数：

```powershell
.\run_all.ps1 -Audio ".\radio.mp3" `
  -TermsFile ".\terms.json" `
  -ReferenceFile ".\program-notes.md" `
  -PromptFile ".\proofreading.md"
```

ASR 后端默认是 Whisper。可以在本次运行中切换到 Qwen：

```powershell
.\run_all.ps1 -Audio ".\radio.mp3" -AsrBackend qwen3
```

也可以在 `.env` 中设置 `ASR_BACKEND=qwen3` 作为默认值；命令行 `-AsrBackend` 优先级更高。单独运行转录脚本时同样支持 `--backend qwen3` 或 `ASR_BACKEND=qwen3`：

```powershell
python scripts\02_transcribe_srt.py ".\work\radio.wav" --backend qwen3
```

若使用默认 Whisper 后端但不想自动调用 Qwen 辅助，可在一键流程中传入 `-NoQwenAssist`，或单独运行时传入 `--no-qwen-assist`。

Qwen 后端仍输出标准 `.ja.srt`，可继续用于中文翻译和校对。ForcedAligner 提供的是转录文本的时间戳，不是 Whisper 的 `avg_logprob` 置信度；因此 Qwen 生成的字幕不会带 Whisper 置信度元数据，基于低置信度自动筛选的流程不会将其识别为低置信度字幕。切换模型也不会保证识别结果与 Whisper 相同。

Whisper 后端默认启用低置信度辅助：当单个 Whisper 片段的 `avg_logprob <= ln(0.85) ≈ -0.1625` 时，脚本会用 Qwen3-ASR + ForcedAligner 重识别该片段；模型只在首次遇到此类片段时加载。Qwen 返回有效文本时替换该片段，Qwen 依赖、模型或推理不可用时会显示警告并保留 Whisper 原文。替换后保留的置信度仍是 Whisper 对原片段的评分，不代表 Qwen 置信度。可用 `--no-qwen-assist` 或 `-NoQwenAssist` 关闭本次运行，也可设 `QWEN3_LOW_CONF_ASSIST=0`；阈值与后续低置信度标记共用 `WHISPER_CONFIDENCE_THRESHOLD`，默认等效 0.85。

- `-TermsFile`：UTF-8 JSON 术语表，统一节目名、人名等译法，例如 `{"番組名":"节目名称"}`。
- `-ReferenceFile`：UTF-8 Markdown 背景资料，供初译和校对参考。
- `-PromptFile`：UTF-8 Markdown 校对补充要求。
- `-ProofreadThinking disabled`：关闭校对推理；也可在 `.env` 中设置 `PROOFREAD_THINKING=disabled`。

默认开启校对推理，且模板的 `PROOFREAD_MAX_TOKENS` 上限较高。请根据模型能力和 API 预算调整 `.env` 中的 `PROOFREAD_THINKING`、`PROOFREAD_LINES` 和 `PROOFREAD_MAX_TOKENS`。关闭推理时，可从每批 50 条、`PROOFREAD_MAX_TOKENS=8192` 开始。

默认输出到 `work\`，例如：

| 文件 | 内容 |
|---|---|
| `radio.wav` | 转换后的音频 |
| `radio.ja.srt` | 日文转录 |
| `radio.zh.srt` | 中文初译 |
| `radio.zh-ja.srt` | 最终中日双语字幕，每条先日文后中文 |

`work\`、`models\`、音频文件和 `.env` 均已加入 `.gitignore`，不会随源码提交。首次下载模型或处理长音频时，运行时间和磁盘占用可能较大。

## 分步运行

如果需要检查或重复执行某个阶段，可以单独运行脚本：

```powershell
# 转换音频；默认输出 work\radio.wav
python scripts\01_get_audio.py ".\radio.mp3"

# 日语转录；默认生成与输入 WAV 同目录的 radio.ja.srt
python scripts\02_transcribe_srt.py ".\work\radio.wav"

# Ollama / Hy-MT 本地初译；默认生成 radio.zh.srt
python scripts\03_translate_srt.py ".\work\radio.ja.srt"

# 校对初译；默认生成中日双语字幕
python scripts\04_proofread_srt.py ".\work\radio.ja.srt" `
  --zh-srt ".\work\radio.zh.srt"
```

只精修质量门控标记的可疑/疑难字幕，可加 `--suspects-only`，减少发送给 DeepSeek 的内容：

```powershell
python scripts\04_proofread_srt.py ".\work\radio.ja.srt" `
  --zh-srt ".\work\radio.zh.srt" `
  --suspects-only
```

这种模式仍需要 DeepSeek API 密钥；若没有字幕被标记，则不会发送精修请求。门控使用启发式规则筛选，不代表自动确认翻译正确。省略 `--suspects-only` 则执行全量校对；校对过程中模型也可修正日文转录、合并字幕或删除确认无效的条目。

需要在调用 API 前估算输入和字幕输出的 token 数，可单独运行 `scripts\05_token_counter.py`。该工具不发送 API 请求，也不需要 API 密钥：

```powershell
# 估算日文字幕直接翻译
python scripts\05_token_counter.py ".\work\radio.ja.srt"

# 估算日文原文与中文初译的校对
python scripts\05_token_counter.py ".\work\radio.ja.srt" `
  --zh-srt ".\work\radio.zh.srt" `
  --proofread-lines 100 `
  --terms-file ".\terms.json" `
  --reference-file ".\program-notes.md" `
  --prompt-file ".\proofreading.md"
```

输入估算包括系统提示词、字幕批次和提供的参考内容；输出估算按原字幕/初译长度推算。计数基于字符启发式，并非模型 tokenizer 的精确结果；不会包含 API 重试或推理 token。

转录生成的 `.ja.srt` 会在字幕行尾保存 Whisper `avg_logprob` 元数据，例如 `<-0.75->`。启用低置信度 Qwen 辅助时，Qwen 替换文本仍保留原 Whisper 置信度元数据，方便后续复核；元数据表示 Whisper 对原片段的评分，不是 Qwen 的评分。中文初译不会把该标记发送给本地模型；使用 `--suspects-only` 时，低于或等于默认阈值 85%（`avg_logprob <= ln(0.85) ≈ -0.1625`）的行也会进入精修，并把置信度作为单独提示及相邻日文上下文提供给 DeepSeek。可用 `--confidence-threshold` 调整阈值。最终双语字幕会在低置信度行的日文后紧接百分比标注，例如 `【low conf：75.2%（概率近似）】`；该百分比是 `exp(avg_logprob) × 100%` 的模型概率近似值，不是经校准的识别正确率。最终双语 SRT 中，超过 50 个字符的字幕行会自动换行，优先在置信度标签前断行，其次在顿号等标点后断行。

可用 `--max-duration 300` 只转录音频开头五分钟，便于评估置信度阈值；例如：

```powershell
python scripts\02_transcribe_srt.py ".\work\INT_20261021013000_20261021020000.wav" `
  --max-duration 300
```

Whisper 支持注入专有名词，帮助识别节目名或人名：

```powershell
python scripts\02_transcribe_srt.py ".\work\radio.wav" `
  --initial-prompt "节目名、主持人名等日语专有名词" `
  --hotwords "节目名,主持人名"
```

各脚本的完整参数可通过 `--help` 查看，例如：

```powershell
python scripts\02_transcribe_srt.py --help
python scripts\03_translate_srt.py --help
python scripts\04_proofread_srt.py --help
```

## 合成带字幕的视频

`scripts\06_mux_video.ps1` 用 FFmpeg 把音频、SRT 字幕和封面图合成为可直接分享的视频。默认输出 MP4，字幕烧进画面；加 `-Soft` 则输出 MKV，字幕作为可开关的字幕轨道。

```powershell
.\scripts\06_mux_video.ps1 -Audio ".\work\radio.wav" -Image ".\cover.jpg"
```

脚本会自动在音频所在目录和 `work\` 中查找同名字幕，依次尝试 `.zh-ja.srt`（中日双语）、`.zh.srt`、`.ja.srt`；`-Image` 省略时查找 `cover.jpg`、`封面.png` 等，都找不到则使用纯色背景。

常用参数：

- `-Out`：输出路径，默认是音频同目录的 `<音频名>.mp4`（`-Soft` 时为 `.mkv`）。
- `-Size 1280x720`：输出分辨率，默认 `1920x1080`。
- `-Fit`：裁剪填满画面；默认等比缩放并补黑边，图片不变形。
- `-SrtShift -0.5`：字幕整体提前 0.5 秒，正数则延后。
- `-FontName`、`-FontSize 22`、`-MarginV 40`：字幕字体、字号与底边距。
- `-Crf 23`、`-Preset medium`、`-Fps 10`：画质、编码速度与帧率；静态封面用较低帧率即可。
- `-PreviewSeconds 30`：只编码前 30 秒，用来快速确认排版。
- `-NoSubtitle`：不合成字幕；`-Soft`：字幕作为独立轨道而非烧进画面。
- `-Force`：覆盖已存在的输出文件。

字幕文件名含中文或空格都可以直接使用；输出目录不存在时会自动创建。

## 测试

安装依赖后，在项目根目录运行：

```powershell
python -m pytest tests
```
