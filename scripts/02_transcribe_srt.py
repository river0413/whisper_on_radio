"""Step 2: transcribe Japanese audio with a local Whisper model -> Japanese SRT."""
import argparse
import math
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Cue, ROOT, env, fail, format_ts, load_env, write_srt  # noqa: E402

MAX_CUE_DURATION = 5.0
SHORT_CUE_DURATION = 1.0
SHORT_CUE_MAX_GAP = 0.3
MAX_COMPRESSION_RATIO = 2.4
HALLUCINATION_NO_SPEECH_THRESHOLD = 0.3
SHORT_CUE_NO_SPEECH_THRESHOLD = 0.6
SHORT_CUE_AVG_LOGPROB_THRESHOLD = -1.0
HALLUCINATION_PHRASES = (
    "ご視聴ありがとうございました",
    "チャンネル登録",
    "提供",
)


def _timestamp_seconds(value):
    hours, minutes, rest = value.split(":")
    seconds, millis = rest.split(",")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000


def _ends_sentence(text):
    return text.rstrip("」』”’\"')]}）】").endswith(("。", "！", "？", "!", "?"))


def _split_long_cue(cue, max_duration):
    """Split long Whisper segments proportionally when no word timings are available."""
    start = _timestamp_seconds(cue.start)
    end = _timestamp_seconds(cue.end)
    duration = end - start
    text = cue.text.strip()
    part_count = max(1, math.ceil(duration / max_duration))
    part_count = min(part_count, len(text))
    if part_count <= 1 or duration <= max_duration:
        return [cue]

    parts = []
    previous_cut = 0
    for part in range(1, part_count):
        target_time = min(part * max_duration, duration)
        target = round(len(text) * target_time / duration)
        remaining_parts = part_count - part
        cut = min(max(target, previous_cut + 1), len(text) - remaining_parts)
        parts.append(text[previous_cut:cut])
        previous_cut = cut
    parts.append(text[previous_cut:])

    result = []
    for part, part_text in enumerate(parts):
        part_start = start + min(part * max_duration, duration)
        part_end = start + min((part + 1) * max_duration, duration)
        result.append(Cue(
            cue.index, format_ts(part_start), format_ts(part_end), part_text,
            cue.avg_logprob,
        ))
    return result


def _prepare_transcription_segments(segments):
    """Filter suspicious Whisper segments and mark trustworthy short cues."""
    cues = []
    short_cue_indices = set()
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        if segment.compression_ratio > MAX_COMPRESSION_RATIO:
            continue
        if (segment.no_speech_prob > HALLUCINATION_NO_SPEECH_THRESHOLD
                and any(phrase in text for phrase in HALLUCINATION_PHRASES)):
            continue

        duration = segment.end - segment.start
        is_short = duration <= SHORT_CUE_DURATION
        if (is_short
                and segment.no_speech_prob > SHORT_CUE_NO_SPEECH_THRESHOLD
                and segment.avg_logprob < SHORT_CUE_AVG_LOGPROB_THRESHOLD):
            continue

        avg_logprob = getattr(segment, "avg_logprob", None)
        if avg_logprob is not None:
            avg_logprob = float(avg_logprob)
            if not math.isfinite(avg_logprob):
                avg_logprob = None
        cue = Cue(len(cues) + 1, format_ts(segment.start),
                  format_ts(segment.end), text, avg_logprob)
        cues.append(cue)
        if is_short:
            short_cue_indices.add(cue.index)
    return cues, short_cue_indices


def _merge_sentence_cues(cues, max_gap=1.0, max_duration=MAX_CUE_DURATION,
                         short_cue_indices=None):
    """Join sentence fragments, limiting each output cue to five seconds."""
    merged = []
    current_text = ""
    current_start = current_end = None
    current_index = 0
    current_has_short_cue = False
    current_logprobs = []
    if short_cue_indices is None:
        short_cue_indices = set()

    def flush():
        if current_text:
            avg_logprob = min(current_logprobs) if current_logprobs else None
            merged.append(Cue(current_index, current_start, current_end,
                              current_text, avg_logprob))

    split_cues = (
        (part, cue.index in short_cue_indices)
        for cue in cues
        for part in _split_long_cue(cue, max_duration)
    )
    for cue, is_short_cue in split_cues:
        if current_text:
            gap = _timestamp_seconds(cue.start) - _timestamp_seconds(current_end)
            duration = _timestamp_seconds(cue.end) - _timestamp_seconds(current_start)
            allowed_gap = max_gap
            if current_has_short_cue or is_short_cue:
                allowed_gap = min(max_gap, SHORT_CUE_MAX_GAP)
            if (gap > allowed_gap or _ends_sentence(current_text)
                    or duration > max_duration):
                flush()
                current_text = ""
                current_has_short_cue = False
                current_logprobs = []
        if not current_text:
            current_index = cue.index
            current_start = cue.start
            current_text = cue.text.strip()
            current_has_short_cue = is_short_cue
        else:
            separator = " " if current_text[-1:].isascii() and cue.text[:1].isascii() else ""
            current_text += separator + cue.text.strip()
            current_has_short_cue = current_has_short_cue or is_short_cue
        if cue.avg_logprob is not None:
            current_logprobs.append(cue.avg_logprob)
        current_end = cue.end
    flush()
    return merged


def main(argv=None):
    load_env()
    p = argparse.ArgumentParser(description="本地 Whisper 转录日语 -> 日文 SRT")
    p.add_argument("audio", help="WAV/音频文件（可用 01_get_audio.py 的输出）")
    p.add_argument("-o", "--output", help="输出 SRT，默认与音频同名 .ja.srt")
    p.add_argument("--model", default=env("WHISPER_MODEL", "large-v3-turbo"),
                   help="模型名或本地模型目录")
    p.add_argument("--device", default=env("WHISPER_DEVICE", "auto"))
    p.add_argument("--compute-type", default=env("WHISPER_COMPUTE_TYPE", "auto"))
    p.add_argument("--model-dir", default=env("WHISPER_MODEL_DIR", str(ROOT / "models")),
                   help="模型下载/缓存目录（本地）")
    p.add_argument("--initial-prompt", default=env("WHISPER_INITIAL_PROMPT", ""),
                   help="给模型的开场提示，可写入节目名、专有名词以纠正识别")
    p.add_argument("--hotwords", default=env("WHISPER_HOTWORDS", ""),
                   help="热词列表（逗号分隔），仅偏置解码，不占开场提示")
    p.add_argument("--condition-on-previous-text", type=str, default="auto",
                   choices=("auto", "on", "off"),
                   help="是否把前文传给后续解码；off 可减少重复幻觉（默认 auto）")
    p.add_argument("--hallucination-silence-threshold", type=float,
                   default=float(env("WHISPER_HALLUCINATION_SILENCE", "0")),
                   help="静音超过该秒数时跳过疑似幻觉片段，0 为关闭（默认 0）")
    p.add_argument("--vad-min-silence", type=int,
                   default=int(env("WHISPER_VAD_MIN_SILENCE", "500")),
                   help="VAD 切分的最小静音毫秒数（默认 500）")
    p.add_argument("--sentence-gap", type=float, default=1.0,
                   help="合并同一句字幕允许的最大片段间隔秒数（默认 1.0）")
    p.add_argument("--max-duration", type=float,
                   help="仅转录音频开头指定秒数；用于短样本评估")
    args = p.parse_args(argv)
    if args.sentence_gap < 0:
        fail("--sentence-gap 不能小于 0")
    if args.max_duration is not None and (
            not math.isfinite(args.max_duration) or args.max_duration <= 0):
        fail("--max-duration 必须是有限且大于 0 的秒数")

    audio = Path(args.audio)
    if not audio.is_file():
        fail(f"找不到音频: {audio}")
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        fail("未安装 faster-whisper，请运行 pip install -r requirements.txt")

    def run(device, compute_type):
        print(f"加载模型 {args.model} ({device}/{compute_type}) ...", file=sys.stderr)
        model = WhisperModel(args.model, device=device, compute_type=compute_type,
                             download_root=args.model_dir)
        options = dict(
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": args.vad_min_silence},
            log_progress=args.max_duration is None,
            beam_size=5,
        )
        if args.initial_prompt:
            options["initial_prompt"] = args.initial_prompt
        if args.hotwords:
            options["hotwords"] = args.hotwords
        # 广播常穿插音乐/音效，前文条件易把噪声带成重复幻觉；off 更稳，auto 保持默认
        if args.condition_on_previous_text == "off":
            options["condition_on_previous_text"] = False
        if args.hallucination_silence_threshold > 0:
            options["hallucination_silence_threshold"] = args.hallucination_silence_threshold
        if args.max_duration is not None:
            options["clip_timestamps"] = [0, args.max_duration]
        segments, info = model.transcribe(str(audio), language="ja", **options)
        transcription_segments = []
        progress_duration = min(info.duration, args.max_duration) \
            if args.max_duration is not None else info.duration
        with tqdm(total=progress_duration, desc="Whisper 转录", unit="s",
                  file=sys.stderr) as progress:
            for s in segments:
                transcription_segments.append(s)
                progress_end = min(s.end, progress_duration)
                if progress_end > progress.n:
                    progress.update(progress_end - progress.n)
                progress.set_postfix_str(f"{len(transcription_segments)} 个片段")
            if progress_duration and progress.n < progress_duration:
                progress.update(progress_duration - progress.n)
        cues, short_cue_indices = _prepare_transcription_segments(
            transcription_segments)
        return _merge_sentence_cues(
            cues, max_gap=args.sentence_gap,
            short_cue_indices=short_cue_indices)

    try:
        gpu_type = "float16" if args.compute_type == "auto" else args.compute_type
        cues = run(args.device, gpu_type if args.device != "cpu" else args.compute_type)
    except RuntimeError as e:
        if args.device == "cpu":
            raise
        print(f"GPU 推理失败({e})，回退到 CPU", file=sys.stderr)
        cues = run("cpu", "int8" if args.compute_type == "auto" else args.compute_type)
    if not cues:
        fail("清洗后没有可输出的字幕")
    out = Path(args.output) if args.output else audio.with_suffix(".ja.srt")
    write_srt(cues, out)
    print(out)


if __name__ == "__main__":
    main()
