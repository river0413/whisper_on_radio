"""Qwen3-ASR and ForcedAligner helpers for the transcription pipeline."""
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from tqdm import tqdm

from common import Cue, fail, format_ts


class QwenDependencyError(ImportError):
    pass


class Qwen3Runner:
    def __init__(self, args):
        self.args = args
        self.model = None

    def _load_model(self):
        if self.model is not None:
            return self.model
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise QwenDependencyError(
                "未安装 Qwen3 ASR 依赖，请运行 "
                "pip install -r requirements-qwen-asr.txt"
            ) from exc

        device, dtype = _qwen_device_and_dtype(
            torch, self.args.device, self.args.qwen_dtype)
        print(
            f"加载 Qwen ASR {self.args.qwen_asr_model} 和对齐器 "
            f"{self.args.qwen_aligner_model} ({device}/{self.args.qwen_dtype}) ...",
            file=sys.stderr,
        )
        self.model = Qwen3ASRModel.from_pretrained(
            self.args.qwen_asr_model,
            dtype=dtype,
            device_map=device,
            max_inference_batch_size=1,
            max_new_tokens=self.args.qwen_max_new_tokens,
            forced_aligner=self.args.qwen_aligner_model,
            forced_aligner_kwargs={"dtype": dtype, "device_map": device},
        )
        return self.model

    def _context(self):
        return " ".join(
            value.strip()
            for value in (self.args.initial_prompt, self.args.hotwords)
            if value.strip()
        )

    def _transcribe(self, audio):
        options = {
            "audio": str(audio),
            "language": "Japanese",
            "return_time_stamps": True,
        }
        context = self._context()
        if context:
            options["context"] = context
        results = self._load_model().transcribe(**options)
        if not results:
            raise ValueError("Qwen ASR 没有返回结果")
        return results[0]

    def assist_low_confidence(self, audio, segments, threshold):
        candidates = [
            (index, segment)
            for index, segment in enumerate(segments)
            if _is_low_confidence(segment, threshold)
        ]
        if not candidates:
            return segments

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            _warn("未找到 ffmpeg，保留 Whisper 原文")
            return segments
        try:
            self._load_model()
        except (ImportError, OSError, RuntimeError, ValueError, TypeError,
                AttributeError) as exc:
            _warn(f"模型加载失败，保留 Whisper 原文：{exc}")
            return segments

        result_segments = list(segments)
        with tempfile.TemporaryDirectory(prefix="qwen3_assist_") as temp_dir:
            for index, segment in candidates:
                try:
                    replacement = self._assist_segment(
                        audio, segment, ffmpeg, Path(temp_dir), index + 1,
                    )
                except (ImportError, OSError, RuntimeError, ValueError,
                        TypeError, AttributeError,
                        subprocess.SubprocessError) as exc:
                    _warn(f"片段 {index + 1} 重转录失败，保留 Whisper 原文：{exc}")
                    continue
                if replacement:
                    result_segments[index:index + 1] = replacement
                    print(
                        f"Qwen 已辅助重转录 Whisper 低置信度片段 {index + 1}",
                        file=sys.stderr,
                    )
        return result_segments

    def _assist_segment(self, audio, segment, ffmpeg, temp_dir, number):
        start = float(segment.start)
        end = float(segment.end)
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise ValueError(f"Whisper 片段时间范围无效：{start}–{end}")
        padding = 0.25
        crop_start = max(0.0, start - padding)
        crop_end = end + padding
        clip_path = temp_dir / f"low-confidence-{number:04d}.wav"
        command = [
            ffmpeg, "-v", "error", "-y", "-ss", f"{crop_start:.3f}",
            "-i", str(audio), "-t", f"{crop_end - crop_start:.3f}",
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(clip_path),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        result = self._transcribe(clip_path)
        text = str(getattr(result, "text", "") or "").strip()
        if not text:
            raise ValueError("Qwen ASR 返回空文本")

        return [SimpleNamespace(
            start=start,
            end=end,
            text=text,
            avg_logprob=getattr(segment, "avg_logprob", None),
            no_speech_prob=0.0,
            compression_ratio=1.0,
        )]


def _warn(message):
    print(f"警告：Qwen ASR 辅助不可用，{message}", file=sys.stderr)


def _is_low_confidence(segment, threshold):
    score = getattr(segment, "avg_logprob", None)
    if score is None:
        return False
    try:
        score = float(score)
    except (TypeError, ValueError):
        return False
    return math.isfinite(score) and score <= threshold


def _qwen_device_and_dtype(torch, requested_device, requested_dtype):
    device = requested_device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif device == "cuda":
        device = "cuda:0"

    if requested_dtype == "auto":
        if device.startswith("cuda"):
            dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported()
                     else torch.float16)
        else:
            dtype = torch.float32
    else:
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[requested_dtype]
    return device, dtype


def _probe_audio_duration(audio):
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        fail("Qwen3 后端需要 FFmpeg 的 ffprobe，请先安装 FFmpeg 并加入 PATH")
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(audio)],
        check=True, capture_output=True, text=True,
    )
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        fail(f"无法读取音频时长: {audio}")
    if not math.isfinite(duration) or duration <= 0:
        fail(f"音频时长无效: {audio}")
    return duration


def transcribe_full(args, audio, merge_sentence_cues):
    runner = Qwen3Runner(args)
    try:
        runner._load_model()
    except QwenDependencyError as exc:
        fail(f"{exc}；原因: {exc.__cause__}")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        fail("Qwen3 后端需要 FFmpeg，请先安装 FFmpeg 并加入 PATH")
    source_duration = _probe_audio_duration(audio)
    duration = min(source_duration, args.max_duration) if args.max_duration else source_duration
    chunk_seconds = 240.0
    overlap_seconds = 2.0
    stride = chunk_seconds - overlap_seconds
    starts = [0.0]
    while starts[-1] + chunk_seconds < duration:
        starts.append(starts[-1] + stride)

    aligned_cues = []
    with tempfile.TemporaryDirectory(prefix="qwen3_asr_") as temp_dir:
        with tqdm(total=duration, desc="Qwen3-ASR+ForcedAligner",
                  unit="s", file=sys.stderr) as progress:
            for chunk_index, chunk_start in enumerate(starts):
                chunk_end = min(chunk_start + chunk_seconds, duration)
                chunk_length = chunk_end - chunk_start
                chunk_path = Path(temp_dir) / f"chunk-{chunk_index:04d}.wav"
                if (len(starts) == 1 and duration == source_duration
                        and not args.max_duration):
                    chunk_path = audio
                else:
                    command = [
                        ffmpeg, "-v", "error", "-y", "-ss", f"{chunk_start:.3f}",
                        "-i", str(audio), "-t", f"{chunk_length:.3f}",
                        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                        str(chunk_path),
                    ]
                    try:
                        subprocess.run(command, check=True, capture_output=True, text=True)
                    except subprocess.CalledProcessError as exc:
                        fail(f"切分 Qwen 音频失败: {exc.stderr.strip()}")
                try:
                    result = runner._transcribe(chunk_path)
                except (ImportError, OSError, RuntimeError, ValueError) as exc:
                    fail(f"Qwen ASR 识别失败（音频片段 {chunk_index + 1}）：{exc}")
                if result.time_stamps is None:
                    fail(f"Qwen forced aligner 未返回时间戳（音频片段 {chunk_index + 1}）")

                is_first = chunk_index == 0
                is_last = chunk_index == len(starts) - 1
                for stamp in result.time_stamps:
                    text = str(stamp.text).strip()
                    start = float(stamp.start_time)
                    end = float(stamp.end_time)
                    if not text:
                        continue
                    if (not math.isfinite(start) or not math.isfinite(end)
                            or start < 0 or end < start or end > chunk_length + 1):
                        fail(f"Qwen forced aligner 返回无效时间戳: {start}–{end}")
                    if not is_first and start < overlap_seconds / 2:
                        continue
                    if not is_last and end > chunk_seconds - overlap_seconds / 2:
                        continue
                    aligned_cues.append(Cue(
                        len(aligned_cues) + 1,
                        format_ts(chunk_start + start),
                        format_ts(chunk_start + end),
                        text,
                    ))
                progress.update(chunk_length if chunk_index == 0
                                else stride if not is_last
                                else chunk_length - overlap_seconds)
    return merge_sentence_cues(aligned_cues, max_gap=args.sentence_gap)


def assist_low_confidence(args, audio, segments, threshold):
    return Qwen3Runner(args).assist_low_confidence(audio, segments, threshold)
