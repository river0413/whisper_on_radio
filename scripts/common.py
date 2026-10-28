"""Shared helpers: .env loading, SRT parsing/writing."""
import os
import re
import sys
from dataclasses import dataclass
from math import ceil, isfinite
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_env(path=None):
    """Load KEY=VALUE lines from .env into os.environ (existing vars win)."""
    path = Path(path) if path else ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def env(name, default=None):
    return os.environ.get(name, default)


def estimate_tokens(text):
    """Approximate tokens: one per CJK character, four ASCII word chars per token."""
    tokens = 0
    ascii_run = 0
    for char in text:
        codepoint = ord(char)
        if (0x3040 <= codepoint <= 0x30FF
                or 0x3400 <= codepoint <= 0x9FFF
                or 0xF900 <= codepoint <= 0xFAFF):
            tokens += ceil(ascii_run / 4)
            ascii_run = 0
            tokens += 1
        elif char.isascii() and (char.isalnum() or char == "_"):
            ascii_run += 1
        elif char.isspace():
            tokens += ceil(ascii_run / 4)
            ascii_run = 0
        else:
            tokens += ceil(ascii_run / 4)
            ascii_run = 0
            tokens += 1
    return tokens + ceil(ascii_run / 4)


def fail(msg, code=1):
    print(f"错误: {msg}", file=sys.stderr)
    sys.exit(code)


@dataclass
class Cue:
    index: int
    start: str
    end: str
    text: str
    avg_logprob: float | None = None


def format_ts(seconds):
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


_TS = re.compile(r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})")
_AVG_LOGPROB_SUFFIX = re.compile(
    r"[ \t]*<(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)->[ \t]*$"
)


def parse_srt(text):
    cues = []
    text = text.lstrip("\ufeff").replace("\r\n", "\n")
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.split("\n")
        for i, line in enumerate(lines):
            m = _TS.search(line)
            if m:
                body = "\n".join(lines[i + 1:]).strip()
                avg_logprob = None
                body_lines = body.splitlines()
                if body_lines:
                    confidence_match = _AVG_LOGPROB_SUFFIX.search(body_lines[-1])
                    if confidence_match:
                        candidate = float(confidence_match.group(1))
                        if isfinite(candidate):
                            avg_logprob = candidate
                            body_lines[-1] = body_lines[-1][:confidence_match.start()].rstrip()
                            body = "\n".join(body_lines).strip()
                cues.append(Cue(len(cues) + 1, m.group(1).replace(".", ","),
                                m.group(2).replace(".", ","), body, avg_logprob))
                break
    return cues


def write_srt(cues, path, include_confidence=True):
    out = []
    for i, c in enumerate(cues, 1):
        text = c.text
        if include_confidence and c.avg_logprob is not None:
            lines = text.splitlines()
            if lines:
                lines[-1] = f"{lines[-1]} <{c.avg_logprob!r}->"
                text = "\n".join(lines)
        out.append(f"{i}\n{c.start} --> {c.end}\n{text}\n")
    Path(path).write_text("\n".join(out), encoding="utf-8")


def merge_bilingual(cues, translations):
    """Each cue: Japanese line followed by Chinese line."""
    if len(cues) != len(translations):
        raise ValueError("字幕与译文数量不一致")
    return [
        Cue(c.index, c.start, c.end, f"{c.text}\n{t}", c.avg_logprob)
        for c, t in zip(cues, translations)
    ]


def numbered(items):
    return "\n".join(f"{i}\t{t.replace(chr(10), ' ')}" for i, t in enumerate(items, 1))


_NUMBERED_LINE = re.compile(r"^\s*(\d+)\s*(?:[\t.、:：)）\]|-]+)?\s*(.*\S)\s*$")


def parse_numbered_map(text):
    result = {}
    for line in text.strip().splitlines():
        m = _NUMBERED_LINE.match(line)
        if m:
            result.setdefault(int(m.group(1)), m.group(2))
    return result


def parse_numbered(text, expected):
    result = parse_numbered_map(text)
    if any(i not in result for i in range(1, expected + 1)):
        raise ValueError(f"返回行数不匹配: 期望 {expected}, 得到 {len(result)}")
    return [result[i] for i in range(1, expected + 1)]


def complete_numbers(numbers, request, operation):
    """Request the given subtitle numbers and retry only the ones left out."""
    numbers = list(numbers)
    result = parse_numbered_map(request(numbers))
    missing = [i for i in numbers if i not in result]
    if missing:
        repaired = parse_numbered_map(request(missing))
        for index in missing:
            if index in repaired:
                result[index] = repaired[index]
    missing = [i for i in numbers if i not in result or not result[i].strip()]
    if missing:
        raise ValueError(
            f"{operation}仍缺少字幕编号 {', '.join(map(str, missing))}，请检查模型输出"
        )
    return [result[i].strip() for i in numbers]


def complete_numbered(items, request, operation):
    """Retry only omitted rows while retaining their neighboring context."""
    return complete_numbers(range(1, len(items) + 1), request, operation)
