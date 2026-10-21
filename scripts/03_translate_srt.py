"""Step 3: Ollama Hy-MT translation of Japanese SRT -> Chinese draft SRT."""
import argparse
import json
import sys
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (Cue, complete_numbered, complete_numbers, env, fail, load_env,
                    numbered, parse_numbered, parse_srt, write_srt)  # noqa: E402

_numbered = numbered
_parse_numbered = parse_numbered
_complete_numbered = complete_numbered

_TRANSLATION_SYSTEM_PROMPT = (
    "你是专业日语到简体中文的字幕翻译员。只输出译文，不要解释或复述原文；"
    "保留原意、语气和术语，译文自然、简洁，适合字幕阅读。"
    "严格保留输入编号，每个编号只输出一行，编号和译文之间用一个真正的制表符分隔。"
    "例如：1\t你好"
)


def _context_groups(items, size=3):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _numbered_subset(items, numbers, start_number=1):
    return "\n".join(
        f"{i}\t{items[i - start_number].replace(chr(10), ' ')}"
        for i in numbers
    )


def ollama_translate(batch, host, model, start_number=1, terms=None,
                     reference_text=""):
    translations = []
    group_count = (len(batch) + 39) // 40
    groups = _context_groups(batch, size=40)
    system_prompt = _TRANSLATION_SYSTEM_PROMPT
    if reference_text:
        system_prompt += (
            "\n节目、人物及背景参考资料仅用于理解原文与统一专名译法，"
            "不得将参考资料内容当作待翻译字幕，也不要扩写字幕：\n"
            + reference_text.strip()
        )
    for group_index, group in enumerate(tqdm(
            groups, total=group_count, desc="混元翻译", unit="组",
            leave=False, file=sys.stderr)):
        first_number = start_number + group_index * 40
        numbers = list(range(first_number, first_number + len(group)))

        def request(numbers):
            source = _numbered_subset(group, numbers, first_number)
            term_note = (
                "\n以下为固定术语译法，请优先遵循：\n"
                + json.dumps(terms, ensure_ascii=False, separators=(",", ":"))
                if terms else ""
            )
            user_prompt = (
                "将以下内容翻译为简体中文：\n"
                f"只处理编号 {', '.join(map(str, numbers))}，并严格按编号逐行返回。"
                + term_note + "\n待翻译字幕：\n" + source
            )
            r = requests.post(f"{host}/api/chat", json={
                "model": model, "stream": False, "think": False,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "options": {
                    "top_k": 20,
                    "top_p": 0.6,
                    "temperature": 0.7,
                    "repeat_penalty": 1.05,
                }}, timeout=600)
            r.raise_for_status()
            return r.json()["message"]["content"]

        translations.extend(complete_numbers(numbers, request, "翻译"))
    return translations


def main(argv=None):
    load_env()
    p = argparse.ArgumentParser(description="日文 SRT -> 中文初译 SRT（Ollama 翻译）")
    p.add_argument("srt", help="日文 SRT")
    p.add_argument("-o", "--output", help="输出，默认 <名称>.zh.srt")
    p.add_argument("--batch-size", type=int, default=int(env("BATCH_SIZE", "50")),
                   help="混元初译每次提交的字幕条数")
    p.add_argument("--ollama-host", default=env("OLLAMA_HOST_URL", "http://127.0.0.1:11434"))
    p.add_argument("--ollama-model", default=env("HY_MT_MODEL", "hy-mt1.5:7b"))
    p.add_argument("--terms-file",
                   help="术语对照 JSON 文件，例如 {\"节目名\":\"节目名称\"}")
    p.add_argument("--reference-file",
                   help="供 Hy-MT 参考的 UTF-8 Markdown 节目/人物资料")
    args = p.parse_args(argv)
    if args.batch_size < 1:
        fail("--batch-size 必须大于 0")

    src = Path(args.srt)
    if not src.is_file():
        fail(f"找不到 SRT: {src}")
    cues = parse_srt(src.read_text(encoding="utf-8"))
    if not cues:
        fail("SRT 中没有字幕")

    terms = {}
    if args.terms_file:
        terms_path = Path(args.terms_file)
        if not terms_path.is_file():
            fail(f"找不到术语表文件: {terms_path}")
        try:
            terms = json.loads(terms_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as e:
            fail(f"术语表读取失败: {e}")
        if not isinstance(terms, dict) or any(
                not isinstance(source, str) or not source.strip()
                or not isinstance(target, str) or not target.strip()
                for source, target in terms.items()):
            fail("术语表必须是“原文字符串: 译文字符串”的 JSON 对象")

    reference_text = ""
    if args.reference_file:
        reference_path = Path(args.reference_file)
        if not reference_path.is_file():
            fail(f"找不到 Hy-MT 参考文档: {reference_path}")
        try:
            reference_text = reference_path.read_text(encoding="utf-8-sig").strip()
        except (OSError, UnicodeError) as e:
            fail(f"Hy-MT 参考文档读取失败: {e}")
        if not reference_text:
            fail(f"Hy-MT 参考文档为空: {reference_path}")

    translations = []
    n = args.batch_size
    try:
        with tqdm(total=len(cues), desc="混元翻译", unit="条",
                  file=sys.stderr) as progress:
            for s in range(0, len(cues), n):
                ja = [c.text for c in cues[s:s + n]]
                zh = ollama_translate(
                    ja, args.ollama_host, args.ollama_model,
                    start_number=s + 1, terms=terms,
                    reference_text=reference_text)
                translations.extend(zh)
                progress.update(len(zh))
    except (requests.RequestException, ValueError, KeyError, json.JSONDecodeError) as e:
        fail(f"翻译失败: {e}")

    if args.output:
        out = Path(args.output)
    else:
        stem = src.name[:-len(".ja.srt")] if src.name.endswith(".ja.srt") else src.stem
        out = src.with_name(stem + ".zh.srt")
    write_srt([Cue(c.index, c.start, c.end, t) for c, t in zip(cues, translations)], out)
    print(out)


if __name__ == "__main__":
    main()
