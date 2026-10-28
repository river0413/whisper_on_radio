"""Step 3: Ollama Hy-MT translation of Japanese SRT -> Chinese draft SRT."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (Cue, complete_numbered, complete_numbers, env, fail, load_env,
                    estimate_tokens, numbered, parse_numbered, parse_srt,
                    write_srt)  # noqa: E402

_numbered = numbered
_parse_numbered = parse_numbered
_complete_numbered = complete_numbered

_DEFAULT_NUM_CTX = 1800
_DEFAULT_MAX_INPUT_TOKENS = 900
_MAX_NUM_CTX = 2047
_OLLAMA_STARTUP_TIMEOUT = 60
_MAX_TRANSLATION_GROUP = 40
_OUTPUT_CONTEXT_RESERVE = 64

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


def _translation_user_prompt(items, numbers, start_number, terms):
    source = _numbered_subset(items, numbers, start_number)
    term_note = (
        "\n以下为固定术语译法，请优先遵循：\n"
        + json.dumps(terms, ensure_ascii=False, separators=(",", ":"))
        if terms else ""
    )
    return (
        "将以下内容翻译为简体中文：\n"
        f"只处理编号 {', '.join(map(str, numbers))}，并严格按编号逐行返回。"
        + term_note + "\n待翻译字幕：\n" + source
    )


def _translation_request_estimate(items, numbers, start_number, terms,
                                  system_prompt):
    user_prompt = _translation_user_prompt(items, numbers, start_number, terms)
    input_tokens = estimate_tokens(system_prompt) + estimate_tokens(user_prompt)
    output_tokens = estimate_tokens(_numbered_subset(items, numbers, start_number))
    context_tokens = input_tokens + output_tokens + _OUTPUT_CONTEXT_RESERVE
    return user_prompt, input_tokens, context_tokens


def _token_limited_groups(items, start_number, terms, system_prompt,
                          max_input_tokens, max_context_tokens):
    groups = []
    current = []
    current_start = start_number

    def fits(group, first_number):
        numbers = list(range(first_number, first_number + len(group)))
        _, input_tokens, context_tokens = _translation_request_estimate(
            group, numbers, first_number, terms, system_prompt)
        return input_tokens <= max_input_tokens and context_tokens <= max_context_tokens

    def check_singleton(item, number):
        if not fits([item], number):
            numbers = [number]
            _, input_tokens, context_tokens = _translation_request_estimate(
                [item], numbers, number, terms, system_prompt)
            raise ValueError(
                f"字幕编号 {number} 单条请求的估算输入为 {input_tokens} tokens、"
                f"含输出预留的上下文为 {context_tokens} tokens，超过限制 "
                f"（输入 {max_input_tokens}、上下文 {max_context_tokens}）；"
                "请缩短术语表/参考资料或调整 token 限额"
            )

    for offset, item in enumerate(items):
        number = start_number + offset
        if len(current) >= _MAX_TRANSLATION_GROUP:
            groups.append((current_start, current))
            current = []
            current_start = number

        candidate = current + [item]
        if fits(candidate, current_start):
            current = candidate
            continue

        if current:
            groups.append((current_start, current))
            current = []
            current_start = number
        check_singleton(item, number)
        current = [item]
        current_start = number

    if current:
        groups.append((current_start, current))
    return groups


def ensure_ollama(host):
    host = host.rstrip("/")
    version_url = f"{host}/api/version"
    try:
        response = requests.get(version_url, timeout=3)
        response.raise_for_status()
        return
    except requests.RequestException as initial_error:
        parsed_host = urlparse(host)
        if parsed_host.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise RuntimeError(f"Ollama 服务不可用，无法自动启动远程服务: {host}") from initial_error

    ollama = shutil.which("ollama")
    if not ollama:
        raise RuntimeError("Ollama 服务未运行，且找不到 ollama 命令；请安装 Ollama 并加入 PATH")

    print("Ollama 服务未运行，正在启动...", file=sys.stderr)
    process_env = os.environ.copy()
    process_env["OLLAMA_HOST"] = parsed_host.netloc
    try:
        subprocess.Popen(
            [ollama, "serve"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=process_env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError as error:
        raise RuntimeError(f"启动 Ollama 失败: {error}") from error

    deadline = time.monotonic() + _OLLAMA_STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        try:
            response = requests.get(version_url, timeout=3)
            response.raise_for_status()
            return
        except requests.RequestException:
            time.sleep(1)
    raise RuntimeError(f"Ollama 服务启动超时（{_OLLAMA_STARTUP_TIMEOUT} 秒）: {host}")


def ollama_translate(batch, host, model, start_number=1, terms=None,
                     reference_text="", num_ctx=_DEFAULT_NUM_CTX,
                     max_input_tokens=_DEFAULT_MAX_INPUT_TOKENS):
    translations = []
    system_prompt = _TRANSLATION_SYSTEM_PROMPT
    if reference_text:
        system_prompt += (
            "\n节目、人物及背景参考资料仅用于理解原文与统一专名译法，"
            "不得将参考资料内容当作待翻译字幕，也不要扩写字幕：\n"
            + reference_text.strip()
        )
    groups = _token_limited_groups(
        batch, start_number, terms, system_prompt,
        max_input_tokens, num_ctx)
    for first_number, group in tqdm(
            groups, total=len(groups), desc="混元翻译", unit="组",
            leave=False, file=sys.stderr):
        numbers = list(range(first_number, first_number + len(group)))

        def request(numbers):
            user_prompt, _, context_tokens = _translation_request_estimate(
                group, numbers, first_number, terms, system_prompt)
            return make_request(numbers, user_prompt, context_tokens)

        def make_request(numbers, user_prompt, context_tokens):
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
                    "num_ctx": context_tokens,
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
    p.add_argument("--num-ctx", type=int,
                   default=env("OLLAMA_NUM_CTX", str(_DEFAULT_NUM_CTX)),
                   help=f"Ollama 单次上下文 token 上限（1-{_MAX_NUM_CTX}）")
    p.add_argument("--max-input-tokens", type=int,
                   default=env("OLLAMA_MAX_INPUT_TOKENS",
                               str(_DEFAULT_MAX_INPUT_TOKENS)),
                   help="每次请求输入 token 估算上限")
    p.add_argument("--terms-file",
                   help="术语对照 JSON 文件，例如 {\"节目名\":\"节目名称\"}")
    p.add_argument("--reference-file",
                   help="供 Hy-MT 参考的 UTF-8 Markdown 节目/人物资料")
    args = p.parse_args(argv)
    if args.batch_size < 1:
        fail("--batch-size 必须大于 0")
    if not 1 <= args.num_ctx <= _MAX_NUM_CTX:
        fail(f"--num-ctx 必须在 1 到 {_MAX_NUM_CTX} 之间")
    if not 1 <= args.max_input_tokens <= _MAX_NUM_CTX:
        fail(f"--max-input-tokens 必须在 1 到 {_MAX_NUM_CTX} 之间")
    if args.max_input_tokens > args.num_ctx:
        fail("--max-input-tokens 不能大于 --num-ctx")

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
        ensure_ollama(args.ollama_host)
        with tqdm(total=len(cues), desc="混元翻译", unit="条",
                  file=sys.stderr) as progress:
            for s in range(0, len(cues), n):
                ja = [c.text for c in cues[s:s + n]]
                zh = ollama_translate(
                    ja, args.ollama_host, args.ollama_model,
                    start_number=s + 1, terms=terms,
                    reference_text=reference_text, num_ctx=args.num_ctx,
                    max_input_tokens=args.max_input_tokens)
                translations.extend(zh)
                progress.update(len(zh))
    except (requests.RequestException, ValueError, KeyError,
            json.JSONDecodeError, RuntimeError) as e:
        fail(f"翻译失败: {e}")

    if args.output:
        out = Path(args.output)
    else:
        stem = src.name[:-len(".ja.srt")] if src.name.endswith(".ja.srt") else src.stem
        out = src.with_name(stem + ".zh.srt")
    write_srt([
        Cue(c.index, c.start, c.end, t, c.avg_logprob)
        for c, t in zip(cues, translations)
    ], out)
    print(out)


if __name__ == "__main__":
    main()
