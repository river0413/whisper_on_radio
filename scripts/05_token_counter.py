"""Estimate prompt and subtitle-response tokens without calling an API."""
import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import fail, parse_srt  # noqa: E402

_SCRIPTS = Path(__file__).resolve().parent
_TRANSLATION_SYSTEM_PROMPT = (
    "你是日语广播字幕翻译员。将提供的日文字幕翻译成自然、准确、适合字幕阅读的简体中文，"
    "并在所提供条目范围内统一人物称谓、专有名词和语气。"
    "严格按用户消息指定的编号逐条输出，不合并、不遗漏、不改动编号；"
    "每行编号和译文之间用一个真正的制表符分隔，例如：1\t你好，不要附加解释。"
)


def _load_proofread_module():
    name = "proofread_srt_for_token_estimation"
    spec = importlib.util.spec_from_file_location(
        name, _SCRIPTS / "04_proofread_srt.py")
    if spec is None or spec.loader is None:
        raise ImportError("无法加载字幕校对提示词")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_PROOFREAD = _load_proofread_module()


def estimate_tokens(text):
    """Approximate tokens: one per CJK character, four ASCII word chars per token."""
    tokens = 0
    ascii_run = 0
    for char in text:
        codepoint = ord(char)
        if (0x3040 <= codepoint <= 0x30FF
                or 0x3400 <= codepoint <= 0x9FFF
                or 0xF900 <= codepoint <= 0xFAFF):
            tokens += math.ceil(ascii_run / 4)
            ascii_run = 0
            tokens += 1
        elif char.isascii() and (char.isalnum() or char == "_"):
            ascii_run += 1
        elif char.isspace():
            tokens += math.ceil(ascii_run / 4)
            ascii_run = 0
        elif not char.isspace():
            tokens += math.ceil(ascii_run / 4)
            ascii_run = 0
            tokens += 1
    return tokens + math.ceil(ascii_run / 4)


def _numbered_subset(items, numbers):
    return "\n".join(
        f"{number}\t{items[number - 1].replace(chr(10), ' ')}"
        for number in numbers
    )


def _translation_prompt(items, numbers, custom_prompt):
    user_prompt = (
        ("\n用户自定义要求：\n" + custom_prompt.strip()
         if custom_prompt.strip() else "")
        + f"\n只翻译以下带编号的日文字幕，编号为：{', '.join(map(str, numbers))}。\n"
        "待翻译字幕：\n" + _numbered_subset(items, numbers)
    )
    return _TRANSLATION_SYSTEM_PROMPT, user_prompt


def _proofread_prompt(japanese, chinese, cues, romanized, numbers,
                      custom_prompt, terms, reference):
    reference_text = custom_prompt + _PROOFREAD._reference_section(terms, reference)
    user_prompt = (
        ("\n用户自定义校对要求：\n" + reference_text.strip()
         if reference_text.strip() else "")
        + f"\n只校对以下带编号的条目，编号为：{', '.join(map(str, numbers))}。\n"
        + "每行输入依次为编号、日文原文、日文罗马音、中文初译，"
        "列之间以制表符分隔；罗马音仅作读音参考，不需输出。\n"
        + "先结合下面的 Markdown 对照表进行人名/地名/机构名与可能误摘抄核对，"
        "再按系统格式输出三列结果。\n"
        + "按系统规定输出三列；需要纠正日文识别错误时修正日文列；"
        "合并相邻条目时用编号范围，并先确认合并后总时长不超过 8 秒；"
        "确认是幻听或无效噪声时按规定删除。\n"
        + "待校对条目（TSV，末列为该条时长）：\n"
        + _PROOFREAD._numbered_pair_subset(
            japanese, chinese, numbers, cues, romanized)
        + "\n\n校对参考对照表（Markdown）：\n"
        + _PROOFREAD._numbered_pair_markdown_table(
            japanese, chinese, numbers, cues, romanized)
    )
    return _PROOFREAD._PROOFREAD_SYSTEM_PROMPT, user_prompt


def estimate_job(cues, drafts=None, batch_size=200, custom_prompt="",
                 terms=None, reference=""):
    if not cues:
        raise ValueError("SRT 中没有字幕")
    if batch_size < 1:
        raise ValueError("每批字幕条数必须大于 0")
    if drafts is not None and len(cues) != len(drafts):
        raise ValueError("中文初译条数与日文字幕条数不一致")

    japanese = [cue.text for cue in cues]
    romanized = (
        [_PROOFREAD._romanize_japanese(text) for text in japanese]
        if drafts is not None else None
    )
    input_tokens = 0
    output_tokens = 0
    request_count = 0
    for numbers in _PROOFREAD._proofread_slices(len(cues), batch_size):
        if drafts is None:
            system_prompt, user_prompt = _translation_prompt(
                japanese, numbers, custom_prompt)
            expected_output = "\n".join(
                f"{number}\t{japanese[number - 1]}" for number in numbers)
        else:
            system_prompt, user_prompt = _proofread_prompt(
                japanese, drafts, cues, romanized, numbers,
                custom_prompt, terms or {}, reference)
            expected_output = "\n".join(
                f"{number}\t{japanese[number - 1]}\t{drafts[number - 1]}"
                for number in numbers)
        input_tokens += estimate_tokens(system_prompt) + estimate_tokens(user_prompt)
        output_tokens += estimate_tokens(expected_output)
        request_count += 1
    return {
        "mode": "proofread" if drafts is not None else "translate",
        "requests": request_count,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def _read_srt(path, description):
    file_path = Path(path)
    if not file_path.is_file():
        fail(f"找不到{description}: {file_path}")
    try:
        cues = parse_srt(file_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as error:
        fail(f"{description}读取失败: {error}")
    if not cues:
        fail(f"{description} 中没有字幕")
    return cues


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="独立估算字幕翻译/校对的输入与输出 token（不调用 API）")
    parser.add_argument("srt", help="日文 SRT")
    parser.add_argument("--zh-srt", help="中文初译 SRT；指定后估算校对")
    parser.add_argument("--proofread-lines", type=int, default=200,
                        help="每批字幕条数（默认 200）")
    parser.add_argument("--prompt-file", help="额外提示词 Markdown 文件（UTF-8）")
    parser.add_argument("--terms-file", help="术语对照 JSON 文件")
    parser.add_argument("--reference-file", help="背景参考 Markdown 文件（UTF-8）")
    args = parser.parse_args(argv)
    if args.proofread_lines < 1:
        fail("--proofread-lines 必须大于 0")

    cues = _read_srt(args.srt, "日文 SRT")
    drafts = None
    if args.zh_srt:
        drafts = _read_srt(args.zh_srt, "中文初译 SRT")
        if len(drafts) != len(cues):
            fail(f"中文初译条数 ({len(drafts)}) 与日文字幕条数 ({len(cues)}) 不一致")

    def read_text_file(path, description):
        if not path:
            return ""
        file_path = Path(path)
        if not file_path.is_file():
            fail(f"找不到{description}: {file_path}")
        try:
            return file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            fail(f"{description}读取失败: {error}")

    custom_prompt = read_text_file(args.prompt_file, "提示词文件").strip()
    reference = read_text_file(args.reference_file, "背景参考文件")
    terms = {}
    terms_text = read_text_file(args.terms_file, "术语表文件")
    if terms_text:
        try:
            terms = json.loads(terms_text)
        except json.JSONDecodeError as error:
            fail(f"术语表读取失败: {error}")
        if not isinstance(terms, dict) or any(
                not isinstance(source, str) or not source.strip()
                or not isinstance(target, str) or not target.strip()
                for source, target in terms.items()):
            fail("术语表必须是“原文字符串: 译文字符串”的 JSON 对象")

    try:
        estimate = estimate_job(
            cues, drafts, args.proofread_lines, custom_prompt, terms, reference)
    except (ValueError, ImportError) as error:
        fail(f"token 估算失败: {error}")

    mode_text = "校对" if estimate["mode"] == "proofread" else "翻译"
    print(f"模式：{mode_text}；字幕条数：{len(cues)}；预计请求数：{estimate['requests']}")
    print(f"输入 token 估算： {estimate['input_tokens']}")
    print(f"输出 token 估算： {estimate['output_tokens']}")
    print("估算为粗略值：日文/中文按字符近似计数，ASCII 字符按每 4 个 1 token；"
          "输出按现有字幕长度推算，不包含重试及模型推理 token。")


if __name__ == "__main__":
    main()
