"""Step 4: DeepSeek V4 Pro translation/proofreading -> bilingual SRT."""
import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import requests
from pykakasi import kakasi
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (Cue, complete_numbers, env, fail, load_env,  # noqa: E402
                    merge_bilingual, parse_srt, write_srt)

_PROOFREAD_SYSTEM_PROMPT = (
    "你是日语广播字幕校对员。根据日文原文校对中文初译，确保准确、自然、适合字幕阅读，"
    "纠正明显的日语转录错误，不要臆造原文没有的内容；"
    "并在所提供条目范围内统一人物称谓、专有名词和语气。"
    "重点核对人名、地名、机构/节目名等实体，以及可能的误听、误抄、张冠李戴、漏词和添词。"
    "允许将相邻字幕合并，也允许删除确认是识别幻觉或无效噪声的字幕；"
    "合并仅限同一句话内的相邻条目，且合并后总时长不超过 8 秒，"
    "不要跨越停顿、换人或话题转换合并；单条字幕时长尽量不超过 8 秒。"
    "每行必须严格输出为三列：编号或连续编号范围、校正后日文、校对后中文；"
    "列之间用真正的制表符分隔，例如："
    "2-3\t校正后日文\t校对后中文。"
    "合并连续字幕时用例如“2-3”表示覆盖的编号，并输出合并后的日文和中文。"
    "删除整条错误字幕时，输出“编号\t<DELETE>\t<DELETE>”"
    "或“起止编号\t<DELETE>\t<DELETE>”。"
    "输入中的罗马音只用于辅助读音核对，不是输出内容；"
    "所有输入编号必须且只能被覆盖一次；不要遗漏、重复或覆盖未指定编号，不要附加解释。"
    "如果遇到短日文-短中文文本，联系上下文，若有上文中提到了来自xxx的什么歌曲，则一律删除此类文本"
    "文段最后的广告酌情删除。"
)

_TRANSLATION_SYSTEM_PROMPT = (
    "你是日语广播字幕翻译员。将提供的日文字幕翻译成自然、准确、适合字幕阅读的简体中文，"
    "并在所提供条目范围内统一人物称谓、专有名词和语气。"
    "严格按用户消息指定的编号逐条输出，不合并、不遗漏、不改动编号；"
    "每行编号和译文之间用一个真正的制表符分隔，例如：1\t你好，不要附加解释。"
)


@dataclass
class _ProofreadRow:
    indices: tuple
    japanese: str
    chinese: str
    deleted: bool = False


class _TokenUsage:
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_input_tokens = 0
        self.cost_output_tokens = 0
        self.by_model = {}
        self.unpriced_models = set()
        self.reported_input_requests = 0
        self.reported_output_requests = 0
        self.missing_input_requests = 0
        self.missing_output_requests = 0

    def record(self, usage, model=None):
        prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        input_reported = (
            isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool)
            and prompt_tokens >= 0
        )
        output_reported = (
            isinstance(completion_tokens, int) and not isinstance(completion_tokens, bool)
            and completion_tokens >= 0
        )
        if input_reported:
            self.input_tokens += prompt_tokens
            self.reported_input_requests += 1
        else:
            self.missing_input_requests += 1
        if output_reported:
            self.output_tokens += completion_tokens
            self.reported_output_requests += 1
        else:
            self.missing_output_requests += 1
        if model:
            model_usage = self.by_model.setdefault(model, [0, 0])
            if input_reported:
                model_usage[0] += prompt_tokens
            if output_reported:
                model_usage[1] += completion_tokens
        if model and model != "deepseek-v4-pro":
            self.unpriced_models.add(model)
        else:
            if input_reported:
                self.cost_input_tokens += prompt_tokens
            if output_reported:
                self.cost_output_tokens += completion_tokens
        input_text = str(prompt_tokens) if input_reported else "未报告"
        output_text = str(completion_tokens) if output_reported else "未报告"
        model_text = f"（{model}）" if model else ""
        price_text = (
            "费用未估算（需按模型、时段和缓存费率核对）"
            if model and model != "deepseek-v4-pro"
            else f"累计费用（已报告用量）{self.cost:.6f} 元"
        )
        print(
            f"本次 token{model_text}：输入 {input_text}，输出 {output_text}；"
            f"{price_text}",
            file=sys.stderr,
        )

    @property
    def cost(self):
        return (self.cost_input_tokens * 0.30 + self.cost_output_tokens * 9.0) / 1_000_000

    def report(self):
        notes = []
        if self.missing_input_requests or self.missing_output_requests:
            notes.append("部分响应未提供 token")
        if self.unpriced_models:
            notes.append("未估价模型不计入费用")
        completeness = f"（{'; '.join(notes)}）" if notes else ""
        by_model = "；".join(
            f"{model}: 输入 {counts[0]}、输出 {counts[1]}"
            for model, counts in sorted(self.by_model.items())
        )
        model_text = f"；按模型统计：{by_model}" if by_model else ""
        print(
            f"API 报告 token 合计：输入 {self.input_tokens}，输出 {self.output_tokens}；"
            f"费用估算（仅按配置费率）：{self.cost:.6f} 元"
            f"{model_text}{completeness}",
            file=sys.stderr,
        )


_PROOFREAD_ROW = re.compile(r"^(\d+)(?:-(\d+))?$")
_JAPANESE_CHAR = re.compile(r"[\u3040-\u30ff]")
_NUMBER_TOKEN = re.compile(r"\d+(?:[.,]\d+)*")
_LATIN_ACRONYM = re.compile(r"\b[A-Z][A-Z0-9]{1,}\b")
_KATAKANA_TOKEN = re.compile(r"[ァ-ヶー]{2,}")
_INTERJECTION = re.compile(r"^(啊|嗯|哦|哎|唉|欸|诶|喂|哈)(?:[，,、！!？?：:\s]|$)")
_HARD_MARKERS = re.compile(
    r"ダジャレ|駄洒落|言葉遊び|掛詞|かけことば|ダブルミーニング|"
    r"韻を踏|略語|ネットスラング|草|ｗ{2,}|w{2,}"
)
_DELETE_MARKER = re.compile(r"^<\s*DELETE\s*>$", re.IGNORECASE)
_JAPANESE_PRONOUNS = (
    "これ", "それ", "あれ", "どれ", "ここ", "そこ", "あそこ", "どこ",
    "この", "その", "あの", "どの", "彼", "彼女", "あいつ", "こいつ",
    "そいつ", "どいつ", "あなた", "君", "お前",
)
_PROOFREAD_INTERJECTIONS = {"啊", "嗯", "哦", "哎", "唉", "欸", "诶", "喂", "哈"}
_ROMANIZER = kakasi()
DEFAULT_CONFIDENCE_THRESHOLD = math.log(0.85)


def _confidence_percent(avg_logprob):
    """Convert Whisper's average token log probability to a probability-like percent."""
    return math.exp(min(avg_logprob, 0)) * 100


def _romanize_japanese(text):
    return " ".join(
        part["hepburn"] for part in _ROMANIZER.convert(text)
        if part["hepburn"]
    )


def _quality_gate(japanese, drafts, terms=None, confidence_scores=None,
                  confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    """Return 1-based row flags: normal rows are omitted, hard outranks suspect."""
    if len(japanese) != len(drafts):
        raise ValueError("日文字幕与初译数量不一致，无法执行质量门控")
    if confidence_scores is not None and len(confidence_scores) != len(japanese):
        raise ValueError("Whisper 置信度数量与字幕行数不一致")

    flags = {}
    for index, (source, draft) in enumerate(zip(japanese, drafts), 1):
        source_length = len(re.sub(r"\s", "", source))
        draft_length = len(re.sub(r"\s", "", draft))
        ratio = draft_length / source_length if source_length else 0
        source_numbers = _NUMBER_TOKEN.findall(source)
        draft_numbers = _NUMBER_TOKEN.findall(draft)
        residual_japanese = _JAPANESE_CHAR.search(draft)
        source_acronyms = set(_LATIN_ACRONYM.findall(source))
        draft_acronyms = set(_LATIN_ACRONYM.findall(draft))
        source_has_named_marker = any(
            marker in source
            for marker in ("さん", "氏", "町", "市", "県", "駅", "村", "区", "大学",
                           "空港", "神社", "寺", "川", "山")
        )
        source_has_katakana_name = bool(_KATAKANA_TOKEN.search(source))
        draft_has_name_hint = (
            bool(re.search(r"[·•・]", draft))
            or any(
                marker in draft
                for marker in ("先生", "女士", "男士", "市", "县", "镇", "站", "大学",
                               "机场", "神社", "寺", "川", "山")
            )
        )
        possible_entity_mismatch = (
            (source_has_named_marker or source_has_katakana_name) and not draft_has_name_hint
        )
        term_mismatch = any(
            source_term in source and expected_translation not in draft
            for source_term, expected_translation in (terms or {}).items()
        )
        suspect = (
            ratio < 0.5 or ratio > 2.5
            or residual_japanese is not None
            or source_numbers != draft_numbers
            or bool(source_acronyms - draft_acronyms)
            or possible_entity_mismatch
            or term_mismatch
            or (
                confidence_scores is not None
                and confidence_scores[index - 1] is not None
                and confidence_scores[index - 1] <= confidence_threshold
            )
        )

        pronoun_count = sum(source.count(pronoun) for pronoun in _JAPANESE_PRONOUNS)
        hard = pronoun_count >= 2 or bool(_HARD_MARKERS.search(source)) \
            or bool(source_acronyms)
        if hard:
            flags[index] = "hard"
        elif suspect:
            flags[index] = "suspect"

    previous_interjection = None
    previous_index = None
    for index, draft in enumerate(drafts, 1):
        match = _INTERJECTION.match(draft.strip())
        interjection = match.group(1) if match else None
        if (interjection and interjection in _PROOFREAD_INTERJECTIONS
                and interjection == previous_interjection):
            if flags.get(index) != "hard":
                flags[index] = "suspect"
            if flags.get(previous_index) != "hard":
                flags[previous_index] = "suspect"
        previous_interjection = interjection
        previous_index = index

    for index in range(2, len(drafts) + 1):
        previous_draft = re.sub(r"\s", "", drafts[index - 2])
        current_draft = re.sub(r"\s", "", drafts[index - 1])
        previous_source = re.sub(r"\s", "", japanese[index - 2])
        current_source = re.sub(r"\s", "", japanese[index - 1])
        if previous_draft and previous_draft == current_draft and previous_source != current_source:
            if flags.get(index - 1) != "hard":
                flags[index - 1] = "suspect"
            if flags.get(index) != "hard":
                flags[index] = "suspect"
    return flags


def deepseek_refine(ja_batch, zh_batch, numbers, base_url, key, model,
                    max_lines=40, thinking="disabled", max_tokens=8192,
                    custom_prompt="", token_usage=None, romaji_batch=None,
                    confidence_scores=None):
    """Refine selected numbered Japanese/draft pairs without changing source text."""
    if len(ja_batch) != len(zh_batch):
        raise ValueError("日文字幕与初译数量不一致，无法进行精修")
    if confidence_scores is not None and len(confidence_scores) != len(ja_batch):
        raise ValueError("Whisper 置信度数量与字幕行数不一致")
    numbers = list(numbers)
    refined = {}
    for start in range(0, len(numbers), max_lines):
        batch_numbers = numbers[start:start + max_lines]

        def request(target):
            target_set = set(target)
            context_numbers = sorted({
                neighbor
                for number in target
                for neighbor in (number - 1, number + 1)
                if 1 <= neighbor <= len(ja_batch) and neighbor not in target_set
            })
            context = (
                "\n相邻日文上下文（仅供理解，不要翻译或输出）：\n"
                + _numbered_subset(ja_batch, context_numbers)
                if context_numbers else ""
            )
            user_prompt = (
                ("\n用户自定义翻译要求：\n" + custom_prompt.strip()
                 if custom_prompt.strip() else "")
                + f"\n只精修以下编号的中文译文：{', '.join(map(str, target))}。\n"
                + context + "\n"
                + (
                    "每行输入依次为编号、日文原文、日文罗马音、当前中文译文、"
                    "Whisper置信度概率近似值，列之间以制表符分隔。"
                    "罗马音和置信度仅作参考，不需输出。"
                    if romaji_batch is not None and confidence_scores is not None else
                    "每行输入依次为编号、日文原文、日文罗马音、当前中文译文，"
                    "列之间以制表符分隔。罗马音仅作读音参考，不需输出。"
                    if romaji_batch is not None else
                    "每行输入依次为编号、日文原文、当前中文译文、"
                    "Whisper置信度概率近似值，列之间以制表符分隔；"
                    "置信度仅作参考，不需输出。"
                    if confidence_scores is not None else
                    "每行输入依次为编号、日文原文、当前中文译文，列之间以制表符分隔。"
                )
                +
                "只修正中文译文，不修改日文原文；不添加上下文之外的信息。"
                "严格按每个编号输出一行，编号和译文之间用真正的制表符分隔，例如：1\t你好。"
                "不要解释。\n"
                "待精修条目：\n"
                + _numbered_pair_subset(
                    ja_batch, zh_batch, target, romaji_items=romaji_batch,
                    confidence_scores=confidence_scores)
            )
            return _deepseek_request(
                "你是日语广播字幕的中文精修员。准确检查原文与译文，"
                "保留原意、语气、数字及专名，输出自然简洁的简体中文字幕。"
                "只输出指定编号的译文，不要解释。",
                user_prompt, base_url, key, model, thinking, max_tokens,
                token_usage,
            )

        results = complete_numbers(batch_numbers, request, "精修")
        refined.update(zip(batch_numbers, results))
    return refined


def _parse_proofread_rows(text, expected_numbers):
    expected = set(expected_numbers)
    parsed = []
    covered = set()
    for line in text.strip().splitlines():
        fields = line.split("\t", 2)
        if len(fields) != 3:
            raise ValueError(f"校对结果格式错误，要求每行三列（Tab 分隔）: {line}")
        match = _PROOFREAD_ROW.fullmatch(fields[0].strip())
        if not match:
            raise ValueError(f"校对结果编号格式错误: {fields[0]}")
        first = int(match.group(1))
        last = int(match.group(2) or first)
        if first > last:
            raise ValueError(f"校对结果编号范围无效: {fields[0]}")
        if (first not in expected or last not in expected
                or last - first + 1 > len(expected)):
            raise ValueError(f"校对结果包含未指定的字幕编号: {fields[0]}")
        indices = tuple(range(first, last + 1))
        if not set(indices).issubset(expected):
            raise ValueError(f"校对结果包含未指定的字幕编号: {fields[0]}")
        if covered.intersection(indices):
            raise ValueError(f"校对结果重复覆盖字幕编号: {fields[0]}")
        japanese, chinese = fields[1].strip(), fields[2].strip()
        japanese_delete = bool(_DELETE_MARKER.fullmatch(japanese))
        chinese_delete = bool(_DELETE_MARKER.fullmatch(chinese))
        deleted = japanese_delete and chinese_delete
        if not deleted and (not japanese or not chinese):
            raise ValueError(f"校对结果日文或中文为空: {fields[0]}")
        if japanese_delete != chinese_delete:
            raise ValueError(f"删除条目必须在日文和中文列都使用 <DELETE>: {fields[0]}")
        covered.update(indices)
        parsed.append(_ProofreadRow(indices, japanese, chinese, deleted))
    return parsed, expected - covered


def _complete_proofread(numbers, request):
    rows, missing = _parse_proofread_rows(request(numbers), numbers)
    if missing:
        retry_rows, still_missing = _parse_proofread_rows(request(sorted(missing)), missing)
        rows.extend(retry_rows)
        if still_missing:
            raise ValueError(
                f"字幕校对仍缺少字幕编号 {', '.join(map(str, sorted(still_missing)))}，"
                "请检查模型输出"
            )
    return sorted(rows, key=lambda row: row.indices[0])


def _numbered_subset(items, numbers):
    return "\n".join(
        f"{i}\t{items[i - 1].replace(chr(10), ' ')}"
        for i in numbers
    )


def _cue_span(cues, numbers):
    """Duration in seconds covered by the given (sorted, contiguous) cue numbers."""
    if not cues or not numbers:
        return None
    start = _timestamp_seconds(cues[numbers[0] - 1].start)
    end = _timestamp_seconds(cues[numbers[-1] - 1].end)
    return end - start


def _numbered_pair_subset(ja_items, zh_items, numbers, cues=None,
                          romaji_items=None, confidence_scores=None):
    def row(i):
        fields = [str(i), ja_items[i - 1].replace(chr(10), " ")]
        if romaji_items is not None:
            fields.append(romaji_items[i - 1].replace(chr(10), " "))
        fields.append(zh_items[i - 1].replace(chr(10), " "))
        if confidence_scores is not None:
            score = confidence_scores[i - 1]
            fields.append(
                f"{_confidence_percent(score):.1f}%"
                if score is not None else "未提供"
            )
        return "\t".join(fields)

    if cues:
        return "\n".join(
            f"{row(i)}\t{_cue_span(cues, [i]):.1f}s" for i in numbers
        )
    return "\n".join(
        row(i) for i in numbers
    )


def _numbered_pair_markdown_table(ja_items, zh_items, numbers, cues=None,
                                  romaji_items=None):
    headers = ["编号", "日文原文"]
    if romaji_items is not None:
        headers.append("罗马音")
    headers.append("中文初译")
    lines = [
        "| " + " | ".join(headers + (["时长"] if cues else [])) + " |",
        "|---:|" + "---|" * (len(headers) - 1 + bool(cues)),
    ]
    for i in numbers:
        japanese = ja_items[i - 1].replace("\n", " ").replace("|", "\\|").strip()
        chinese = zh_items[i - 1].replace("\n", " ").replace("|", "\\|").strip()
        fields = [str(i), japanese]
        if romaji_items is not None:
            romaji = romaji_items[i - 1].replace("\n", " ").replace("|", "\\|").strip()
            fields.append(romaji)
        fields.append(chinese)
        if cues:
            fields.append(f"{_cue_span(cues, [i]):.1f}s")
        lines.append("| " + " | ".join(fields) + " |")
    return "\n".join(lines)


def _proofread_slices(total, max_lines):
    """Split 1..total so a single reply cannot overrun the output length limit."""
    size = max_lines if max_lines and max_lines > 0 else total or 1
    return [list(range(start, min(start + size - 1, total) + 1))
            for start in range(1, total + 1, size)]


def _deepseek_request(system_prompt, user_prompt, base_url, key, model, thinking, max_tokens,
                      token_usage=None):
    """Send one chat completion; drop the thinking field if the API rejects it."""
    payload = {"model": model, "temperature": 0.2, "max_tokens": max_tokens,
               "messages": [
                   {"role": "system", "content": system_prompt},
                   {"role": "user", "content": user_prompt},
               ]}
    if thinking == "disabled":
        payload["thinking"] = {"type": "disabled"}
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {key}"}
    r = requests.post(url, headers=headers, json=payload, timeout=900)
    if r.status_code == 400 and thinking == "disabled":
        print("提示: 接口不接受关闭推理参数，已按默认推理模式重试", file=sys.stderr)
        payload.pop("thinking")
        r = requests.post(url, headers=headers, json=payload, timeout=900)
    r.raise_for_status()
    data = r.json()
    if token_usage is not None:
        token_usage.record(data.get("usage"), model)
    choice = data["choices"][0]
    if choice.get("finish_reason") == "length":
        raise ValueError("回复被输出长度上限截断，请调小 --proofread-lines")
    content = choice["message"].get("content") or ""
    if not content.strip():
        reasoning = data.get("usage", {}).get("completion_tokens_details", {})
        hint = "（推理占用了全部输出额度，可尝试 --deepseek-thinking disabled）" \
            if reasoning.get("reasoning_tokens") else ""
        raise ValueError(f"模型返回内容为空{hint}")
    return content


def _reference_section(terms, reference):
    parts = []
    if terms:
        rows = "\n".join(
            f"| {s.replace('|', chr(92) + '|')} | {t.replace('|', chr(92) + '|')} |"
            for s, t in terms.items())
        parts.append("术语/人名/地名对照表（以此为准，译文必须统一采用）：\n"
                     "| 日文原文 | 固定译法 |\n|---|---|\n" + rows)
    if reference.strip():
        parts.append("节目/人物背景资料（仅用于核对人名、地名、专有名词及识别误听，"
                     "不是待校对内容，不要输出）：\n" + reference.strip())
    return ("\n\n" + "\n\n".join(parts) + "\n") if parts else ""


def deepseek_proofread(ja_batch, zh_batch, base_url, key, model, max_lines=200,
                       thinking="auto", max_tokens=65536, custom_prompt="",
                       token_usage=None, cues=None, romaji_batch=None):
    """Proofread Japanese/draft pairs with romanization in bounded batches."""
    reference_text = ""
    if len(ja_batch) != len(zh_batch):
        raise ValueError("日文字幕与初译数量不一致，无法进行校对")
    if romaji_batch is not None and len(romaji_batch) != len(ja_batch):
        raise ValueError("日文字幕与罗马音数量不一致，无法进行校对")
    corrected_rows = []
    for numbers in tqdm(_proofread_slices(len(ja_batch), max_lines),
                        desc="DeepSeek 字幕校对", unit="段", file=sys.stderr):

        def request(target):
            user_prompt = (
                ("\n用户自定义校对要求：\n" + custom_prompt.strip()
                 if custom_prompt.strip() else "")
                + reference_text
                + f"\n只校对以下带编号的条目，编号为：{', '.join(map(str, target))}。\n"
                + ("每行输入依次为编号、日文原文、日文罗马音、中文初译，"
                   "列之间以制表符分隔；罗马音仅作读音参考，不需输出。\n"
                   if romaji_batch is not None else
                   "每行输入依次为编号、日文原文、中文初译，列之间以制表符分隔。\n")
                + "先结合下面的 Markdown 对照表进行人名/地名/机构名与可能误摘抄核对，"
                "再按系统格式输出三列结果。\n"
                "按系统规定输出三列；需要纠正日文识别错误时修正日文列；"
                "合并相邻条目时用编号范围，并先确认合并后总时长不超过 8 秒；"
                "确认是幻听或无效噪声时按规定删除。\n"
                "待校对条目（TSV，末列为该条时长）：\n"
                + _numbered_pair_subset(
                    ja_batch, zh_batch, target, cues, romaji_batch)
                + "\n\n校对参考对照表（Markdown）：\n"
                + _numbered_pair_markdown_table(
                    ja_batch, zh_batch, target, cues, romaji_batch)
            )
            return _deepseek_request(
                _PROOFREAD_SYSTEM_PROMPT, user_prompt, base_url, key, model,
                thinking, max_tokens, token_usage)

        corrected_rows.extend(_complete_proofread(numbers, request))
    return corrected_rows


def deepseek_translate(ja_batch, base_url, key, model, max_lines=200,
                        thinking="disabled", max_tokens=65536, custom_prompt="",
                        token_usage=None):
    """Translate numbered Japanese subtitles in bounded batches."""
    translations = []
    for numbers in tqdm(_proofread_slices(len(ja_batch), max_lines),
                        desc="DeepSeek 翻译校对", unit="段", file=sys.stderr):

        def request(target):
            user_prompt = (
                ("\n用户自定义要求：\n" + custom_prompt.strip()
                 if custom_prompt.strip() else "")
                + f"\n只翻译以下带编号的日文字幕，编号为：{', '.join(map(str, target))}。\n"
                "待翻译字幕：\n" + _numbered_subset(ja_batch, target)
            )
            return _deepseek_request(
                _TRANSLATION_SYSTEM_PROMPT, user_prompt, base_url, key, model,
                thinking, max_tokens, token_usage)

        translations.extend(complete_numbers(numbers, request, "翻译校对"))
    return translations


def _timestamp_seconds(value):
    hours, minutes, rest = value.split(":")
    seconds, millis = rest.split(",")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000


def _merge_short_cues(cues, translations, max_chars=12, max_gap=0.5, max_group=3):
    """Merge adjacent short bilingual cues without changing the overall time span."""
    if len(cues) != len(translations):
        raise ValueError("字幕与译文数量不一致")

    groups = []
    current = []
    for cue, translation in zip(cues, translations):
        can_merge = False
        if current and len(current) < max_group:
            previous = current[-1][0]
            gap = _timestamp_seconds(cue.start) - _timestamp_seconds(previous.end)
            duration = _timestamp_seconds(cue.end) - _timestamp_seconds(current[0][0].start)
            can_merge = (
                len(cue.text.replace("\n", "").strip()) <= max_chars
                and all(len(c.text.replace("\n", "").strip()) <= max_chars
                        for c, _ in current)
                and 0 <= gap <= max_gap
                and duration <= 6
            )
        if current and not can_merge:
            groups.append(current)
            current = []
        current.append((cue, translation))
    if current:
        groups.append(current)

    merged = []
    for group in groups:
        first, last = group[0][0], group[-1][0]
        japanese = " / ".join(c.text.replace("\n", " ").strip() for c, _ in group)
        chinese = " / ".join(t.strip() for _, t in group)
        scores = [c.avg_logprob for c, _ in group if c.avg_logprob is not None]
        merged.append(Cue(first.index, first.start, last.end,
                          f"{japanese}\n{chinese}",
                          min(scores) if scores else None))
    return merged


def _proofread_results_to_cues(cues, rows):
    outputs = []
    translations = []
    for row in rows:
        if row.deleted:
            continue
        first = cues[row.indices[0] - 1]
        last = cues[row.indices[-1] - 1]
        scores = [
            cues[index - 1].avg_logprob
            for index in row.indices
            if cues[index - 1].avg_logprob is not None
        ]
        outputs.append(Cue(first.index, first.start, last.end, row.japanese,
                           min(scores) if scores else None))
        translations.append(row.chinese)
    return outputs, translations


def _annotate_low_confidence(cues, threshold):
    annotated = []
    for cue in cues:
        text = cue.text
        if cue.avg_logprob is not None and cue.avg_logprob <= threshold:
            japanese, separator, chinese = text.rpartition("\n")
            percent = _confidence_percent(cue.avg_logprob)
            note = f"<low conf：{percent:.1f}%（概率近似）>"
            text = f"{japanese}{note}\n{chinese}" if separator else f"{text}{note}"
        annotated.append(Cue(cue.index, cue.start, cue.end, text))
    return annotated


def main(argv=None):
    load_env()
    p = argparse.ArgumentParser(
        description="日文 SRT -> 中日最终字幕及罗马音校对稿（DeepSeek 翻译/校对）")
    p.add_argument("srt", help="日文 SRT")
    p.add_argument("--zh-srt", help="中文初译 SRT；指定后校对初译，否则直接翻译日文")
    p.add_argument("--suspects-only", action="store_true",
                   help="只对质量门控标记的可疑/疑难初译调用 DeepSeek")
    p.add_argument("--confidence-threshold", type=float,
                   default=float(env("WHISPER_CONFIDENCE_THRESHOLD",
                                     str(DEFAULT_CONFIDENCE_THRESHOLD))),
                   help="Whisper avg_logprob 低于或等于此值时标为可疑（默认 85%%， -0.1625）")
    p.add_argument("-o", "--output",
                   help="中日最终字幕输出，默认 <名称>.zh-ja.srt；另生成 -romaji.srt 校对稿")
    p.add_argument("--proofread-lines", type=int,
                   default=int(env("PROOFREAD_LINES", "200")),
                   help="DeepSeek 每次翻译/校对的字幕条数")
    p.add_argument("--deepseek-base-url", default=env("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    p.add_argument("--deepseek-model", default=env("DEEPSEEK_MODEL", "deepseek-v4-pro"))
    p.add_argument("--deepseek-chat-model",
                   default=env("DEEPSEEK_CHAT_MODEL", "deepseek-v4-pro"),
                   help="质量门控精修使用的 DeepSeek 模型")
    p.add_argument("--deepseek-thinking", choices=("disabled", "auto"),
                   default=env("DEEPSEEK_THINKING", "disabled"),
                   help="是否关闭 DeepSeek 推理；推理会占满输出额度导致回复为空（默认 disabled）")
    p.add_argument("--proofread-thinking", choices=("disabled", "auto"),
                   default=env("PROOFREAD_THINKING", "auto"),
                   help="校对模式是否启用 DeepSeek 推理（默认 auto）")
    p.add_argument("--proofread-max-tokens", type=int,
                   default=int(env("PROOFREAD_MAX_TOKENS", "65536")),
                   help="每段校对回复的输出上限（默认 65536）")
    p.add_argument("--no-token-usage", action="store_true",
                   help="不显示 DeepSeek API 返回的 token 用量和费用")
    p.add_argument("--proofread-prompt-file",
                   help="包含额外校对要求的 Markdown 文件（UTF-8）")
    p.add_argument("--terms-file",
                   help="术语对照 JSON 文件（也用于检测初译中的术语不一致）")
    p.add_argument("--reference-file",
                   help="节目/人物背景 Markdown（UTF-8），用于全量校对时核对人名地名")
    p.add_argument("--no-merge-short", action="store_true",
                   help="不合并相邻的短字幕句子")
    p.add_argument("--short-cue-chars", type=int, default=12,
                   help="短句合并的每条字幕最大字符数（默认 12）")
    p.add_argument("--merge-gap", type=float, default=0.5,
                   help="短句合并允许的最大字幕间隔秒数（默认 0.5）")
    args = p.parse_args(argv)
    if not math.isfinite(args.confidence_threshold):
        fail("--confidence-threshold 必须是有限数字")
    if args.proofread_lines < 1:
        fail("--proofread-lines 必须大于 0")
    if args.proofread_max_tokens < 1:
        fail("--proofread-max-tokens 必须大于 0")
    if args.short_cue_chars < 1 or args.merge_gap < 0:
        fail("--short-cue-chars 必须大于 0，--merge-gap 不能小于 0")
    if args.suspects_only and not args.zh_srt:
        fail("--suspects-only 必须同时指定 --zh-srt")

    key = env("DEEPSEEK_API_KEY")
    if not key:
        fail("未设置环境变量 DEEPSEEK_API_KEY（校对为必需步骤），请在 .env 或环境中配置")
    src = Path(args.srt)
    if not src.is_file():
        fail(f"找不到 SRT: {src}")
    stem = src.name[:-len(".ja.srt")] if src.name.endswith(".ja.srt") else src.stem
    custom_prompt = ""
    if args.proofread_prompt_file:
        prompt_path = Path(args.proofread_prompt_file)
        if not prompt_path.is_file():
            fail(f"找不到校对提示词文件: {prompt_path}")
        custom_prompt = prompt_path.read_text(encoding="utf-8").strip()
        if not custom_prompt:
            fail(f"校对提示词文件为空: {prompt_path}")
    reference = ""
    if args.reference_file:
        reference_path = Path(args.reference_file)
        if not reference_path.is_file():
            fail(f"找不到背景参考文件: {reference_path}")
        reference = reference_path.read_text(encoding="utf-8")
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
                not isinstance(source_term, str)
                or not source_term.strip()
                or not isinstance(expected_translation, str)
                or not expected_translation.strip()
                for source_term, expected_translation in terms.items()):
            fail("术语表必须是“原文字符串: 译文字符串”的 JSON 对象")
    cues = parse_srt(src.read_text(encoding="utf-8"))
    if not cues:
        fail("SRT 中没有字幕")
    token_usage = None if args.no_token_usage else _TokenUsage()
    proofreading = bool(args.zh_srt)
    try:
        japanese = [cue.text for cue in cues]
        romaji = [_romanize_japanese(text) for text in japanese]
        if proofreading:
            zh_path = Path(args.zh_srt)
            if not zh_path.is_file():
                fail(f"找不到中文初译 SRT: {zh_path}")
            drafts = [c.text for c in parse_srt(zh_path.read_text(encoding="utf-8"))]
            if len(drafts) != len(cues):
                fail(f"中文初译条数 ({len(drafts)}) 与日文字幕条数 ({len(cues)}) 不一致")
            if args.suspects_only:
                confidence_scores = [cue.avg_logprob for cue in cues]
                flags = _quality_gate(
                    japanese, drafts, terms, confidence_scores,
                    args.confidence_threshold)
                suspect_numbers = [
                    number for number, flag in flags.items() if flag == "suspect"
                ]
                hard_numbers = [
                    number for number, flag in flags.items() if flag == "hard"
                ]
                print(
                    f"质量门控：普通可疑 {len(suspect_numbers)} 条，"
                    f"疑难 {len(hard_numbers)} 条，正常 {len(cues) - len(flags)} 条",
                    file=sys.stderr,
                )
                translations = list(drafts)
                refine_numbers = sorted(suspect_numbers + hard_numbers)
                if refine_numbers:
                    translations_by_number = deepseek_refine(
                        japanese, drafts, refine_numbers, args.deepseek_base_url,
                        key, args.deepseek_chat_model, args.proofread_lines,
                        "disabled", args.proofread_max_tokens, custom_prompt,
                        token_usage, romaji, confidence_scores)
                    for number, translation in translations_by_number.items():
                        translations[number - 1] = translation
                source_cues = cues
            else:
                proofread_rows = deepseek_proofread(
                    japanese, drafts, args.deepseek_base_url, key, args.deepseek_model,
                    args.proofread_lines, args.proofread_thinking, args.proofread_max_tokens,
                    custom_prompt + _reference_section(terms, reference),
                    token_usage, cues, romaji)
                source_cues, translations = _proofread_results_to_cues(cues, proofread_rows)
        else:
            source_cues = cues
            translations = deepseek_translate(
                japanese, args.deepseek_base_url, key, args.deepseek_model,
                args.proofread_lines, args.deepseek_thinking, args.proofread_max_tokens,
                custom_prompt, token_usage)
    except (requests.RequestException, ValueError, KeyError, json.JSONDecodeError) as e:
        fail(f"翻译/校对失败: {e}")
    if token_usage is not None:
        token_usage.report()

    out = Path(args.output) if args.output else src.with_name(stem + ".zh-ja.srt")
    if args.no_merge_short:
        output_cues = merge_bilingual(source_cues, translations)
        review_japanese = [cue.text for cue in source_cues]
        review_chinese = translations
    else:
        output_cues = _merge_short_cues(source_cues, translations,
                                        max_chars=args.short_cue_chars,
                                        max_gap=args.merge_gap)
        review_japanese = []
        review_chinese = []
        for cue in output_cues:
            japanese, separator, chinese = cue.text.partition("\n")
            if not separator:
                raise ValueError(
                    f"双语字幕缺少中文行，无法生成罗马音校对稿: {cue.index}")
            review_japanese.append(japanese)
            review_chinese.append(chinese)
    romaji_out = out.with_name(out.stem + "-romaji.srt")
    review_cues = []
    for cue, japanese, chinese in zip(output_cues, review_japanese, review_chinese):
        review_cues.append(Cue(
            cue.index, cue.start, cue.end,
            f"{japanese}\n{_romanize_japanese(japanese)}\n{chinese}",
        ))
    output_cues = _annotate_low_confidence(output_cues, args.confidence_threshold)
    write_srt(output_cues, out, include_confidence=False)
    write_srt(review_cues, romaji_out, include_confidence=False)
    print(out)
    print(romaji_out)


if __name__ == "__main__":
    main()
