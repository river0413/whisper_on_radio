import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
from common import format_ts, merge_bilingual, parse_srt, write_srt  # noqa: E402


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SRT = "1\n00:00:00,000 --> 00:00:01,500\nこんにちは\n\n2\n00:00:02,000 --> 00:00:03,000\nありがとう\n"


def test_roundtrip(tmp_path):
    cues = parse_srt(SRT)
    assert [c.text for c in cues] == ["こんにちは", "ありがとう"]
    bi = merge_bilingual(cues, ["你好", "谢谢"])
    p = tmp_path / "o.srt"
    write_srt(bi, p)
    again = parse_srt(p.read_text(encoding="utf-8"))
    assert again[1].text == "ありがとう\n谢谢"


def test_srt_confidence_metadata_roundtrips_without_becoming_caption_text(tmp_path):
    cue = parse_srt(SRT)[0]
    cue.avg_logprob = -0.754321
    path = tmp_path / "confidence.ja.srt"

    write_srt([cue], path)

    contents = path.read_text(encoding="utf-8")
    assert "こんにちは <-0.754321->" in contents
    parsed = parse_srt(contents)[0]
    assert parsed.text == "こんにちは"
    assert parsed.avg_logprob == pytest.approx(-0.754321)


def test_merge_mismatch():
    with pytest.raises(ValueError):
        merge_bilingual(parse_srt(SRT), ["x"])


def test_format_ts():
    assert format_ts(3661.5) == "01:01:01,500"


def test_numbered_parse():
    m = load("03_translate_srt")
    assert m._parse_numbered("1\ta\n2\tb", 2) == ["a", "b"]
    with pytest.raises(ValueError):
        m._parse_numbered("1\ta", 2)


def test_model_format_prompts_use_real_tab_separators():
    translate = load("03_translate_srt")
    proofread = load("04_proofread_srt")
    prompts = (
        translate._TRANSLATION_SYSTEM_PROMPT,
        proofread._PROOFREAD_SYSTEM_PROMPT,
        proofread._TRANSLATION_SYSTEM_PROMPT,
    )
    assert all("\t" in prompt for prompt in prompts)
    assert all("<Tab>" not in prompt for prompt in prompts)


def test_missing_numbered_rows_are_retried_with_context():
    m = load("03_translate_srt")
    calls = []

    def request(numbers):
        calls.append(numbers)
        if len(calls) == 1:
            return "1\t甲\n3\t丙"
        return "2\t乙"

    assert m._complete_numbered(["a", "b", "c"], request, "翻译") == ["甲", "乙", "丙"]
    assert calls == [[1, 2, 3], [2]]


def test_context_groups_keep_neighboring_lines_together():
    m = load("03_translate_srt")
    assert [len(group) for group in m._context_groups(list(range(8)))] == [3, 3, 2]


def test_translation_request_contains_only_requested_numbered_rows(monkeypatch):
    m = load("03_translate_srt")
    calls = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "10\t译文\n11\t译文\n12\t译文"}}

    def post(url, json, timeout):
        calls.append(json)
        return Response()

    monkeypatch.setattr(m.requests, "post", post)
    assert m.ollama_translate(["甲", "乙", "丙"], "http://ollama", "model",
                              start_number=10) == [
        "译文", "译文", "译文"
    ]
    messages = calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "10\t甲" in messages[1]["content"]
    assert "11\t乙" in messages[1]["content"]
    assert "12\t丙" in messages[1]["content"]
    assert "将以下内容翻译为简体中文" in messages[1]["content"]
    assert "只输出译文" in messages[0]["content"]
    assert calls[0]["options"] == {
        "top_k": 20, "top_p": 0.6, "temperature": 0.7,
        "repeat_penalty": 1.05,
    }


def test_translation_request_injects_json_terminology(monkeypatch):
    m = load("03_translate_srt")
    payloads = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "1\t节目名"}}

    def post(url, json, timeout):
        payloads.append(json)
        return Response()

    monkeypatch.setattr(m.requests, "post", post)
    assert m.ollama_translate(
        ["番組名"], "http://ollama", "hy-mt", terms={"番組名": "节目名"}
    ) == ["节目名"]
    assert '"番組名":"节目名"' in payloads[0]["messages"][1]["content"]


def test_translation_request_adds_markdown_reference_to_system_prompt(monkeypatch):
    m = load("03_translate_srt")
    payloads = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "1\t野中心菜"}}

    def post(url, json, timeout):
        payloads.append(json)
        return Response()

    monkeypatch.setattr(m.requests, "post", post)
    assert m.ollama_translate(
        ["野中ここな"], "http://ollama", "hy-mt",
        reference_text="主持人：野中心菜\n节目：都道府县知识节目",
    ) == ["野中心菜"]
    system_message, user_message = payloads[0]["messages"]
    assert "主持人：野中心菜" in system_message["content"]
    assert "仅用于理解原文" in system_message["content"]
    assert "将以下内容翻译为简体中文" in user_message["content"]
    assert "主持人：野中心菜" not in user_message["content"]


def test_translation_main_loads_markdown_reference(tmp_path, monkeypatch):
    m = load("03_translate_srt")
    source = tmp_path / "a.ja.srt"
    source.write_text(SRT, encoding="utf-8")
    reference = tmp_path / "coconna.md"
    reference.write_text("## 主持人\n野中心菜", encoding="utf-8")
    calls = []

    def translate(batch, host, model, **kwargs):
        calls.append(kwargs["reference_text"])
        return ["初译"] * len(batch)

    monkeypatch.setattr(m, "load_env", lambda: None)
    monkeypatch.setattr(m, "ollama_translate", translate)
    m.main([str(source), "--reference-file", str(reference)])

    assert calls == ["## 主持人\n野中心菜"]
    assert (tmp_path / "a.zh.srt").is_file()


def test_translation_main_rejects_empty_markdown_reference(tmp_path, monkeypatch):
    m = load("03_translate_srt")
    source = tmp_path / "a.ja.srt"
    source.write_text(SRT, encoding="utf-8")
    reference = tmp_path / "empty.md"
    reference.write_text(" \n", encoding="utf-8")
    monkeypatch.setattr(m, "load_env", lambda: None)

    with pytest.raises(SystemExit):
        m.main([str(source), "--reference-file", str(reference)])


def test_transcription_fragments_merge_into_sentences():
    m = load("02_transcribe_srt")
    cues = parse_srt(
        "1\n00:00:00,000 --> 00:00:01,000\n今日は\n\n"
        "2\n00:00:01,100 --> 00:00:02,000\nいい天気です。\n\n"
        "3\n00:00:02,200 --> 00:00:03,000\nまた明日\n\n"
        "4\n00:00:04,500 --> 00:00:05,000\n会いましょう。\n"
    )
    merged = m._merge_sentence_cues(cues)
    assert [cue.text for cue in merged] == ["今日はいい天気です。", "また明日", "会いましょう。"]
    assert merged[0].start == "00:00:00,000"
    assert merged[0].end == "00:00:02,000"


def _transcription_segment(start, end, text, no_speech_prob=0.1,
                           avg_logprob=-0.2, compression_ratio=1.0):
    return SimpleNamespace(
        start=start,
        end=end,
        text=text,
        no_speech_prob=no_speech_prob,
        avg_logprob=avg_logprob,
        compression_ratio=compression_ratio,
    )


def test_transcription_cleanup_filters_only_suspicious_segments():
    m = load("02_transcribe_srt")
    segments = [
        _transcription_segment(0, 1.2, "ご視聴ありがとうございました",
                               no_speech_prob=0.31),
        _transcription_segment(2, 3.2, "ご視聴ありがとうございました",
                               no_speech_prob=0.3),
        _transcription_segment(4, 5.2, "繰り返し", compression_ratio=2.41),
        _transcription_segment(6, 7.2, "通常の台詞", compression_ratio=2.4),
    ]

    cues, short_indices = m._prepare_transcription_segments(segments)

    assert [cue.text for cue in cues] == [
        "ご視聴ありがとうございました", "通常の台詞"
    ]
    assert short_indices == set()


def test_short_transcription_requires_both_low_confidence_signals_to_drop():
    m = load("02_transcribe_srt")
    segments = [
        _transcription_segment(0, 0.8, "疑似字幕", no_speech_prob=0.61,
                               avg_logprob=-1.01),
        _transcription_segment(1, 1.8, "信号一つだけ", no_speech_prob=0.61,
                               avg_logprob=-0.99),
        _transcription_segment(2, 2.8, "信号一つだけ", no_speech_prob=0.6,
                               avg_logprob=-1.01),
        _transcription_segment(3, 4.0, "境界の台詞", no_speech_prob=0.61,
                               avg_logprob=-1.0),
    ]

    cues, short_indices = m._prepare_transcription_segments(segments)

    assert [cue.text for cue in cues] == [
        "信号一つだけ", "信号一つだけ", "境界の台詞"
    ]
    assert short_indices == {1, 2, 3}


def test_short_transcription_duration_boundary_is_one_second():
    m = load("02_transcribe_srt")
    cues, short_indices = m._prepare_transcription_segments([
        _transcription_segment(0, 1.0, "短い"),
        _transcription_segment(2, 3.001, "少し長い"),
    ])

    assert [cue.text for cue in cues] == ["短い", "少し長い"]
    assert short_indices == {1}


def test_short_transcription_merges_with_neighbors_on_both_sides():
    m = load("02_transcribe_srt")
    cues, short_indices = m._prepare_transcription_segments([
        _transcription_segment(0, 1.5, "前の文"),
        _transcription_segment(1.6, 1.9, "短い応答"),
        _transcription_segment(2.1, 4.5, "後ろの文。"),
    ])

    merged = m._merge_sentence_cues(
        cues, short_cue_indices=short_indices)

    assert [cue.text for cue in merged] == ["前の文短い応答後ろの文。"]
    assert (merged[0].start, merged[0].end) == (
        "00:00:00,000", "00:00:04,500"
    )


def test_transcription_merging_keeps_worst_whisper_score():
    m = load("02_transcribe_srt")
    cues, short_indices = m._prepare_transcription_segments([
        _transcription_segment(0, 1.5, "前の文", avg_logprob=-0.2),
        _transcription_segment(1.6, 2.5, "低置信度", avg_logprob=-0.9),
        _transcription_segment(2.6, 4.5, "后面的文。", avg_logprob=-0.4),
    ])

    merged = m._merge_sentence_cues(cues, short_cue_indices=short_indices)

    assert len(merged) == 1
    assert merged[0].avg_logprob == pytest.approx(-0.9)


def test_short_transcription_does_not_merge_across_long_gap_or_sentence_end():
    m = load("02_transcribe_srt")
    cues, short_indices = m._prepare_transcription_segments([
        _transcription_segment(0, 1.5, "ここで終わり。"),
        _transcription_segment(1.6, 1.9, "短い応答"),
        _transcription_segment(2.3, 3.0, "離れた文"),
    ])

    merged = m._merge_sentence_cues(
        cues, short_cue_indices=short_indices)

    assert [cue.text for cue in merged] == [
        "ここで終わり。", "短い応答", "離れた文"
    ]


def test_short_transcription_merging_keeps_five_second_limit():
    m = load("02_transcribe_srt")
    cues, short_indices = m._prepare_transcription_segments([
        _transcription_segment(0, 2.5, "前の長句"),
        _transcription_segment(2.6, 2.8, "短い応答"),
        _transcription_segment(3.0, 5.4, "後ろの長句"),
    ])

    merged = m._merge_sentence_cues(
        cues, short_cue_indices=short_indices)

    assert [cue.text for cue in merged] == ["前の長句短い応答", "後ろの長句"]
    assert all(
        m._timestamp_seconds(cue.end) - m._timestamp_seconds(cue.start) <= 5
        for cue in merged
    )


def test_transcription_fails_if_cleanup_removes_every_segment(monkeypatch, tmp_path,
                                                              capsys):
    m = load("02_transcribe_srt")
    audio = tmp_path / "input.wav"
    audio.touch()

    class WhisperModel:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, *args, **kwargs):
            segment = _transcription_segment(
                0, 0.5, "疑似字幕", no_speech_prob=0.7,
                avg_logprob=-1.2)
            return [segment], SimpleNamespace(duration=1.0)

    monkeypatch.setattr(m, "load_env", lambda: None)
    monkeypatch.setitem(sys.modules, "faster_whisper",
                        SimpleNamespace(WhisperModel=WhisperModel))

    with pytest.raises(SystemExit) as exc:
        m.main([str(audio)])

    assert exc.value.code == 1
    assert "清洗后没有可输出的字幕" in capsys.readouterr().err


def test_transcription_short_trial_limits_whisper_and_writes_logprob_metadata(
        monkeypatch, tmp_path):
    m = load("02_transcribe_srt")
    audio = tmp_path / "input.wav"
    audio.touch()
    options_seen = []

    class WhisperModel:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, path, **kwargs):
            options_seen.append(kwargs)
            return [
                _transcription_segment(0, 1.2, "試験の字幕", avg_logprob=-0.82)
            ], SimpleNamespace(duration=900.0)

    monkeypatch.setattr(m, "load_env", lambda: None)
    monkeypatch.setitem(sys.modules, "faster_whisper",
                        SimpleNamespace(WhisperModel=WhisperModel))

    m.main([str(audio), "--max-duration", "300"])

    assert options_seen[0]["clip_timestamps"] == [0, 300.0]
    assert options_seen[0]["log_progress"] is False
    output = (tmp_path / "input.ja.srt").read_text(encoding="utf-8")
    assert "試験の字幕 <-0.82->" in output
    parsed = parse_srt(output)[0]
    assert parsed.text == "試験の字幕"
    assert parsed.avg_logprob == pytest.approx(-0.82)


def test_transcription_cues_are_split_at_five_seconds():
    m = load("02_transcribe_srt")
    cues = parse_srt(
        "1\n00:00:00,000 --> 00:00:12,000\nABCDEFGHIJKL <-0.8->\n"
    )
    split = m._merge_sentence_cues(cues)
    assert [cue.text for cue in split] == ["ABCDE", "FGHIJ", "KL"]
    assert [cue.avg_logprob for cue in split] == [-0.8, -0.8, -0.8]
    assert [(cue.start, cue.end) for cue in split] == [
        ("00:00:00,000", "00:00:05,000"),
        ("00:00:05,000", "00:00:10,000"),
        ("00:00:10,000", "00:00:12,000"),
    ]


def test_sentence_merge_does_not_exceed_five_seconds():
    m = load("02_transcribe_srt")
    cues = parse_srt(
        "1\n00:00:00,000 --> 00:00:03,000\n今日は\n\n"
        "2\n00:00:03,100 --> 00:00:06,000\nいい天気\n"
    )
    merged = m._merge_sentence_cues(cues)
    assert [cue.text for cue in merged] == ["今日は", "いい天気"]
    assert all(
        m._timestamp_seconds(cue.end) - m._timestamp_seconds(cue.start) <= 5
        for cue in merged
    )


def test_short_adjacent_cues_merge_and_keep_time_range():
    m = load("04_proofread_srt")
    cues = parse_srt(
        "1\n00:00:00,000 --> 00:00:00,500\nうん\n\n"
        "2\n00:00:00,700 --> 00:00:01,200\nそう\n\n"
        "3\n00:00:01,300 --> 00:00:01,800\nはい\n"
    )
    merged = m._merge_short_cues(cues, ["嗯", "是", "好"])
    assert len(merged) == 1
    assert merged[0].start == "00:00:00,000"
    assert merged[0].end == "00:00:01,800"
    assert merged[0].text == "うん / そう / はい\n嗯 / 是 / 好"


def test_short_cues_do_not_merge_across_large_gap():
    m = load("04_proofread_srt")
    cues = parse_srt(
        "1\n00:00:00,000 --> 00:00:00,500\nうん\n\n"
        "2\n00:00:02,000 --> 00:00:02,500\nそう\n"
    )
    assert len(m._merge_short_cues(cues, ["嗯", "是"])) == 2


def test_missing_key_fails(tmp_path, monkeypatch):
    m = load("04_proofread_srt")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(m, "load_env", lambda: None)
    f = tmp_path / "a.ja.srt"
    f.write_text(SRT, encoding="utf-8")
    with pytest.raises(SystemExit):
        m.main([str(f)])


def test_translate_writes_chinese_srt(tmp_path, monkeypatch):
    m = load("03_translate_srt")
    monkeypatch.setattr(m, "ollama_translate", lambda b, h, mo, **kwargs: ["初"] * len(b))
    f = tmp_path / "a.ja.srt"
    f.write_text(SRT, encoding="utf-8")
    m.main([str(f)])
    out = parse_srt((tmp_path / "a.zh.srt").read_text(encoding="utf-8"))
    assert [c.text for c in out] == ["初", "初"]
    assert out[1].start == "00:00:02,000"


def test_local_translation_does_not_receive_confidence_suffix(tmp_path, monkeypatch):
    m = load("03_translate_srt")
    source = tmp_path / "confidence.ja.srt"
    source.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nこんにちは <-0.8->\n",
        encoding="utf-8",
    )
    sent = []
    monkeypatch.setattr(
        m, "ollama_translate",
        lambda batch, *args, **kwargs: sent.extend(batch) or ["你好"],
    )

    m.main([str(source)])

    assert sent == ["こんにちは"]
    translated = parse_srt(
        (tmp_path / "confidence.zh.srt").read_text(encoding="utf-8")
    )[0]
    assert translated.text == "你好"
    assert translated.avg_logprob == pytest.approx(-0.8)


def test_quality_gate_marks_suspect_and_hard_translation_rows():
    m = load("04_proofread_srt")
    flags = m._quality_gate(
        [
            "こんにちは",
            "これはそれ",
            "彼は2025年です",
            "何ですか",
            "何ですか",
        ],
        [
            "你好啊好吗",
            "这是那个",
            "他是2024年了",
            "嗯，好的",
            "嗯，没错",
        ],
    )

    assert flags == {2: "hard", 3: "suspect", 4: "suspect", 5: "suspect"}


def test_quality_gate_detects_missing_glossary_translation():
    m = load("04_proofread_srt")
    flags = m._quality_gate(
        ["番組名について話します", "番組名について話します"],
        ["节目名称的话题", "节目的话题"],
        {"番組名": "节目名称"},
    )
    assert flags == {2: "suspect"}


def test_quality_gate_marks_low_confidence_and_keeps_hard_priority():
    m = load("04_proofread_srt")

    flags = m._quality_gate(
        ["普通日文", "これはそれ"],
        ["普通翻译", "这是那个"],
        confidence_scores=[-0.75, -0.9],
    )

    assert flags == {1: "suspect", 2: "hard"}


def test_default_confidence_gate_uses_85_percent_threshold():
    m = load("04_proofread_srt")
    assert m.DEFAULT_CONFIDENCE_THRESHOLD == pytest.approx(math.log(0.85))
    assert m._quality_gate(
        ["普通日文", "普通日文"],
        ["普通中文", "普通中文"],
        confidence_scores=[-0.2, -0.15],
    ) == {1: "suspect"}


def test_deepseek_refinement_includes_only_target_pairs_and_adjacent_japanese_context(
        monkeypatch):
    m = load("04_proofread_srt")
    calls = []

    def request(system_prompt, user_prompt, base_url, key, model, thinking,
                max_tokens, token_usage):
        calls.append((user_prompt, model, thinking))
        return "2\t精修一"

    monkeypatch.setattr(m, "_deepseek_request", request)
    result = m.deepseek_refine(
        ["相邻句一", "待润色原文", "相邻句三"],
        ["上下文译文一", "待润色初译", "上下文译文三"],
        [2], "url", "key", "deepseek-chat", max_lines=1,
        thinking="disabled",
    )

    assert result == {2: "精修一"}
    assert len(calls) == 1
    assert all(model == "deepseek-chat" and thinking == "disabled"
               for _, model, thinking in calls)
    prompt = calls[0][0]
    assert "2\t待润色原文\t待润色初译" in prompt
    assert "相邻日文上下文（仅供理解，不要翻译或输出）" in prompt
    assert "1\t相邻句一" in prompt
    assert "3\t相邻句三" in prompt
    assert "上下文译文一" not in prompt
    assert "上下文译文三" not in prompt


def test_deepseek_refinement_receives_confidence_as_separate_metadata(monkeypatch):
    m = load("04_proofread_srt")
    calls = []

    def request(system_prompt, user_prompt, base_url, key, model, thinking,
                max_tokens, token_usage):
        calls.append(user_prompt)
        return "1\t精修"

    monkeypatch.setattr(m, "_deepseek_request", request)
    m.deepseek_refine(
        ["原文"], ["初译"], [1], "url", "key", "model",
        romaji_batch=["genbun"], confidence_scores=[-0.75],
    )

    assert "原文\tgenbun\t初译\t47.2%" in calls[0]
    assert "<-0.75->" not in calls[0]


def test_final_output_marks_low_confidence_but_not_raw_logprob(tmp_path, monkeypatch):
    m = load("04_proofread_srt")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    source = tmp_path / "marked.ja.srt"
    source.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n普通日文 <-0.75->\n",
        encoding="utf-8",
    )
    zh = tmp_path / "marked.zh.srt"
    zh.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n普通翻译\n",
        encoding="utf-8",
    )
    requests = []

    def refine(ja, drafts, numbers, *args):
        requests.append((ja, numbers))
        return {number: "精修译文" for number in numbers}

    monkeypatch.setattr(m, "deepseek_refine", refine)
    m.main([
        str(source), "--zh-srt", str(zh), "--suspects-only",
        "--no-merge-short", "--no-token-usage",
    ])

    output_text = (tmp_path / "marked.zh-ja.srt").read_text(encoding="utf-8")
    assert requests == [(["普通日文"], [1])]
    assert "<low conf：47.2%（概率近似）>" in output_text
    assert "<-0.75->" not in output_text
    assert parse_srt(output_text)[0].text == (
        "普通日文<low conf：47.2%（概率近似）>\n精修译文"
    )


def test_suspects_only_routes_all_flags_to_chat_and_preserves_other_rows(
        tmp_path, monkeypatch):
    m = load("04_proofread_srt")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    japanese = (
        "1\n00:00:00,000 --> 00:00:01,000\nこんにちは\n\n"
        "2\n00:00:01,100 --> 00:00:02,000\nこれはそれ\n\n"
        "3\n00:00:02,100 --> 00:00:03,000\n彼は2025年です\n"
    )
    drafts = (
        "1\n00:00:00,000 --> 00:00:01,000\n你好啊好吗\n\n"
        "2\n00:00:01,100 --> 00:00:02,000\n这是那个\n\n"
        "3\n00:00:02,100 --> 00:00:03,000\n他是2024年了\n"
    )
    source = tmp_path / "a.ja.srt"
    source.write_text(japanese, encoding="utf-8")
    zh = tmp_path / "a.zh.srt"
    zh.write_text(drafts, encoding="utf-8")
    calls = []

    def refine(ja, zh, numbers, base_url, key, model, *args):
        calls.append((numbers, model, args[1]))
        return {
            number: "Chat精修"
            for number in numbers
        }

    monkeypatch.setattr(m, "deepseek_refine", refine)
    m.main([
        str(source), "--zh-srt", str(zh), "--suspects-only",
        "--no-merge-short", "--no-token-usage",
    ])

    output = parse_srt((tmp_path / "a.zh-ja.srt").read_text(encoding="utf-8"))
    assert calls == [([2, 3], "deepseek-v4-pro", "disabled")]
    assert [cue.text for cue in output] == [
        "こんにちは\n你好啊好吗",
        "これはそれ\nChat精修",
        "彼は2025年です\nChat精修",
    ]


def test_proofread_mocked(tmp_path, monkeypatch):
    m = load("04_proofread_srt")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    custom_prompts = []

    def proofread(j, z, u, k, mo, *args):
        assert z == ["初", "初"]
        custom_prompts.append(args[-4])
        return [
            m._ProofreadRow((i,), source, "校")
            for i, source in enumerate(j, 1)
        ]

    monkeypatch.setattr(m, "deepseek_proofread", proofread)
    f = tmp_path / "a.ja.srt"
    f.write_text(SRT, encoding="utf-8")
    (tmp_path / "a.zh.srt").write_text(SRT.replace("こんにちは", "初").replace("ありがとう", "初"),
                                       encoding="utf-8")
    prompt_file = tmp_path / "proofreading.md"
    prompt_file.write_text("## 专名\n统一人物译名", encoding="utf-8")
    m.main([str(f), "--zh-srt", str(tmp_path / "a.zh.srt"),
            "--proofread-prompt-file", str(prompt_file)])
    out = parse_srt((tmp_path / "a.zh-ja.srt").read_text(encoding="utf-8"))
    assert len(out) == 1
    assert out[0].text == "こんにちは / ありがとう\n校 / 校"
    assert custom_prompts == ["## 专名\n统一人物译名"]


def test_direct_translate_mocked(tmp_path, monkeypatch):
    m = load("04_proofread_srt")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    monkeypatch.setattr(m, "deepseek_translate", lambda j, *args: ["直译"] * len(j))
    f = tmp_path / "a.ja.srt"
    f.write_text(SRT, encoding="utf-8")
    m.main([str(f)])
    out = parse_srt((tmp_path / "a.zh-ja.srt").read_text(encoding="utf-8"))
    assert len(out) == 1
    assert out[0].text == "こんにちは / ありがとう\n直译 / 直译"


def test_proofread_custom_prompt_is_included():
    m = load("04_proofread_srt")
    prompts = []

    def request(system_prompt, user_prompt, *args):
        prompts.append((system_prompt, user_prompt))
        return "1\tこんにちは\t校对后"

    m._deepseek_request = request
    rows = m.deepseek_proofread(
        ["こんにちは"], ["你好"], "url", "key", "model",
        custom_prompt="## 专名\n将某人名统一译为小明",
    )
    assert [(row.indices, row.japanese, row.chinese) for row in rows] == [
        ((1,), "こんにちは", "校对后")
    ]
    system_prompt, user_prompt = prompts[0]
    assert "你是日语广播字幕校对员" in system_prompt
    assert "用户自定义校对要求：\n## 专名\n将某人名统一译为小明" in user_prompt
    assert "编号范围" in system_prompt
    assert "<DELETE>" in system_prompt


def test_deepseek_api_receives_separate_system_and_user_messages(monkeypatch):
    m = load("04_proofread_srt")
    payloads = []
    token_usage = m._TokenUsage()

    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [{"finish_reason": "stop",
                             "message": {"content": "1\t译文"}}],
                "usage": {"prompt_tokens": 37, "completion_tokens": 11},
            }

    def post(url, headers, json, timeout):
        payloads.append(json)
        return Response()

    monkeypatch.setattr(m.requests, "post", post)
    assert m._deepseek_request(
        "system rules", "this task", "https://example.test", "key",
        "deepseek-v4-pro", "enabled", 100, token_usage,
    ) == "1\t译文"
    assert token_usage.input_tokens == 37
    assert token_usage.output_tokens == 11
    assert token_usage.cost == pytest.approx(0.0001101)
    assert payloads[0]["messages"] == [
        {"role": "system", "content": "system rules"},
        {"role": "user", "content": "this task"},
    ]


def test_token_usage_cost_includes_input_and_output_and_marks_missing(capsys):
    m = load("04_proofread_srt")
    usage = m._TokenUsage()
    usage.record({"prompt_tokens": 12, "completion_tokens": 3})
    usage.record({})
    usage.record({"prompt_tokens": 8, "completion_tokens": 2})
    usage.report()
    assert usage.input_tokens == 20
    assert usage.output_tokens == 5
    assert usage.cost == pytest.approx(0.000051)
    assert usage.missing_input_requests == 1
    assert usage.missing_output_requests == 1
    output = capsys.readouterr().err
    assert "输入 20，输出 5" in output
    assert "部分响应未提供 token" in output


def test_token_usage_does_not_apply_legacy_prices_to_chat_or_reasoner(capsys):
    m = load("04_proofread_srt")
    usage = m._TokenUsage()
    usage.record({"prompt_tokens": 12, "completion_tokens": 3}, "deepseek-chat")
    usage.record({"prompt_tokens": 8, "completion_tokens": 2}, "deepseek-reasoner")
    usage.report()

    assert usage.input_tokens == 20
    assert usage.output_tokens == 5
    assert usage.cost == 0
    report = capsys.readouterr().err
    assert "deepseek-chat: 输入 12、输出 3" in report
    assert "deepseek-reasoner: 输入 8、输出 2" in report
    assert "未估价模型不计入费用" in report


def test_token_counter_estimates_input_and_output_by_batches():
    m = load("05_token_counter")
    cues = parse_srt(SRT)

    translated = m.estimate_job(cues, batch_size=1)
    proofread = m.estimate_job(cues, ["你好", "谢谢"], batch_size=1)

    assert translated["mode"] == "translate"
    assert translated["requests"] == 2
    assert translated["input_tokens"] > 0
    assert translated["output_tokens"] > 0
    assert proofread["mode"] == "proofread"
    assert proofread["requests"] == 2
    assert proofread["input_tokens"] > translated["input_tokens"]
    assert proofread["output_tokens"] > translated["output_tokens"]


def test_token_counter_rejects_mismatched_srt_rows():
    m = load("05_token_counter")
    with pytest.raises(ValueError, match="条数"):
        m.estimate_job(parse_srt(SRT), ["只有一条"])


def test_token_counter_estimator_handles_cjk_and_ascii():
    m = load("05_token_counter")
    assert m.estimate_tokens("你好") == 2
    assert m.estimate_tokens("hello") == 2


def test_deepseek_requests_only_targeted_rows():
    m = load("04_proofread_srt")
    prompts = []

    def request(system_prompt, user_prompt, *args):
        prompts.append((system_prompt, user_prompt))
        if len(prompts) == 1:
            return "1\t结果\n2\t结果"
        return "3\t结果"

    m._deepseek_request = request
    assert m.deepseek_translate(
        ["甲", "乙", "丙"], "url", "key", "model", max_lines=2,
    ) == ["结果", "结果", "结果"]
    assert "1\t甲" in prompts[0][1] and "2\t乙" in prompts[0][1]
    assert "3\t丙" not in prompts[0][1]
    assert "3\t丙" in prompts[1][1]


def test_proofread_prompt_sends_only_requested_source_and_draft_rows():
    m = load("04_proofread_srt")
    prompts = []

    def request(system_prompt, user_prompt, *args):
        prompts.append((system_prompt, user_prompt))
        if len(prompts) == 1:
            return "1\t甲\t校对后\n2\t乙\t校对后"
        return "3\t丙\t校对后"

    m._deepseek_request = request
    rows = m.deepseek_proofread(
        ["甲", "乙", "丙"], ["初译甲", "初译乙", "初译丙"],
        "url", "key", "model", max_lines=2,
    )
    assert [row.indices for row in rows] == [(1,), (2,), (3,)]
    assert "1\t甲\t初译甲" in prompts[0][1]
    assert "2\t乙\t初译乙" in prompts[0][1]
    assert "3\t丙\t初译丙" not in prompts[0][1]
    assert "3\t丙\t初译丙" in prompts[1][1]


def test_proofread_parser_supports_merge_delete_and_missing_row_retry():
    m = load("04_proofread_srt")
    rows, missing = m._parse_proofread_rows(
        "1-2\t合并后的日文\t合并后的中文\n3\t<DELETE>\t<DELETE>",
        [1, 2, 3, 4],
    )
    assert [row.indices for row in rows] == [(1, 2), (3,)]
    assert rows[1].deleted
    assert missing == {4}

    calls = []

    def request(numbers):
        calls.append(numbers)
        if len(calls) == 1:
            return "1-2\t合并\tmerged"
        return "3\t日文\t中文"

    completed = m._complete_proofread([1, 2, 3], request)
    assert [row.indices for row in completed] == [(1, 2), (3,)]
    assert calls == [[1, 2, 3], [3]]


def test_proofread_parser_rejects_unrequested_or_huge_ranges():
    m = load("04_proofread_srt")
    with pytest.raises(ValueError, match="未指定"):
        m._parse_proofread_rows("1-999999999\t日文\t中文", [1])


def test_proofread_rows_become_merged_and_deleted_timed_cues():
    m = load("04_proofread_srt")
    cues = parse_srt(
        "1\n00:00:00,000 --> 00:00:01,000\n誤認\n\n"
        "2\n00:00:01,100 --> 00:00:02,000\n正しい続き\n\n"
        "3\n00:00:02,100 --> 00:00:03,000\n幻聴\n"
    )
    rows = [
        m._ProofreadRow((1, 2), "正しい文章", "正确的一句"),
        m._ProofreadRow((3,), "<DELETE>", "<DELETE>", deleted=True),
    ]
    source, translations = m._proofread_results_to_cues(cues, rows)
    assert len(source) == len(translations) == 1
    assert source[0].start == "00:00:00,000"
    assert source[0].end == "00:00:02,000"
    assert source[0].text == "正しい文章"
    assert translations == ["正确的一句"]


def test_proofread_main_writes_corrected_merged_cue_and_omits_deleted_cue(
        tmp_path, monkeypatch):
    m = load("04_proofread_srt")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    japanese = (
        "1\n00:00:00,000 --> 00:00:01,000\n認識誤り\n\n"
        "2\n00:00:01,100 --> 00:00:02,000\n続き\n\n"
        "3\n00:00:02,100 --> 00:00:03,000\n無音幻聴\n"
    )
    f = tmp_path / "a.ja.srt"
    f.write_text(japanese, encoding="utf-8")
    zh = tmp_path / "a.zh.srt"
    zh.write_text(
        japanese.replace("認識誤り", "初译一")
        .replace("続き", "初译二")
        .replace("無音幻聴", "初译三"),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        m, "deepseek_proofread",
        lambda *args: [
            m._ProofreadRow((1, 2), "修正後の日文", "合并后的中文"),
            m._ProofreadRow((3,), "<DELETE>", "<DELETE>", deleted=True),
        ],
    )
    m.main([str(f), "--zh-srt", str(zh), "--no-merge-short"])
    out = parse_srt((tmp_path / "a.zh-ja.srt").read_text(encoding="utf-8"))
    assert len(out) == 1
    assert out[0].start == "00:00:00,000"
    assert out[0].end == "00:00:02,000"
    assert out[0].text == "修正後の日文\n合并后的中文"
