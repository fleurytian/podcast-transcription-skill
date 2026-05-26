#!/usr/bin/env python3
"""Local podcast transcription helper for continuous speaker-aware Markdown."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_MODEL = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
DEFAULT_SPEECHBRAIN = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_SPEECHBRAIN_CACHE = "~/.cache/speechbrain/spkrec-ecapa-voxceleb"


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None
    score: float | None = None


def parse_time(value: str) -> float:
    parts = value.strip().split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    raise ValueError(f"Bad time value: {value}")


def parse_speaker_ranges(values: list[str]) -> dict[str, list[tuple[float, float]]]:
    ranges: dict[str, list[tuple[float, float]]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=start:end,start:end, got: {value}")
        speaker, raw_ranges = value.split("=", 1)
        speaker = speaker.strip()
        parsed: list[tuple[float, float]] = []
        for raw_range in raw_ranges.split(","):
            if not raw_range.strip():
                continue
            if "-" in raw_range:
                start_raw, end_raw = raw_range.split("-", 1)
            else:
                start_raw, end_raw = raw_range.split(":", 1)
            start, end = parse_time(start_raw), parse_time(end_raw)
            if end <= start:
                raise ValueError(f"Anchor range end must be after start: {raw_range}")
            parsed.append((start, end))
        if parsed:
            ranges.setdefault(speaker, []).extend(parsed)
    return ranges


def fmt_time(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def transcribe(audio: Path, model_name: str, language: str) -> list[Segment]:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name)
    segments, _info = model.transcribe(
        str(audio),
        language=language,
        vad_filter=True,
        word_timestamps=False,
    )
    result: list[Segment] = []
    for seg in segments:
        text = normalize_text(seg.text)
        if text:
            result.append(Segment(start=float(seg.start), end=float(seg.end), text=text))
    return result


def load_audio_16k(audio: Path):
    import torch
    import torchaudio

    waveform, sr = torchaudio.load(str(audio))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    return waveform, 16000, torch


def cosine(a, b) -> float:
    import numpy as np

    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)


def speaker_embeddings(audio: Path, ranges: dict[str, list[tuple[float, float]]], source: str, savedir: str):
    import numpy as np
    from speechbrain.inference.speaker import EncoderClassifier

    waveform, sr, torch = load_audio_16k(audio)
    classifier = EncoderClassifier.from_hparams(source=source, savedir=str(Path(savedir).expanduser()))

    def embed_range(start: float, end: float):
        lo = max(0, int(start * sr))
        hi = min(waveform.shape[1], int(end * sr))
        if hi <= lo:
            return None
        clip = waveform[:, lo:hi]
        with torch.no_grad():
            emb = classifier.encode_batch(clip).squeeze().detach().cpu().numpy()
        return emb

    anchors = {}
    for speaker, speaker_ranges in ranges.items():
        embs = [embed_range(start, end) for start, end in speaker_ranges]
        embs = [emb for emb in embs if emb is not None]
        if embs:
            anchors[speaker] = np.mean(embs, axis=0)
    return anchors, waveform, sr, classifier, torch


def diarize(
    audio: Path,
    segments: list[Segment],
    ranges: dict[str, list[tuple[float, float]]],
    source: str,
    savedir: str,
    min_clip: float,
    max_clip: float,
) -> list[Segment]:
    anchors, waveform, sr, classifier, torch = speaker_embeddings(audio, ranges, source, savedir)
    if not anchors:
        return segments

    def segment_clip(seg: Segment):
        duration = max(0.1, seg.end - seg.start)
        if duration > max_clip:
            center = (seg.start + seg.end) / 2
            start = center - max_clip / 2
            end = center + max_clip / 2
        else:
            start, end = seg.start, seg.end
        if end - start < min_clip:
            pad = (min_clip - (end - start)) / 2
            start -= pad
            end += pad
        lo = max(0, int(start * sr))
        hi = min(waveform.shape[1], int(end * sr))
        return waveform[:, lo:hi]

    assigned: list[Segment] = []
    for seg in segments:
        clip = segment_clip(seg)
        if clip.shape[1] == 0:
            assigned.append(seg)
            continue
        with torch.no_grad():
            emb = classifier.encode_batch(clip).squeeze().detach().cpu().numpy()
        scores = {speaker: cosine(emb, anchor) for speaker, anchor in anchors.items()}
        speaker, score = max(scores.items(), key=lambda item: item[1])
        assigned.append(Segment(seg.start, seg.end, seg.text, speaker=speaker, score=score))
    return smooth_short_islands(assigned)


def smooth_short_islands(segments: list[Segment], max_duration: float = 2.2) -> list[Segment]:
    smoothed = [Segment(s.start, s.end, s.text, s.speaker, s.score) for s in segments]
    for i in range(1, len(smoothed) - 1):
        prev_s = smoothed[i - 1].speaker
        next_s = smoothed[i + 1].speaker
        cur = smoothed[i]
        if prev_s and prev_s == next_s and cur.speaker != prev_s and (cur.end - cur.start) <= max_duration:
            cur.speaker = prev_s
    return smoothed


def merge_turns(segments: Iterable[Segment], fallback_speaker: str) -> list[Segment]:
    turns: list[Segment] = []
    for seg in segments:
        speaker = seg.speaker or fallback_speaker
        text = normalize_text(seg.text)
        if not text:
            continue
        if turns and turns[-1].speaker == speaker:
            turns[-1].end = max(turns[-1].end, seg.end)
            turns[-1].text = normalize_text(turns[-1].text + " " + text)
        else:
            turns.append(Segment(seg.start, seg.end, text, speaker=speaker, score=seg.score))
    return turns


def write_raw_markdown(path: Path, title: str, source: str, segments: list[Segment]) -> None:
    lines = [f"# {title}", "", f"来源：{source}", "", "## Raw Transcript", ""]
    for seg in segments:
        lines.append(f"[{fmt_time(seg.start)}] {seg.text}")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_continuous_markdown(
    path: Path,
    title: str,
    podcast: str,
    source: str,
    turns: list[Segment],
    model_name: str,
    diarization_note: str,
) -> None:
    lines = [
        f"# {title}",
        "",
        f"来源：{source}",
        f"播客：{podcast}",
        f"转写：faster-whisper {model_name}",
        f"说话人区分：{diarization_note}",
        "规则：同一说话人连续发言合并为一个段落组，只在说话人变化处显示时间。",
        "",
        "## 全文逐字稿",
        "",
    ]
    for turn in turns:
        lines.append(f"### [{fmt_time(turn.start)}] {turn.speaker}")
        lines.append("")
        lines.append(turn.text)
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def check_no_adjacent_same_speaker(markdown: Path) -> None:
    headings = []
    pattern = re.compile(r"^### \[[^\]]+\] (.+)$")
    for line in markdown.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            headings.append(match.group(1).strip())
    for left, right in zip(headings, headings[1:]):
        if left == right:
            raise SystemExit(f"Adjacent same-speaker headings found in {markdown}: {left}")


def dump_json(path: Path, segments: list[Segment]) -> None:
    data = [segment.__dict__ for segment in segments]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--out-dir", default="transcripts", type=Path)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--podcast", default="Unknown podcast")
    parser.add_argument("--source", default="")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--speaker-ranges", action="append", default=[], help="NAME=start:end,start:end")
    parser.add_argument("--single-speaker", default="")
    parser.add_argument("--speechbrain-source", default=DEFAULT_SPEECHBRAIN)
    parser.add_argument("--speechbrain-savedir", default=DEFAULT_SPEECHBRAIN_CACHE)
    parser.add_argument("--min-clip", default=1.2, type=float)
    parser.add_argument("--max-clip", default=8.0, type=float)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    segments = transcribe(args.audio, args.model, args.language)
    raw_json = args.out_dir / f"{args.slug}_raw_segments.json"
    raw_md = args.out_dir / f"{args.slug}_raw_逐字稿.md"
    dump_json(raw_json, segments)
    write_raw_markdown(raw_md, args.title, args.source, segments)

    ranges = parse_speaker_ranges(args.speaker_ranges)
    diarization_note = "single speaker"
    if args.single_speaker:
        for seg in segments:
            seg.speaker = args.single_speaker
    elif ranges:
        segments = diarize(
            args.audio,
            segments,
            ranges,
            args.speechbrain_source,
            args.speechbrain_savedir,
            args.min_clip,
            args.max_clip,
        )
        diarization_note = "SpeechBrain ECAPA voice embeddings from user-provided anchor ranges; labels may contain diarization uncertainty."
    else:
        for seg in segments:
            seg.speaker = "Speaker 1"
        diarization_note = "no speaker anchors provided; output kept as one speaker."

    diarized_json = args.out_dir / f"{args.slug}_diarized_segments.json"
    dump_json(diarized_json, segments)

    turns = merge_turns(segments, fallback_speaker=args.single_speaker or "Speaker 1")
    final_md = args.out_dir / f"{args.slug}_声纹区分_连续段落版.md"
    write_continuous_markdown(final_md, args.title, args.podcast, args.source, turns, args.model, diarization_note)
    check_no_adjacent_same_speaker(final_md)
    print(final_md)


if __name__ == "__main__":
    main()
