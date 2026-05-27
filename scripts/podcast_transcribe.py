#!/usr/bin/env python3
"""Local podcast transcription helper for continuous speaker-aware Markdown."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
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


def download_youtube_audio(url: str, out_dir: Path, slug: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    output_template = out_dir / f"{slug}.%(ext)s"
    before = {p.resolve() for p in out_dir.glob(f"{slug}.*")}
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-f",
        "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio",
        "-o",
        str(output_template),
        url,
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise SystemExit("yt-dlp is required for --youtube-url. Install it with: pip install yt-dlp") from exc

    candidates = [p for p in out_dir.glob(f"{slug}.*") if p.resolve() not in before]
    if not candidates:
        candidates = list(out_dir.glob(f"{slug}.*"))
    candidates = [p for p in candidates if p.suffix.lower() in {".m4a", ".webm", ".mp3", ".opus", ".wav"}]
    if not candidates:
        raise SystemExit(f"yt-dlp finished but no audio file was found for slug {slug!r}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


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
    from faster_whisper.audio import decode_audio

    waveform = torch.from_numpy(decode_audio(str(audio), sampling_rate=16000)).float().unsqueeze(0)
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


def l2_normalize(matrix):
    import numpy as np

    matrix = np.nan_to_num(matrix.astype("float32"), nan=0.0, posinf=0.0, neginf=0.0)
    denom = np.linalg.norm(matrix, axis=1, keepdims=True)
    denom[denom == 0] = 1
    return matrix / denom


def auto_diarize(
    audio: Path,
    segments: list[Segment],
    n_speakers: int,
    source: str,
    savedir: str,
    min_clip: float,
    max_clip: float,
) -> list[Segment]:
    import numpy as np
    from sklearn.cluster import KMeans
    from speechbrain.inference.speaker import EncoderClassifier

    if n_speakers <= 1 or len(segments) < 2:
        return [Segment(seg.start, seg.end, seg.text, speaker="Speaker 1", score=seg.score) for seg in segments]

    waveform, sr, torch = load_audio_16k(audio)
    classifier = EncoderClassifier.from_hparams(source=source, savedir=str(Path(savedir).expanduser()))

    embeddings = []
    emb_indices = []
    for idx, seg in enumerate(segments):
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
        if hi <= lo:
            continue
        clip = waveform[:, lo:hi]
        with torch.no_grad():
            emb = classifier.encode_batch(clip).squeeze().detach().cpu().numpy()
        embeddings.append(emb)
        emb_indices.append(idx)

    if len(embeddings) < n_speakers:
        return [Segment(seg.start, seg.end, seg.text, speaker="Speaker 1", score=seg.score) for seg in segments]

    labels = KMeans(n_clusters=n_speakers, n_init=25, random_state=17).fit_predict(l2_normalize(np.vstack(embeddings)))
    full_labels: list[int | None] = [None] * len(segments)
    for idx, label in zip(emb_indices, labels):
        full_labels[idx] = int(label)

    last_label = int(labels[0])
    for idx, label in enumerate(full_labels):
        if label is None:
            full_labels[idx] = last_label
        else:
            last_label = label

    first_seen: list[int] = []
    for label in full_labels:
        label = int(label)
        if label not in first_seen:
            first_seen.append(label)
    names = {label: f"Speaker {i + 1}" for i, label in enumerate(first_seen)}

    assigned = [
        Segment(seg.start, seg.end, seg.text, speaker=names[int(label)], score=seg.score)
        for seg, label in zip(segments, full_labels)
    ]
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
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--audio", type=Path)
    source_group.add_argument("--youtube-url")
    parser.add_argument("--out-dir", default="transcripts", type=Path)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--podcast", default="Unknown podcast")
    parser.add_argument("--source", default="")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--speaker-ranges", action="append", default=[], help="NAME=start:end,start:end")
    parser.add_argument("--auto-speakers", default=0, type=int, help="Automatically cluster this many speakers when no anchors are provided")
    parser.add_argument("--single-speaker", default="")
    parser.add_argument("--speechbrain-source", default=DEFAULT_SPEECHBRAIN)
    parser.add_argument("--speechbrain-savedir", default=DEFAULT_SPEECHBRAIN_CACHE)
    parser.add_argument("--min-clip", default=1.2, type=float)
    parser.add_argument("--max-clip", default=8.0, type=float)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audio = args.audio
    if args.youtube_url:
        audio = download_youtube_audio(args.youtube_url, args.out_dir, args.slug)

    source = args.source or args.youtube_url or str(audio)
    segments = transcribe(audio, args.model, args.language)
    raw_json = args.out_dir / f"{args.slug}_raw_segments.json"
    raw_md = args.out_dir / f"{args.slug}_raw_逐字稿.md"
    dump_json(raw_json, segments)
    write_raw_markdown(raw_md, args.title, source, segments)

    ranges = parse_speaker_ranges(args.speaker_ranges)
    diarization_note = "single speaker"
    if args.single_speaker:
        for seg in segments:
            seg.speaker = args.single_speaker
    elif ranges:
        segments = diarize(
            audio,
            segments,
            ranges,
            args.speechbrain_source,
            args.speechbrain_savedir,
            args.min_clip,
            args.max_clip,
        )
        diarization_note = "SpeechBrain ECAPA voice embeddings from user-provided anchor ranges; labels may contain diarization uncertainty."
    elif args.auto_speakers > 1:
        segments = auto_diarize(
            audio,
            segments,
            args.auto_speakers,
            args.speechbrain_source,
            args.speechbrain_savedir,
            args.min_clip,
            args.max_clip,
        )
        diarization_note = f"SpeechBrain ECAPA voice embeddings with automatic {args.auto_speakers}-speaker clustering; labels are generic and may contain diarization uncertainty."
    else:
        for seg in segments:
            seg.speaker = "Speaker 1"
        diarization_note = "no speaker anchors provided; output kept as one speaker."

    diarized_json = args.out_dir / f"{args.slug}_diarized_segments.json"
    dump_json(diarized_json, segments)

    turns = merge_turns(segments, fallback_speaker=args.single_speaker or "Speaker 1")
    final_md = args.out_dir / f"{args.slug}_声纹区分_连续段落版.md"
    write_continuous_markdown(final_md, args.title, args.podcast, source, turns, args.model, diarization_note)
    check_no_adjacent_same_speaker(final_md)
    print(final_md)


if __name__ == "__main__":
    main()
