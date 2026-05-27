---
name: podcast-transcription
description: Use when turning podcast, YouTube, or local audio links into transcripts, especially 小宇宙/Xiaoyuzhou episodes, YouTube videos, faster-whisper, 逐字稿, 声纹区分, 说话人区分, 连续段落版, transcript zip packages, or transcript folder cleanup.
---

# Podcast Transcription

## Default Promise

Produce user-facing podcast transcripts as continuous paragraph Markdown with voiceprint-level speaker labels whenever feasible. A typical request means: fetch the episode audio, transcribe with local faster-whisper, diarize by speaker voice embeddings, merge adjacent speech by the same speaker, and package only the readable final files.

## Local Setup

- Workspace default: the current project folder.
- Transcript output folder: `transcripts/`.
- Preferred Python: a virtual environment that has `faster-whisper`, `speechbrain`, `torch`, `torchaudio`, and `numpy` installed.
- Optional download helper for YouTube: `yt-dlp`.
- Preferred ASR model: `mobiuslabsgmbh/faster-whisper-large-v3-turbo`.
- Preferred speaker embedding model: `speechbrain/spkrec-ecapa-voxceleb`.
- SpeechBrain cache: use the default Hugging Face/SpeechBrain cache, or pass a local cache path to the helper script.

Search locally for existing models before installing or downloading anything new.

## Xiaoyuzhou Workflow

1. Resolve metadata and audio:
   - Start from the episode page title and podcast name.
   - Prefer RSS/enclosure URLs when available. Xiaoyuzhou pages may omit `__NEXT_DATA__`; try RSSHub mirrors or the podcast's RSS feed.
   - For older episodes outside current RSS pages, search Apple Podcasts/iTunes metadata or known RSS feeds, then match by episode id/title/date.
2. Download audio into `transcripts/` with an episode-id filename while working.
3. Transcribe locally with faster-whisper:
   - language `zh`
   - `vad_filter=True`
   - save raw JSON during work for recovery and checks.
4. Diarize:
   - Prefer voiceprint-level labels using SpeechBrain ECAPA embeddings.
   - Build anchors from clear self-introductions, host-only openings, guest introductions, or long single-speaker answers.
   - For solo episodes, use one speaker instead of inventing multiple speakers.
   - If real names are uncertain, use `主持人`, `嘉宾`, `Speaker 1`, etc., and note uncertainty.
5. Write final Markdown as continuous paragraphs:
   - Only place a time label when the speaker changes.
   - Never split one speaker's continuous speech just because time advanced or a segment boundary occurred.
   - Merge adjacent same-speaker segments before formatting.
   - Keep the transcript text verbatim enough to be useful; do not summarize inside the transcript.
6. When multiple episodes are requested, create one zip containing only final continuous Markdown files unless the user asks for raw files too.

## YouTube Workflow

YouTube links can use the same transcription and diarization path:

1. Download the audio track with `yt-dlp` using the best available audio-only format.
2. Save the file under `transcripts/` or another working folder.
3. Run local faster-whisper on that audio file.
4. Use SpeechBrain ECAPA embeddings for speaker separation:
   - Prefer explicit voiceprint anchors when speaker names matter.
   - If anchors are not available, use automatic speaker clustering and label speakers generically.
5. Format exactly like podcast transcripts: continuous paragraphs, and only a new timestamp when the speaker changes.

## Output Contract

Final transcript filenames should include `连续段落` and, when applicable, `声纹区分`, for example:

```text
transcripts/{slug}_声纹区分_连续段落版.md
```

Markdown structure:

```markdown
# {episode title}

来源：{url}
播客：{podcast name}
转写：faster-whisper large-v3-turbo
说话人区分：SpeechBrain ECAPA voice embeddings; labels may contain diarization uncertainty.
规则：同一说话人连续发言合并为一个段落组，只在说话人变化处显示时间。

## 全文逐字稿

### [00:00:12] 主持人

...

### [00:03:41] 嘉宾

...
```

The same speaker must not appear in two adjacent sections. If a formatter creates adjacent identical speaker headings, merge them.

## Useful Script

Use `scripts/podcast_transcribe.py` when starting from a local audio file, a YouTube URL, and known metadata. It can:

- run faster-whisper
- download YouTube audio with `yt-dlp`
- assign speakers from voiceprint anchor ranges
- automatically cluster speakers when anchor ranges are not available
- smooth short speaker-label islands
- write raw JSON plus final continuous Markdown

Typical use:

```bash
python \
  scripts/podcast_transcribe.py \
  --audio transcripts/xiaoyuzhou_episode.m4a \
  --slug example_episode \
  --title "Episode title" \
  --podcast "Podcast name" \
  --source "https://www.xiaoyuzhoufm.com/episode/..." \
  --speaker-ranges "主持人=0:45,60:90" \
  --speaker-ranges "嘉宾=180:240,600:660"
```

YouTube example:

```bash
python \
  scripts/podcast_transcribe.py \
  --youtube-url "https://www.youtube.com/watch?v=..." \
  --slug example_youtube_episode \
  --title "Video title" \
  --podcast "YouTube" \
  --auto-speakers 2
```

Use explicit `--speaker-ranges` for the best voiceprint-level result, `--auto-speakers 2` when names are unknown, and `--single-speaker 名字` for solo episodes.

## Validation

Before saying a batch is done:

- Check that every requested episode has one final `*连续段落*.md`.
- Check JSON files parse if they were generated.
- Check final Markdown has no adjacent same-speaker headings.
- Check the zip contains exactly the intended final Markdown files.
- If cleanup is requested, keep only final continuous Markdown and final continuous zip in `transcripts/`; move audio, raw JSON, raw Markdown, and temporary package folders to an archive folder outside `transcripts/`.

## Cleanup Rule

For this user, `transcripts/` is the readable final-output folder. Unless they explicitly ask to keep working files there, archive or remove:

- audio downloads (`.m4a`, `.mp3`, `.wav`)
- raw ASR JSON
- diarization JSON
- raw/uncorrected Markdown
- temporary package folders
- duplicate older zips

Keep:

- `*连续段落*.md`
- the latest final zip whose name indicates `连续段落版`
