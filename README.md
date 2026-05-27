# Podcast Transcription Skill

Codex skill for turning podcasts, YouTube videos, and local audio files into continuous paragraph transcripts with speaker separation.

The opinionated output format is designed for reading, not subtitle editing: adjacent speech by the same person is merged into one paragraph block, and timestamps appear only when the speaker changes.

## What It Does

- Transcribes audio locally with `faster-whisper`.
- Separates speakers with SpeechBrain ECAPA voice embeddings.
- Supports voiceprint anchors for stronger speaker identity assignment.
- Can automatically cluster speakers when names are unknown.
- Produces final Markdown transcripts in a continuous paragraph format.
- Keeps raw working files separate from final readable transcripts.
- Supports YouTube by downloading audio with `yt-dlp` first, then running the same transcription and diarization flow.

## Install As A Codex Skill

```bash
mkdir -p ~/.agents/skills
git clone https://github.com/fleurytian/podcast-transcription-skill.git \
  ~/.agents/skills/podcast-transcription
```

Then invoke it naturally in Codex, for example:

```text
Use $podcast-transcription to turn this Xiaoyuzhou episode into a continuous speaker-separated transcript.
```

## Python Dependencies

Create a Python environment with the local speech stack:

```bash
pip install faster-whisper speechbrain torch torchaudio numpy scikit-learn
```

For YouTube links:

```bash
pip install yt-dlp
```

`yt-dlp` downloads the audio source. The transcript and speaker separation path is still the same: local audio -> faster-whisper -> SpeechBrain speaker embeddings -> continuous Markdown.

## Local Audio Example

```bash
python scripts/podcast_transcribe.py \
  --audio transcripts/example.m4a \
  --slug example_episode \
  --title "Episode title" \
  --podcast "Podcast name" \
  --source "https://example.com/episode" \
  --speaker-ranges "Host=00:30-01:20,05:10-06:00" \
  --speaker-ranges "Guest=08:00-09:30,18:20-19:10"
```

Voiceprint anchors are short time ranges where you are confident who is speaking. They are the best way to get stable named speaker labels.

## YouTube Example

```bash
python scripts/podcast_transcribe.py \
  --youtube-url "https://www.youtube.com/watch?v=..." \
  --slug youtube_interview \
  --title "Interview title" \
  --podcast "YouTube" \
  --auto-speakers 2
```

If you know who speaks where, use `--speaker-ranges` with the YouTube command too. If you do not know the names, `--auto-speakers 2` will cluster the voices and label them as `Speaker 1`, `Speaker 2`, etc.

For solo audio:

```bash
python scripts/podcast_transcribe.py \
  --audio solo_episode.m4a \
  --slug solo_episode \
  --title "Solo episode" \
  --single-speaker "Host"
```

## Output

The helper writes:

- `{slug}_raw_segments.json`
- `{slug}_raw_逐字稿.md`
- `{slug}_diarized_segments.json`
- `{slug}_声纹区分_连续段落版.md`

The final Markdown follows this shape:

```markdown
# Episode title

来源：https://example.com/episode
播客：Podcast name
转写：faster-whisper mobiuslabsgmbh/faster-whisper-large-v3-turbo
说话人区分：SpeechBrain ECAPA voice embeddings...
规则：同一说话人连续发言合并为一个段落组，只在说话人变化处显示时间。

## 全文逐字稿

### [00:00:12] Host

...

### [00:03:41] Guest

...
```

## Speaker Separation Notes

Speaker diarization is probabilistic. For best results:

- Use explicit voiceprint anchors when real speaker names matter.
- Choose clean anchor ranges with one speaker, minimal music, and little overlap.
- Use several anchors per speaker if the recording is long.
- Use `--auto-speakers` for quick unnamed separation.
- Review the final transcript when the audio has cross-talk, music beds, phone audio, or many speakers.

The formatter validates that the same speaker does not appear in adjacent transcript sections. If the same person keeps talking, the text stays in one continuous block instead of being split by timeline labels.
