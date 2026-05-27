# Podcast Transcription Skill / 播客转写 Skill

Codex skill for turning podcasts, YouTube videos, and local audio files into continuous paragraph transcripts with speaker separation.

一个用于小宇宙播客（Xiaoyuzhou / 小宇宙）、YouTube 视频和本地音频转写的 Codex skill：先提取音源，再用本地 `faster-whisper` 转写，并通过 SpeechBrain ECAPA 声纹嵌入区分说话人，最后输出适合阅读的连续段落版 Markdown。

The output format is designed for reading, not subtitle editing: adjacent speech by the same person is merged into one paragraph block, and timestamps appear only when the speaker changes.

输出格式面向阅读，而不是字幕编辑：同一个人连续说话会合并成一个段落组，只有说话人变化时才显示新的时间标签。

## What It Does / 功能

- Transcribes audio locally with `faster-whisper`.
- 使用本地 `faster-whisper` 转写音频。
- Separates speakers with SpeechBrain ECAPA voice embeddings.
- 使用 SpeechBrain ECAPA 声纹嵌入区分说话人。
- Supports voiceprint anchors for stronger speaker identity assignment.
- 支持手动提供声纹锚点，让真实说话人姓名匹配更稳定。
- Can automatically cluster speakers when names are unknown.
- 不知道说话人姓名时，也可以自动聚类成 `Speaker 1`, `Speaker 2` 等。
- Produces final Markdown transcripts in a continuous paragraph format.
- 输出连续段落版 Markdown，避免同一说话人被时间线标签反复切断。
- Keeps raw working files separate from final readable transcripts.
- 将 raw JSON、音频等工作文件与最终可读稿分开管理。
- Supports YouTube by downloading audio with `yt-dlp` first, then running the same transcription and diarization flow.
- 支持 YouTube 链接：先用 `yt-dlp` 下载音轨，再走同一套转写与声纹区分流程。
- Supports Xiaoyuzhou podcast transcription through RSS/enclosure audio extraction when available.
- 支持小宇宙播客转写：可通过 RSS/enclosure 音频地址提取音源，再生成带说话人区分的全文逐字稿。

## Supported Sources / 支持来源

- Xiaoyuzhou / 小宇宙 podcasts
- YouTube videos
- Local audio files such as `.m4a`, `.mp3`, `.wav`, `.webm`, and `.opus`
- Podcast RSS enclosure audio

- 小宇宙播客链接
- YouTube 视频链接
- 本地音频文件，如 `.m4a`, `.mp3`, `.wav`, `.webm`, `.opus`
- 播客 RSS 中的 enclosure 音频地址

## Install As A Codex Skill / 安装为 Codex Skill

```bash
mkdir -p ~/.agents/skills
git clone https://github.com/fleurytian/podcast-transcription-skill.git \
  ~/.agents/skills/podcast-transcription
```

Then invoke it naturally in Codex, for example:

安装后可以在 Codex 里自然调用，例如：

```text
Use $podcast-transcription to turn this Xiaoyuzhou episode into a continuous speaker-separated transcript.
```

## Python Dependencies / Python 依赖

Create a Python environment with the local speech stack:

创建一个带本地语音处理依赖的 Python 环境：

```bash
pip install faster-whisper speechbrain torch torchaudio numpy scikit-learn
```

For YouTube links:

如果要处理 YouTube 链接：

```bash
pip install yt-dlp
```

`yt-dlp` downloads the audio source. The transcript and speaker separation path is still the same: local audio -> faster-whisper -> SpeechBrain speaker embeddings -> continuous Markdown.

`yt-dlp` 只负责下载音源；后续流程仍然是：本地音频 -> faster-whisper 转写 -> SpeechBrain 声纹嵌入 -> 连续段落版 Markdown。

## Local Audio Example / 本地音频示例

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

声纹锚点是你确信某个人正在说话的短时间段。想要稳定地标出真实姓名时，锚点是最好的方式。

## YouTube Example / YouTube 示例

```bash
python scripts/podcast_transcribe.py \
  --youtube-url "https://www.youtube.com/watch?v=..." \
  --slug youtube_interview \
  --title "Interview title" \
  --podcast "YouTube" \
  --auto-speakers 2
```

If you know who speaks where, use `--speaker-ranges` with the YouTube command too. If you do not know the names, `--auto-speakers 2` will cluster the voices and label them as `Speaker 1`, `Speaker 2`, etc.

如果你知道某个时间段是谁在说话，也可以在 YouTube 命令里使用 `--speaker-ranges`。如果不知道姓名，`--auto-speakers 2` 会自动按声音聚类，并标成 `Speaker 1`, `Speaker 2` 等。

For solo audio:

单人音频可以直接指定一个说话人：

```bash
python scripts/podcast_transcribe.py \
  --audio solo_episode.m4a \
  --slug solo_episode \
  --title "Solo episode" \
  --single-speaker "Host"
```

## Output / 输出

The helper writes:

脚本会写出：

- `{slug}_raw_segments.json`
- `{slug}_raw_逐字稿.md`
- `{slug}_diarized_segments.json`
- `{slug}_声纹区分_连续段落版.md`

The final Markdown follows this shape:

最终 Markdown 大致如下：

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

## Speaker Separation Notes / 说话人区分说明

Speaker diarization is probabilistic. For best results:

说话人区分是概率判断。为了得到更好的结果：

- Use explicit voiceprint anchors when real speaker names matter.
- 如果真实姓名很重要，尽量提供明确的声纹锚点。
- Choose clean anchor ranges with one speaker, minimal music, and little overlap.
- 选择干净的锚点：单人说话、少背景音乐、少重叠说话。
- Use several anchors per speaker if the recording is long.
- 长音频建议每个说话人提供多个锚点。
- Use `--auto-speakers` for quick unnamed separation.
- 只需要快速区分不同声音时，可以用 `--auto-speakers`。
- Review the final transcript when the audio has cross-talk, music beds, phone audio, or many speakers.
- 如果音频里有抢话、垫乐、电话音质或多人讨论，建议人工复核最终稿。

The formatter validates that the same speaker does not appear in adjacent transcript sections. If the same person keeps talking, the text stays in one continuous block instead of being split by timeline labels.

格式化器会检查同一说话人不会出现在相邻标题里。如果同一个人持续说话，文本会保持在一个连续段落组里，而不会被时间标签切碎。

## Keywords / 关键词

Xiaoyuzhou transcript, 小宇宙转写, 小宇宙逐字稿, podcast transcription, Chinese podcast transcript, YouTube transcription, speaker diarization, voiceprint speaker separation, faster-whisper, SpeechBrain, 连续段落版逐字稿, 声纹区分说话人。
