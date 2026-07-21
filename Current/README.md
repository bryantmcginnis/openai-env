# D&D Table Assistant

A Discord bot for game night: a Claude-powered DM assistant for rules/lore
questions and improvised NPCs, plus a voice-channel transcriber that posts
to a text channel using a local Whisper model.

## What changed from the old version

- **Chat backend**: OpenAI (`gpt-3.5-turbo`) -> Claude (`claude-haiku-4-5`),
  in `claude_caller/claude_caller.py`. Now keeps a short per-channel
  conversation history instead of treating every message as a fresh
  one-shot call.
- **Voice transcription**: new. Joins a voice channel, buffers each
  speaker's audio until a natural pause (not a fixed-size chunk -- that's
  what usually causes words to get cut in half and mistranscribed), and
  runs it through a local `faster-whisper` `large-v3` model with a D&D
  vocabulary hint so it stops mangling spell/monster/class names.
- **Discord library**: swapped `discord.py` for `py-cord`. Vanilla
  `discord.py` cannot receive voice audio at all -- only send it -- so
  there was no way to build the transcription feature on it.

## Why the old transcription bot probably mangled words

The two most common causes for this exact symptom (D&D voice transcription
"mishearing everything") are:

1. **Fixed-size audio chunks instead of pause-based segmentation.** If
   audio gets sliced every N seconds regardless of what's being said, words
   land split across two chunks and Whisper has no way to reconstruct them.
   This build buffers per-speaker audio and only transcribes once it
   detects a real pause in speech, so each chunk is a whole utterance.
2. **No D&D vocabulary hint.** Whisper has never heard "Eldritch Blast" or
   your NPC's name in training data, so it guesses the nearest English
   phrase. `transcription/dnd_vocabulary.py` feeds it a glossary of 5e
   terms as an `initial_prompt`, plus your own character/NPC names via
   `CAMPAIGN_VOCAB` in `.env` -- fill that in with your table's names, it
   matters more than anything else here.

A distant third possibility is a too-small local model (tiny/base) --
this build pins `large-v3`, the most accurate size that runs locally.

## Setup

Requires **Python 3.12** (the version this was built and tested against).

### 1. Install dependencies

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`faster-whisper` downloads the `large-v3` model (~3 GB) the first time the
bot records something -- that first `!record` will be slow. After that it's
cached under `~/.cache/huggingface` and reused.

> **Note on py-cord:** `requirements.txt` pins `py-cord` to a specific
> unmerged upstream commit rather than a PyPI release. This is deliberate --
> see [Discord's DAVE encryption](#known-limitation-discords-dave-end-to-end-encryption)
> below. When py-cord ships an official release with the DAVE voice-receive
> fix, swap that pin back to a normal version.

### 1b. GPU acceleration (optional, recommended if you have an NVIDIA card)

The bot auto-detects a GPU and falls back to CPU on its own -- you don't
have to configure anything for CPU-only use, and CPU is genuinely fine for
`large-v3` on a decent machine, just slower than GPU.

On Windows, `faster-whisper`'s CUDA backend needs cuBLAS, cuDNN, and their
dependencies as DLLs. `requirements.txt` pulls these in as pip wheels on
Windows automatically -- `nvidia-cublas-cu12`, `nvidia-cudnn-cu12`,
`nvidia-cuda-nvrtc-cu12`, and `nvidia-nvjitlink-cu12` -- and
`transcription/transcriber.py`'s `_register_cuda_dll_dirs()` points Windows
at their bundled DLLs (via both `os.add_dll_directory()` and prepending to
`PATH`) before `faster-whisper` loads. So `pip install -r requirements.txt`
is all you need; no system CUDA Toolkit install or manual `PATH` editing.

> **Why nvjitlink is in there:** cuBLAS 12.x depends on it at runtime, but
> pip does *not* install it automatically. Without it you get a misleading
> `Library cublas64_12.dll is not found or cannot be loaded` -- Windows
> reports an unresolved *dependency* of a DLL the same way it reports the
> DLL itself missing. That error almost always means a missing sibling
> wheel, not a missing cuBLAS.

If GPU init fails for any reason, the bot logs a clear warning and **keeps
running on CPU** instead of crashing. The device it actually used is echoed
back in the "Recording on ..." confirmation message in Discord, and logged
on model load. You can force a device with `WHISPER_DEVICE` in `.env`
(`cuda` / `cpu` / `auto`).

### 2. Discord Developer Portal

Create an application at <https://discord.com/developers/applications>,
add a Bot user, and enable:

- **Privileged Gateway Intents**: `Server Members Intent`,
  `Message Content Intent`
- **Bot permissions** when generating the invite link: `Send Messages`,
  `Read Message History`, `Connect`, `Speak`, `Use Voice Activity`

Copy the bot token into `.env`.

To add the bot to a server you don't own, you (or the owner) need the
**Manage Server** permission there -- generate an OAuth2 URL (Scopes: `bot`,
plus the permissions above) and open it, or send it to the owner to
authorize.

### 3. Configure `.env`

```powershell
cp .env.example .env
```

Fill in:

- `DISCORD_BOT_TOKEN` -- from the Developer Portal.
- `ANTHROPIC_API_KEY` -- from <https://console.anthropic.com> -> API Keys.
  Chat only; transcription runs locally and doesn't need it.
- `CAMPAIGN_VOCAB` -- your PC/NPC/place names (single biggest accuracy lever).
- `TRANSCRIPTION_CHANNEL_NAME` -- defaults to `transcription`. The channel
  must already exist in the server; the bot won't create it. Each server
  needs its own channel by this name.
- `WHISPER_DEVICE` / `WHISPER_COMPUTE_TYPE` -- optional, leave blank to
  auto-detect. See [tuning](#tuning-transcription).

## Running the bot

Activate the venv and run `main.py`:

```powershell
.venv\Scripts\Activate.ps1
python main.py
```

Or without activating, point straight at the venv's Python:

```powershell
.venv\Scripts\python.exe main.py
```

Logs (with timestamps) print to the console. `Ctrl+C` stops the bot.

To keep a persistent log file you can tail while it runs, redirect output:

```powershell
.venv\Scripts\python.exe -u main.py > bot.log 2>&1
```

...and in another terminal:

```powershell
Get-Content bot.log -Wait -Tail 20
```

(`bot.log` is git-ignored.) The bot logs each utterance it processes, which
device Whisper is on, and any errors -- so if something misbehaves, the log
is the first place to look.

## Using it

- **Chat**: `@BotName <question>` in any channel, or DM the bot directly.
  `!forget` clears that channel's conversation history.
- **Dice**: `!roll 2d6`
- **Transcription**: join a voice channel, then `!record`. The bot joins
  whichever voice channel *you* are in and posts transcripts to your
  transcription channel as `**DisplayName:** text`. `!stoprecording` to stop
  and leave. There's no "drag the bot into voice" step -- it joins itself in
  response to `!record`.

## Tuning transcription

All of these take effect on the next bot restart.

### Latency vs. accuracy (`transcription/audio_sink.py`)

- **`SILENCE_HANG_MS`** (default `400`) -- how long a pause must be before an
  utterance is considered finished and sent to Whisper. **Lower = snappier**
  (transcripts appear sooner) but chops mid-sentence more readily when
  someone pauses to think, which costs Whisper sentence context and can hurt
  accuracy. Try 300 for a livelier feel, 500-600 if you're seeing sentences
  split across two messages.
- **`MAX_UTTERANCE_MS`** (default `12000`) -- worst-case ceiling: force-flush
  a buffer even without a pause, so a continuous talker still gets chunked
  and transcribed instead of nothing until they stop. Lower for a monologue-
  heavy table, higher if you'd rather have fewer, longer, more-context
  transcripts.
- **`MIN_UTTERANCE_MS`** (default `300`) -- drop anything shorter than this
  (coughs, mic bumps). Raise it if you're getting spammy one-word noise.
- **`SILENCE_RMS_THRESHOLD`** (default `300.0`) -- volume below which audio
  counts as "silence". Raise it if a noisy mic / background hum is stopping
  pauses from ever registering; lower it if quiet talkers get clipped.

### Model quality / speed (`.env`)

- **`WHISPER_COMPUTE_TYPE`** -- precision the model runs at. Blank uses the
  ctranslate2 default (`float16` on GPU). `int8_float16` is noticeably faster
  and uses less VRAM for a small accuracy cost; `int8` is the CPU-friendly
  option. Good first knob if GPU transcription is keeping up but you want
  more headroom.
- **`WHISPER_DEVICE`** -- `auto` (default), `cuda`, or `cpu`. Force `cpu` to
  sidestep GPU entirely.
- **Model size** -- `MODEL_SIZE` in `transcription/transcriber.py` is
  `large-v3`. `distil-large-v3` or `medium` are faster with some accuracy
  loss if `large-v3` is too heavy for your hardware.

### Accuracy, above all else

Keep `CAMPAIGN_VOCAB` current. Adding your table's proper nouns (PCs, NPCs,
locations, homebrew spells) does more for perceived accuracy than any of the
knobs above -- it's the difference between "Eldritch Blast" and "elder itch
blast."

## Ideas for where to take it next

Now that transcription works, some fun directions:

- **Session recap.** Have Claude read back the day's `#transcription`
  channel and post a "Previously on..." summary at the top of next session.
  You already have both halves (a transcript feed and a Claude caller).
- **Auto-summarize on `!stoprecording`.** Same idea, triggered automatically
  when recording ends -- a tidy end-of-session log.
- **Searchable session archive.** Write transcripts to dated files so you can
  grep "what did the innkeeper say his name was" three sessions later.
- **NPC voice generator.** A `!npc <trait>` command that has Claude improvise
  a name, voice, and one-liner on the spot.
- **Initiative / combat tracker.** `!init` commands, or even parse initiative
  order out of the live transcript.
- **Slash commands.** Migrate `!`-prefix commands to Discord slash commands
  for autocomplete and nicer UX.
- **Speaker-attributed summaries.** The transcript already tags who said
  what -- feed that to Claude for "each player's key moments" recaps.

## Known limitation: Discord's DAVE end-to-end encryption

Discord has rolled out E2EE (the DAVE protocol) for voice. This broke voice
*receiving* in stock py-cord releases:

- py-cord 2.7.x has no DAVE support and can't even connect to voice on a
  DAVE-enforced server (close code `4017`).
- py-cord 2.8.0 connects but its voice-receive rewrite mishandles decryption
  and crashes recording (`OpusError: corrupted stream`).

This build works around it by pinning py-cord to the unmerged upstream PR
that fixes the decrypt path (tracked at
<https://github.com/Pycord-Development/pycord/issues/3139>), adding a handful
of `Sink` compatibility shims in `transcription/audio_sink.py`, and forcing
DAVE "passthrough" mode on once the session is ready in `dnd_bot.py` so
unencrypted packets aren't rejected. It's working, but it's leaning on an
in-flight fix -- **when py-cord ships an official release with DAVE
receive support, move the `requirements.txt` pin back to a normal version**
and the shims can likely be retired.

## Cost note

Chat runs on Claude Haiku 4.5 (~$1/$5 per million tokens) -- cheap for
hobby use. Transcription runs entirely locally via `faster-whisper`, so it
costs nothing per minute of audio but does use your CPU/GPU while
recording.

---

## Development notes (build history)

Short changelog of what it took to get voice transcription working
end-to-end on Windows with a GPU, in case any of it regresses:

- **py-cord version dance.** Landed on an unmerged PR commit for DAVE
  voice-receive (see the limitation section above). Along the way the
  `Sink` API had changed: the sink needs `__sink_listeners__`,
  `walk_children`, and `is_opus` shims, `write()` now receives a `VoiceData`
  wrapper + `Member` (not raw PCM + user id), and `sink.init(voice_client)`
  has to be called manually.
- **DAVE passthrough.** Even on the fixed py-cord, packets were rejected as
  `UnencryptedWhenPassthroughDisabled` until we waited for the DAVE session
  to go `ready` and called `set_passthrough_mode(True, ...)` ourselves.
- **Windows GPU / cuBLAS.** The pip CUDA wheels don't put their DLLs on the
  search path, and cuBLAS's `nvjitlink` dependency isn't installed
  automatically -- both surfaced as the same misleading "cublas64_12.dll not
  found." Fixed by shipping all four nvidia wheels and registering their
  dirs via `add_dll_directory` **and** `PATH` (the `add_dll_directory`
  handles must also be kept alive, or the registration self-undoes on GC).
- **Concurrency.** Transcription is serialized behind an `asyncio.Lock` so
  overlapping utterances can't call the GPU model from two threads at once.
- **Latency tuning.** Utterance flush thresholds were lowered
  (`SILENCE_HANG_MS` 700 -> 400, `MAX_UTTERANCE_MS` 30s -> 12s) so
  transcripts land promptly instead of after long pauses.
- **Logging.** `main.py` configures timestamped logging; voice/opus debug
  logging can be re-enabled there if the receive path needs diagnosing
  again.