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

### The easy way

Double-click **`start_bot.bat`**, or the "D&D Bot" shortcut on your Desktop.
The console window shows the live log; close it to stop the bot.

**It restarts itself after a crash** (`run_bot.py`), waiting 10 s, then 30 s,
then 60 s. Five crashes inside 15 minutes is a crash loop rather than bad
luck, so it stops there and leaves the error on screen. `!shutdown` exits
with code 0 and is never restarted, and closing the window stops it all.

**If it was recording when it died, it carries on.** `!record` leaves a
note in `campaign_memory/active_recording.json`, and `!stoprecording` /
`!shutdown` remove it. On startup the bot rejoins any recording from the
last 6 hours, provided someone is still in the voice channel, and says once:

> ♻️ I crashed and restarted, so I've rejoined voice and I'm recording
> again. The transcript has a gap from about 21:14 until now.

That covers an accidentally closed window too: relaunch within 6 hours and
the recording resumes.

To create the Desktop and Start Menu shortcuts (once):

```powershell
powershell -ExecutionPolicy Bypass -File install_shortcut.ps1
```

Then right-click the Desktop shortcut → *Show more options* → *Pin to
taskbar*, or find "D&D Bot" in the Start app list → *Pin to Start*.

> The shortcut targets `cmd.exe` with the batch file as an argument rather
> than pointing at the `.bat` directly. Windows refuses to pin a shortcut to
> the **taskbar** unless its target is an executable -- aim it at the `.bat`
> and you get something that pins to Start but not the taskbar.

The bot writes its own logs, so they exist however you started it -- no
shell redirection needed:

- **`bot.log`** -- everything, rotating at 5 MB × 3.
- **`bot_events.log`** -- only warnings, errors and session milestones
  (recording started/stopped, recap posted, audio stalled/recovered,
  threads and quotes updated, a startup check of this PC's clock against
  Discord's -- it warns if they're more than 10 s apart),
  rotating at 1 MB × 5. This is the one to read after a bad night: it holds
  weeks of timeline, and a flood in `bot.log` can't push it out.

A repeated warning or error is logged in full once, then counted, with one
summary line every 30 s (`suppressed 1874 more of [ERROR] ...`). That's
because on 2026-10-14 the voice receiver logged a traceback for every
undecryptable packet -- about 60 a second -- and its 20 MB of rotation
filled in six minutes, overwriting the hour that would have shown what
broke. INFO lines, including every transcribed utterance, are never
throttled. See `log_setup.py`.

### From a terminal

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
- **Dice**: `!roll 2d6`, `!roll 1d20+5`, `!roll 2d6+1d4+3`, `!roll 4d6kh3`
  (keep the highest 3 -- stat rolling), `!roll 1d20 adv` / `dis` for
  advantage and disadvantage. `!roll adv` on its own is a d20 with
  advantage. Dropped dice show ~~struck through~~ in the breakdown. See
  [dice notation](#dice-notation).
- **Recap**: `!recap` summarizes everything since the last recap (or since
  the session started, if it's the first). `!recap 60` overrides that with a
  literal last-60-minutes window. One is posted automatically when you
  `!stoprecording`. See [session recaps](#session-recaps).
- **Transcription**: join a voice channel, then `!record`. The bot joins
  whichever voice channel *you* are in and posts transcripts to your
  transcription channel as `**DisplayName:** text`. `!stoprecording` to stop
  and leave. There's no "drag the bot into voice" step -- it joins itself in
  response to `!record`.

## Dice notation

`!roll` (`discord_bot/dice.py`) takes the notation people actually type:

| You type | You get |
|---|---|
| `!roll 2d6` | both dice and the total |
| `!roll 1d20+5` | modifier applied, shown in the breakdown |
| `!roll 2d6+1d4+3` | several dice terms and modifiers in one roll |
| `!roll 4d6kh3` | roll 4, **k**eep the **h**ighest 3 -- the stat-rolling idiom |
| `!roll 2d20kl1` | keep the lowest |
| `!roll 4d6dl1` | **d**rop the **l**owest (same result, other way round) |
| `!roll 1d20 adv` | roll twice, take the higher |
| `!roll 1d20+7 dis` | disadvantage, modifier still applied |
| `!roll adv` | bare d20 with advantage |

Advantage doubles the first dice term and keeps the better half, so it's
exactly 5e's rule for the usual `1d20+5` case. Spaces around `+`/`-` are
fine (`1d20 + 5`). Limits: 100 dice per term, 200 per roll, up to d1000,
10 terms. Anything it can't parse comes back as a short message saying so
rather than a stack trace.

## Session recaps

`!stoprecording` posts a recap automatically, and `!recap` generates one on
demand.

### Where a recap starts, and why it isn't a timer

Nothing about the boundary lives in the bot's memory -- it's re-derived from
the transcription channel every time, so a mid-session restart doesn't cost
you the earlier half of the night. `_resolve_recap_start()` walks the
channel backwards from now and stops at whichever of these it reaches first:

- **the last recap it posted** -- so a second `!recap` covers only what has
  happened since the first, instead of retelling the whole evening;
- **a gap longer than `SESSION_GAP_MINUTES`** (90) -- the start of tonight's
  session. This is what makes restarts survivable: the bot's own start time
  is gone, but the transcript still shows where the session began.

Because the scan stops at the *first* boundary it meets, the more recent one
always wins -- last week's recap can't beat this week's session start.

Recaps are posted as **embeds** titled "Session recap", which does double
duty: an embed carries no message text, so a recap can never be scooped back
up as transcript and fed into the next recap, and the title makes it a
self-identifying marker for the scan above. The footer records the window it
covered.

`!recap 60` takes an explicit window **literally** -- if you ask for 60
minutes you get 60 minutes, even where that re-covers ground an earlier
recap already handled. Use it to redo a recap you didn't like, or to catch
up on just the last fight. Bare `!recap` is the one that tracks boundaries.

Both forms are capped at `MAX_LOOKBACK_HOURS` (24) of scanning.

### Styles

`!recap <style>` changes the register. Combine with anything else, in any
order -- `!recap sarcastic 90` and `!recap 90 sarcastic` are the same.

| Style | Reads like |
|---|---|
| `funny` | comedy pushed harder, still only jokes that happened |
| `sarcastic` | dry, deadpan, the friend who warned them (aliases: `sarcasm`, `dry`) |
| `short` | 150 words, spine only (aliases: `brief`, `tldr`) |
| `long` | up to 1200 words, scene detail and dialogue (aliases: `detailed`, `full`) |
| `factual` | no jokes, chronological, plain (aliases: `facts`, `plain`) |
| `epic` | chronicle/saga register (aliases: `dramatic`, `saga`) |
| `noir` | hard-boiled, short sentences, world-weary |

`!recap styles` lists them in Discord. Styles are appended to the **system**
prompt rather than the first pass, because a style set only on the draft
gets quietly edited back out by the polish pass.

### Personal recaps

`!recap <character>` writes a recap of that character's night -- second
person, their decisions and spotlight moments, what they specifically
learned, what's unresolved for them -- and **sends it as a DM**.

```
!recap Sykth              -> Sykth's night, DM'd to you
!recap @Justin Llyr       -> Llyr's night, DM'd to Justin
!recap Thia long          -> the long version
```

A bare character name DMs *you*; the bot only DMs someone else when you
mention them, so nobody gets an unexpected DM. If their DMs are closed to
server members you get told, rather than it failing silently.

Two deliberate differences from a party recap: a personal recap is titled
differently so it **never becomes the boundary marker** for the next party
recap, and it resolves its window from the *session start* rather than the
last recap -- otherwise asking for one after the party recap would hand you
a recap of the four minutes since.

**"Not enough transcript" usually means "already recapped".** Running
`!recap` twice in a row hits the boundary from the first one and finds
almost nothing new -- which is the intended behaviour, not a failure. The
message says which boundary it used and suggests the `!recap <minutes>`
override, rather than implying the channel is empty.

### Where the recap gets posted

The recap always goes to the **transcription channel**, wherever you ran the
command from. If you ran `!recap` somewhere else, the status message in that
channel ends with "Recap posted in #transcription" so you know where to
look -- without that, invoking it from another channel looks exactly like
the bot silently giving up.

The bot reads back its own messages in the transcription channel, strips
them to plain `Name: line` form, and sends them to **Claude Opus 5.5** at
`effort: "high"`. (Chat stays on Haiku -- at the table you want a rules
answer immediately; the recap is written once and read many times, so it's
worth the better model and the wait.)

The prompts (`claude_caller/claude_caller.py`) tell it the transcript is
machine-generated -- misheard proper nouns, cut-off sentences, table talk
mixed in with play -- so it infers names from context and skips the parts
about pizza. They also ask for **wit**: dry and affectionate, with the
comedy coming from what actually happened rather than jokes bolted on, and
accuracy outranking the laugh every time. Output is a "Previously on..."
narrative, a **Loose threads** bullet list, and **Notable** moments.

### How a recap is built

1. **Map** -- the transcript is split into `SUMMARY_CHUNK_CHARS` (25k
   characters, ~6k tokens) slices, each summarized in detail on its own.
   Deliberately small: more, finer passes keep the names and numbers that
   one broad pass flattens away.
2. **Reduce** -- the slice summaries are stitched into a single narrative.
3. **Polish** -- a second pass edits the finished draft for timing and
   flatness, with the source material alongside it so it can't drift from
   what happened. Editing sharpens voice far more reliably than asking for
   accuracy and comedy in one shot.

A single-chunk session skips step 2.

**Measured cost on Opus 5.5 ($4 / $20 per million tokens):**

| Load | Passes | Tokens (in + out) | Cost | Time |
|---|---|---|---|---|
| A real captured session, 20k characters | 2 | 18k + 6k | **$0.19** | 54 s |
| A 13k-word, ~4-hour session | 5 | 41k + 15k | **$0.47** | 145 s |

Most of that is reasoning, not text. Cost scales with the number of
*passes*, not session length -- the recap is ~550 words whether the session
ran two hours or five -- so a longer night costs only slightly more.

Two dials if that's the wrong number for you: `RECAP_EFFORT` (`"medium"` is
about 40% cheaper and a little flatter -- measured $0.07 vs $0.11 on a small
sample) and `RECAP_MODEL` in `.env` (`latest-haiku` gives a recap in about 25
seconds for around a cent, but it reads noticeably flatter). See
[choosing models](#choosing-models).

> **Don't set `RECAP_EFFORT = "max"` on Opus 5.5.** It exhausted the entire
> 32,000-token budget on reasoning and wrote nothing, 3 runs out of 3, even
> on a 326-word transcript (Opus 5 used ~1,600 tokens for the same pass).
> Each failed pass costs ~$0.64. See below.

### It takes minutes, and that's expected

A long session is up to 5-8 Opus passes at high effort -- around **2-3
minutes** end to end. The bot posts a status message and edits it after each
pass (`Recap 3/5: reading part 3 of 3...`) so you can tell it's working
rather than hung.

### `SUMMARY_MAX_TOKENS` is mostly headroom for thinking

`max_tokens` caps thinking **plus** response text in one budget. When
reasoning exhausts it, the API returns HTTP 200 with a lone thinking block,
**no text at all**, and `stop_reason: "max_tokens"`. That is not an error you
can catch by wrapping the call; it looks like a successful response that
happens to be empty.

`_attempt()` checks for it and raises, so an empty result becomes a readable
failure instead of a recap that silently never appears. The ceiling is 32k
(the recap itself is only ~700 tokens, so nearly all of that is thinking
headroom, and unused budget costs nothing). It is a safety net, not a
target: on Opus 5.5 at `effort: "max"` it is exactly what gets exhausted,
which is why the default is `"high"`.

## Choosing models

Set in `.env`:

```
CHAT_MODEL=latest-haiku
RECAP_MODEL=claude-opus-5-5
```

Each takes either a **pinned model ID** (`claude-opus-5-5`), which never
changes underneath you, or **`latest-<family>`** (`latest-haiku`,
`latest-sonnet`, `latest-opus`), which uses the newest model of that family
the API offers. The API has no floating "latest" alias of its own -- a
dateless ID like `claude-haiku-5-5` is a fixed snapshot -- so the bot looks
it up once at startup and logs what it picked:

```
chat model: claude-haiku-5-5 (from 'latest-haiku') | recap model: claude-opus-5-5 (from 'claude-opus-5-5')
```

If the API can't be reached at startup, `latest-` settings fall back to a
known model rather than failing. Request settings like effort and adaptive
thinking are only sent to models that support them, so pinning an older
model (e.g. `claude-haiku-4-5`) still works.

**Why the recap is pinned and chat isn't.** A new model version can behave
differently under the same settings. Moving the recap from Opus 5 to Opus 5.5
at the same `effort: "max"` made every recap fail: the model spent its
whole 32k-token budget thinking and wrote nothing (3 runs out of 3, ~$0.64
each). Chat has almost nothing to tune, so following the newest Haiku is
safe; for the recap, change `RECAP_MODEL` deliberately and run a test recap.
Changes take effect on the next bot start.

## When the transcript stalls

On 2026-10-14 the bot stopped transcribing 32 minutes into a session and
nobody knew until the recap came out short 77 minutes later. It hadn't
crashed: it was still in voice, receiving everyone's audio, and failing to
decrypt every packet. py-cord's receiver drops packets it can't decrypt
without counting them or telling anyone, and it builds its decryptor once,
from the voice key at the start of recording -- its `update_secret_key()`
exists but nothing calls it -- so if the voice session is re-keyed
mid-recording, every packet after that fails. That's the leading theory;
the logs that would prove it were lost to the flood described above.

`discord_bot/voice_health.py` now watches every recording. If packets keep
arriving for 15 s and none of them decrypt, it escalates, and never repeats
a visible step within one recording:

1. **Re-sync the decryption key** from the live voice connection. Silent and
   instant. If a stale key was the cause, this is the whole fix, and nobody
   is bothered -- it's noted in `bot_events.log`.
2. **One full voice rejoin**, announced in the channel where `!record` was
   typed *and* in `#transcripts` (the table mutes transcripts, so an alert
   only there would go unseen):
   > ⚠️ **Transcription problem:** I've been hearing the voice channel since
   > 21:03 but can't decode the audio... Rejoining voice to fix it.

   The same transcript continues afterwards. If audio comes back, that
   message is edited in place to ✅ with the gap's start and end times.
3. **If it doesn't come back, or it stalls again later that recording,** one
   ❌ message says what to run, and the bot stops trying, so the table gets
   one clear message rather than a stream of them.

A quiet table never trips it -- silence sends no packets, so there's
nothing failing. Two stalls in one recording cost at most two posts and one
edit.

## Character names, and fixing a recap

**Who's who.** `campaign_context.md` has a "speaker labels" block:

```
- wildgoose = Thia (Goose), she/her
```

Before any recap, question or scan reads the transcript, the bot relabels
each line, so `wildgoose: ...` becomes `Thia (Goose): ...`. The recap is
told to use character names and pronouns in the story, and to use players'
real names only in the **IRL:** bullets under Notable, which hold the best
out-of-character moments. Add a line whenever someone new joins.

**`!correct` (or `!fix`)** edits the latest party recap in place. It works
best in a DM to the bot:

> !correct The "terrible at practicing law" line was the DM, not Sykth.
> The five-headed screamer IS the teen.

The bot re-reads that session's transcript, rewrites only what the
corrections touch, edits the original message (adding a "corrected by"
footer), and replies with a list of the changes and a link. If the
corrections also make an open thread or a quote's speaker wrong, it fixes
those on `#potjf-to-do` and `#quotes` too. Corrections are kept in
`campaign_memory/corrections.json` and added to that session's archived
recap. A re-run recap, `!threads scan`, or a question about that night all
treat them as fact, so the mistake doesn't come back. A correction costs
about $0.10–0.30 on Opus.

Some mistakes start in the transcript, not the recap. If someone's mic picks
up the DM, the DM's line gets transcribed under that player's name, and the
recap faithfully credits the wrong person. `!correct` is how you fix that.

## Asking about the session

`@bot` a question while a session is on, or for 12 hours after it ends, and
it answers from what was actually said: tonight's transcript with
timestamps, the last five recaps, the open threads, and the campaign notes.

> **@dndbot** what did Sykth roll the last time he attacked?
> At [22:37] Sykth said "there's a 22", which looks like his attack roll...

Rules questions still get rules answers -- there's no separate router
deciding which kind of question it is. The model gets both and is told
which to use when; a router would cost a second call and an extra way to be
wrong, to save a fraction of a cent. If something isn't in the records it
says "I didn't catch that" instead of guessing, and it flags when an answer
comes from an older session's recap rather than a transcript.

A question costs about $0.003 on Haiku 5.5. The records are prompt-cached,
so follow-ups a few minutes later are cheaper and faster (measured: 7.8 s
for the first, 3.8 s after).

## Open threads (`#potjf-to-do`)

After each party recap, one extra call reads the night and updates the
campaign's open threads: new hooks, promises and mysteries get added, and
threads the transcript clearly settles are ticked off, each with how it was
resolved and a quip. The list lives in **one pinned bot message in
`#potjf-to-do`**, edited in place -- in the channel's own ⬜/✅ style -- and
the bot never touches anyone else's messages there.

| Command | Does |
|---|---|
| `!threads` | list the open threads |
| `!threads add <text>` | add one by hand |
| `!threads resolve <n> <how>` | tick one off |
| `!threads reopen <n>` | undo that |
| `!threads scan` | build threads and quotes from the last session |

Personal recaps and literal `!recap 60` windows don't touch the list, so
nothing is counted twice. Threads that differ by a word ("the silver key" /
"the golden key") are kept separate; only near-identical restatements are
merged. Everything is stored in `campaign_memory/` (threads, quotes, and an
archive of every party recap), which is committed to git.

## Quote board (`#quotes`)

Three ways in:

- **React ⭐ to a line in `#transcripts`** -- every line in that message
  goes on the board, and the bot ticks it ✅.
- **`!quote`** -- the last thing anyone said. `!quote Thia` for one
  person's last line (Discord name or character name, read from
  `campaign_context.md`), `!quote Thia 2` for the one before that.
- **Automatically after each recap** -- the night's most memorable lines,
  posted together in one message.

Each quote gets **an italic setup line** above it, so it still makes sense
weeks later:

> *Joe mentions he leaves for work at 6:45.*
> > Everyone likes a morning quickie.
> — **Thia (Goose)**, 24 Oct 2026

Recap quotes get theirs from the same call that picks them. ⭐ and `!quote`
quotes post straight away, then a few seconds later the bot reads the
lines around the quote (marked, so it knows which moment), asks Haiku for
one sentence, and edits it in (about $0.005 each). If that fails, the quote
just stays as it was.

The recap quotes are **checked word for word against the transcript**
before posting. The model is told to copy them exactly, and anything it
tidied up or invented is dropped rather than posted as something a friend
supposedly said. Nothing is ever posted twice.

## Release notes (`#dndbot-release-notes`)

The bot announces its own updates. At startup it compares the current git
commit with the last one it announced (`campaign_memory/release_notes.json`).
If there are new commits, it gives their messages and code diff to the
recap model, which writes a post for the table: what's new and how to use
it, what improved, and which bugs were fixed. Technical details are
explained in plain words. The post goes to `#dndbot-release-notes` and is
archived in `release_notes/`.

- **Only committed code is announced.** Edits you haven't committed yet
  are never described.
- **To ship an update:** commit, then restart the bot (or send
  `!releasenotes` if you're the owner and don't want to restart).
- **The first run** posts `release_notes/history.md`, the hand-written
  story of the bot so far, before any automatic notes.
- **No channel, no post.** The bot never creates the channel, and nothing
  is marked as announced, so the notes wait until the channel exists. If
  writing the notes fails, the next start tries again.
- **Cost:** one Opus call per release, typically $0.10–0.50.
- **Better commit messages make better notes.** The writer only uses the
  commit messages and the code, so "fix" tells it nothing, while "the bot
  stopped hearing Goose after a reconnect" gives it something to work with.

## Tuning transcription

All of these take effect on the next bot restart.

### Latency vs. accuracy (`transcription/audio_sink.py`)

- **`SILENCE_HANG_MS`** (default `900`) -- how long a pause must be before an
  utterance is considered finished and sent to Whisper. **Lower = snappier**
  (transcripts appear sooner) but chops mid-sentence more readily when
  someone pauses to think, which costs Whisper sentence context and can hurt
  accuracy. Try 500-600 for a livelier feel, 1200+ for a DM who narrates in
  long paragraphs with dramatic pauses.
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

### How transcripts get posted (`discord_bot/transcript_writer.py`)

Whisper wants whole sentences, but one Discord message per sentence turns a
minute of DM narration into a dozen entries. So consecutive utterances from
the same speaker are coalesced: the first posts immediately, and the ones
after it *edit that message*, growing it in place. A different speaker, a
long gap, or Discord's length limit rolls over to a new message.

- **`COALESCE_WINDOW_S`** (default `90.0`) -- how long a speaker's message
  stays open for more text. Raise it for longer unbroken blocks; lower it if
  messages feel like they're absorbing things said much later.
- **`EDIT_DEBOUNCE_S`** (default `2.0`) -- appends are batched into one edit
  per this many seconds, to stay clear of Discord's edit rate limit. Lower
  makes text appear more incrementally, at the cost of more API calls.
- **`MAX_MESSAGE_CHARS`** (default `1900`) -- roll over to a new message
  before Discord's hard 2000-character limit.

### "Thanks for watching!" and other things nobody said

If a transcript line appears that no one at the table spoke -- classically
**"Thanks for watching!"**, but also "Thank you.", "Subscribe!", "Bye" --
that's a *Whisper hallucination*, not a stray listener. Whisper was trained
on a huge amount of YouTube audio paired with its subtitles, so a segment
containing no real speech (a breath, a chair creak, hiss that slipped past
VAD) gets captioned with the most likely text for that kind of audio: a
video outro. The model generated it from noise; nothing was overheard.

Three defenses are in place, all in `transcription/transcriber.py`:

- `vad_filter=True` drops non-speech before it ever reaches the decoder.
- `no_speech_threshold` / `log_prob_threshold` / `compression_ratio_threshold`
  discard segments the model is unsure of, and ones that look like a
  repetition loop.
- `_HALLUCINATION_PHRASES` drops an utterance whose *entire* text is a known
  phantom phrase. Only exact whole-utterance matches are removed, so "thanks
  for watching the door" survives. Add to this set if your table sees other
  phantom lines; remove the short entries (`"okay"`, `"so"`, `"you"`) if you
  find genuine one-word replies going missing.

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

Keep **`campaign_vocab.txt`** current. Adding your table's proper nouns (PCs,
NPCs, locations, homebrew spells) does more for perceived accuracy than any
of the knobs above -- it's the difference between "Vesper Kane" and "Whisper
Cain."

One term per line, `#` for comments, blank lines ignored; commas on a line
work too, so an old `CAMPAIGN_VOCAB` value pastes straight in. The file lives
next to `main.py`; point `CAMPAIGN_VOCAB_FILE` elsewhere if you'd rather.
`CAMPAIGN_VOCAB` in `.env` still works and is merged in.

**This is a hard budget, and it is smaller than it looks.** Whisper's
`initial_prompt` is capped at **223 tokens** -- half its 448-token decoder
context. Go over and faster-whisper doesn't error: it silently keeps the
*last* 223 tokens and drops the rest off the front. The built-in 5e glossary
alone measured 316 tokens, so before this was handled roughly 30 core terms
were being discarded every session without a word about it.

`build_prompt()` now budgets deliberately:

- **Campaign terms get first claim.** A general speech model already knows
  what a Paladin is; it has no idea about "the Sunken Archive."
- **Core 5e terms fill what's left**, trimmed from the end, logged at INFO.
- **Campaign terms are placed last** in the prompt, so if anything still
  overflows, truncation-from-the-front eats the generic vocabulary first.
- **If the campaign glossary alone overflows**, the leading entries are kept
  and a WARNING names every dropped term. Entries near the top of the file
  win, so put your most-said, most-mangled names first.

In practice that leaves room for roughly **30-40 short campaign terms**.
Prune names that stop coming up rather than letting the list grow.

> Whisper is **fully local** -- `faster-whisper` loads `large-v3` into VRAM
> once and every utterance after that is GPU inference on your machine. The
> only network traffic is a one-time model cache check with HuggingFace when
> the model loads. "Prompt" and "tokens" here refer to the local decoder's
> conditioning string, not an API call: transcription costs nothing and
> sends nothing off the box. The only paid API calls are chat and recaps.

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

Chat runs on the newest Claude Haiku (Haiku 5.5: from $0.10/$0.50 per
million tokens) -- effectively free for hobby use. Recaps cost cents per
session; see [session recaps](#session-recaps). Transcription runs entirely locally via `faster-whisper`, so it
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