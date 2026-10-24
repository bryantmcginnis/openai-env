# Requires the 'Server Members' and 'Message Content' privileged intents to
# be enabled for this bot in the Discord Developer Portal.

import asyncio
import datetime
import logging
import os
import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
# import time  # only used by the disabled _dump_debug_wav helper
# import wave  # only used by the disabled _dump_debug_wav helper

import discord
import httpx
from discord.ext import commands
from dotenv import load_dotenv

from claude_caller.claude_caller import RECAP_STYLES, ClaudeChatCaller, resolve_style
from discord_bot import campaign_memory as memory
from discord_bot import dice, voice_health
from discord_bot.transcript_writer import TranscriptWriter
from transcription.audio_sink import UtteranceSink
from transcription.dnd_vocabulary import build_recap_briefing, load_speakers, relabel
from transcription.transcriber import Transcriber

load_dotenv()

_log = logging.getLogger(__name__)
# Session milestones. Goes to bot_events.log as well as bot.log, so the
# timeline of a night survives even if bot.log gets flooded (log_setup.py).
_events = logging.getLogger("dndbot.events")

# Debugging aid (disabled): directory for per-utterance WAV dumps. See the
# commented _dump_debug_wav helper and its call in _handle_utterance.
# _DEBUG_AUDIO_DIR = "debug_audio"

# os.getenv's default only applies when the key is absent -- .env sets this
# blank rather than unset when left at its default, which loads as "" (see
# the same issue fixed in transcription/transcriber.py).
TRANSCRIPTION_CHANNEL_NAME = os.getenv("TRANSCRIPTION_CHANNEL_NAME") or "transcription"

# Recaps are posted as embeds titled this. Two reasons, both load-bearing:
# an embed carries no message content, so a recap can never be scooped back
# up as transcript and fed into the next recap; and the title makes a recap
# self-identifying, so the channel itself records where the last one ended
# and no in-memory state has to survive a restart.
RECAP_TITLE = "Session recap"
RECAP_COLOUR = 0x9B59B6

# A pause longer than this in the transcript means a different session
# rather than a lull. Used to find where the current session started when
# nothing else tells us -- notably after the bot restarts mid-session and
# loses the in-memory start time.
SESSION_GAP_MINUTES = 90
# Ceiling on how far back we will ever scan or recap.
MAX_LOOKBACK_HOURS = 24

_MENTION_RE = re.compile(r"<@!?\d+>")

# Where the quote board and the threads board live. Both channels must exist;
# the bot doesn't create channels.
QUOTES_CHANNEL_NAME = os.getenv("QUOTES_CHANNEL_NAME") or "quotes"
TODO_CHANNEL_NAME = os.getenv("TODO_CHANNEL_NAME") or "potjf-to-do"
# React with one of these on a transcript line to put it on the quote board.
QUOTE_EMOJIS = {"⭐", "🌟"}

# Chat answers questions about the game from tonight's transcript while a
# session is on, and for this long after it ends.
SESSION_CONTEXT_HOURS = 12
# The transcript is re-read from Discord at most this often for chat questions.
CONTEXT_CACHE_S = 20
# Warn in bot_events.log when this PC's clock is further off Discord's.
CLOCK_TOLERANCE_S = 10

# Set by !shutdown. main.py turns it into exit code 0, which run_bot.py reads
# as "stopped on purpose, don't restart".
shutdown_requested = threading.Event()

description = (
    "A D&D 5e table assistant. Mention the bot (or DM it) to ask rules "
    "questions or have it improvise NPCs; use !record to transcribe your "
    "voice channel to a text channel."
)

intents = discord.Intents.default()
intents.members = True
intents.message_content = True
intents.voice_states = True

bot = commands.Bot(command_prefix="!", description=description, intents=intents)

claude = ClaudeChatCaller()

# The Whisper model is a few GB -- load it lazily on first use instead of at
# import time, so `!roll`-only sessions never pay for it.
_transcriber: Transcriber | None = None
_active_recordings: set[int] = set()  # guild IDs currently recording
# Per-guild transcript writers, so !stoprecording can flush a message that
# still has an edit pending before we walk away from the session.
_transcript_writers: dict[int, TranscriptWriter] = {}
# When each guild's current recording started, so the recap knows how far
# back to read the transcription channel.
_recording_started_at: dict[int, datetime.datetime] = {}
# The last few hundred lines said, per guild, word for word, for !quote.
_recent_lines: dict[int, deque] = defaultdict(lambda: deque(maxlen=300))
_context_cache: dict[int, tuple[float, tuple[str, str] | None]] = {}
_startup_done = False

# Each utterance flush spawns its own independent asyncio.to_thread call with
# no coordination, so back-to-back utterances (continuous speech, overlapping
# speakers) could call self.model.transcribe() on the same ctranslate2 model
# instance from multiple threads at once. That looks like the real cause of
# the intermittent "cublas64_12.dll not found" failures and hangs -- GPU
# inference engines are frequently not safe to call concurrently on one
# model instance. Serialize all transcription through this lock.
_transcribe_lock = asyncio.Lock()


def get_transcriber() -> Transcriber:
    global _transcriber
    if _transcriber is None:
        _transcriber = Transcriber()
    return _transcriber


async def _clear_stale_voice_states():
    """Leave any voice channel Discord still thinks we're sitting in.

    Discord keeps a bot's voice state server-side until it's told otherwise.
    If the process dies without disconnecting cleanly -- a crash, a kill, or
    a restart mid-session -- the bot keeps *appearing* in the voice channel
    while the new process holds no connection to it. It looks present and
    hears nothing, `!stoprecording` reports it isn't recording, and the next
    `!record` may fail because Discord thinks it's already there.

    guild.me.voice set while guild.voice_client is None is exactly that
    mismatch: a voice state this process doesn't own. Clear it.
    """
    for guild in bot.guilds:
        voice_state = guild.me.voice if guild.me else None
        if voice_state is None or guild.voice_client is not None:
            continue
        _log.warning(
            "clearing stale voice state in %r (channel %r) left by a previous run",
            guild.name, getattr(voice_state.channel, "name", "?"),
        )
        await guild.change_voice_state(channel=None)


@bot.event
async def on_ready():
    global _startup_done
    _log.info("Logged in as %s (ID: %s)", bot.user, bot.user.id)
    # on_ready can fire again on a gateway resume; this is idempotent and
    # only acts when a voice state has no connection behind it.
    await _clear_stale_voice_states()
    if _startup_done:
        return
    _startup_done = True
    await asyncio.to_thread(_check_clock)
    for guild_id, info in memory.pending_resumes().items():
        asyncio.create_task(_resume_recording(guild_id, info))


def _check_clock() -> None:
    """Compare this PC's clock with Discord's, once at startup.

    A PC clock drifted two minutes fast once, which made every log time
    disagree with every message time while diagnosing a lost recording. This
    makes drift show up in bot_events.log instead of as confusion.
    """
    try:
        before = datetime.datetime.now(datetime.timezone.utc)
        response = httpx.get("https://discord.com/api/v10/gateway", timeout=10)
        after = datetime.datetime.now(datetime.timezone.utc)
        server = parsedate_to_datetime(response.headers["date"])
    except Exception as exc:
        _log.info("couldn't check the clock against Discord: %s", exc)
        return
    skew = (before + (after - before) / 2 - server).total_seconds()
    if abs(skew) > CLOCK_TOLERANCE_S:
        _events.warning(
            "this PC's clock is %+.0f s off Discord's, so log and message times won't line up "
            "and `!recap <minutes>` windows are off by that much. Windows: Settings > Time & "
            "language > Date & time > Sync now.", skew,
        )
    else:
        _events.info("clock check: within %.0f s of Discord", abs(skew))


@bot.event
async def on_command_error(ctx: commands.Context, error: commands.CommandError):
    if isinstance(error, commands.NotOwner):
        # Deliberately no detail beyond this -- don't confirm or deny who
        # the owner is to someone who isn't them.
        await ctx.send("That command is restricted to the bot's owner.")
        return

    # py-cord's default handler for this prints straight to stderr with no
    # timestamp (discord/ext/commands/bot.py) -- route it through logging
    # instead so failures like a broken !record are dated in bot.log.
    _log.error("Unhandled exception in command %s", ctx.command, exc_info=error)


# ---------------------------------------------------------------------------
# Chat: mention the bot, or DM it, to talk to the DM assistant.
# ---------------------------------------------------------------------------

@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    # Resolve the command first and stop there if it is one. The old order
    # ran process_commands() and then *also* fell through to the chat path,
    # so a "!recap" typed in a DM ran the command and separately asked
    # Claude what "!recap" meant -- two replies, one of them nonsense.
    ctx = await bot.get_context(message)
    if ctx.valid:
        await bot.invoke(ctx)
        return

    is_dm = isinstance(message.channel, discord.DMChannel)
    is_mention = bot.user in message.mentions
    if not (is_dm or is_mention):
        return

    content = message.content
    if is_mention:
        content = content.replace(f"<@{bot.user.id}>", "").replace(f"<@!{bot.user.id}>", "").strip()
    if not content:
        return

    async with message.channel.typing():
        guild = message.guild or _guild_for_dm(message.author)
        try:
            context = await _session_context(guild) if guild else None
        except discord.HTTPException as exc:
            _log.warning("couldn't build session context: %s", exc)
            context = None
        try:
            reply = await asyncio.to_thread(claude.call, content, message.channel.id, context)
        except Exception as exc:  # surface API errors at the table instead of dropping them silently
            reply = f"(Claude API error: {exc})"

    for start in range(0, len(reply), 2000):  # Discord's message length limit
        await message.channel.send(reply[start:start + 2000])


def _guild_for_dm(user: discord.abc.User) -> discord.Guild | None:
    """The table a DM is about: the one shared server with a transcripts
    channel, or the one that's recording right now. No API calls -- this runs
    on every DM."""
    shared = [g for g in bot.guilds if g.get_member(user.id) and _find_transcription_channel(g)]
    if len(shared) == 1:
        return shared[0]
    live = [g for g in shared if g.id in _active_recordings]
    return live[0] if len(live) == 1 else None


async def _session_context(guild: discord.Guild) -> tuple[str, str] | None:
    """(stable records, tonight's transcript) for answering questions about the game.

    No separate "is this a session question?" router: the model gets both
    the records and its rules knowledge and is told which to use when. A
    router would add a second call to every answer, plus a way to misroute,
    to save a fraction of a cent.
    """
    cached = _context_cache.get(guild.id)
    if cached and time.monotonic() - cached[0] < CONTEXT_CACHE_S:
        return cached[1]

    parts = []
    notes = build_recap_briefing()
    if notes:
        parts.append(f"<campaign_notes>\n{notes}\n</campaign_notes>")
    open_now = memory.open_threads(memory.load_threads())
    if open_now:
        parts.append("<open_threads>\n" + "\n".join(
            f"#{t['id']} {t['title']}" + (f" -- {t['detail']}" if t["detail"] else "") for t in open_now
        ) + "\n</open_threads>")
    recaps = memory.recent_recaps(5)
    if recaps:
        parts.append("<recent_session_recaps>\n" + "\n\n---\n\n".join(recaps) + "\n</recent_session_recaps>")

    transcript = ""
    channel = _find_transcription_channel(guild)
    start = await _resolve_session_start(channel) if channel else None
    if start is not None:
        text, last = await _collect_timed_transcript(channel, start)
        recent = last is not None and discord.utils.utcnow() - last < datetime.timedelta(hours=SESSION_CONTEXT_HOURS)
        if guild.id in _active_recordings or recent:
            transcript = text

    result = ("\n\n".join(parts), transcript) if (parts or transcript) else None
    _context_cache[guild.id] = (time.monotonic(), result)
    return result


async def _collect_timed_transcript(
    text_channel: discord.TextChannel, since: datetime.datetime
) -> tuple[str, datetime.datetime | None]:
    """Like _collect_transcript, with each line's local time so answers can
    point at when something was said. Also returns the newest line's time."""
    lines, last = [], None
    speakers = load_speakers()
    async for message in text_channel.history(limit=None, after=since, oldest_first=True):
        if not _is_transcript_line(message):
            continue
        content = _labelled_line(message.content, speakers)
        if content:
            lines.append(f"[{message.created_at.astimezone():%H:%M}] {content}")
            last = message.edited_at or message.created_at
    return "\n".join(lines), last


def _labelled_line(content: str, speakers: dict) -> str:
    """'**wildgoose:** text' -> 'Thia (Goose): text', per campaign_context.md."""
    match = re.match(r"\*\*(.+?):\*\*\s*(.*)", content, re.S)
    if not match:
        return content.replace("**", "").strip()
    return f"{relabel(match.group(1), speakers)}: {match.group(2).strip()}"


@bot.command()
async def forget(ctx: commands.Context):
    """Clears the assistant's memory of this channel's conversation."""
    claude.reset(ctx.channel.id)
    await ctx.send("Cleared my memory of this channel.")


@bot.command(name="shutdown", aliases=["sleep", "die"])
@commands.is_owner()
async def shutdown(ctx: commands.Context):
    """Stops the bot process entirely. Owner only.

    Restricted via py-cord's is_owner() check, which resolves against
    whoever owns the Discord *application* in the Developer Portal -- your
    account specifically, not a server role or permission. A server admin
    granting themselves every role still can't run this; only you can.

    Works from a DM, which is the point: you don't need to still have the
    console window open, or even be in the server, to be sure nobody else
    is running your Claude API key or Discord token after you've stepped
    away.
    """
    # If we're mid-recording anywhere, stop capturing audio *before*
    # anything else. That's the actual security-sensitive bit here -- an
    # extra few seconds spent trying to post a graceful recap first is a
    # few more seconds the bot (and your API key) stays live, which is
    # exactly what this command exists to avoid. The transcript already
    # written to the channel isn't lost -- `!recap` after the next restart
    # picks it up the same way it does after any other restart.
    for cap in list(_captures.values()):
        cap.stop_watchdog()
    stopped_in: list[str] = []
    for guild in list(bot.guilds):
        voice_client = guild.voice_client
        if voice_client is not None and voice_client.is_connected():
            if guild.id in _active_recordings:
                voice_client.stop_recording()
                _active_recordings.discard(guild.id)
            await voice_client.disconnect(force=True)
            stopped_in.append(guild.name)

    note = (
        f" Was recording in {', '.join(stopped_in)} -- stopped. "
        f"Run `!recap` after restarting to summarize it."
        if stopped_in else ""
    )
    _log.warning("shutdown requested by %s (id %s).%s", ctx.author, ctx.author.id, note)
    shutdown_requested.set()
    memory.clear_recording()  # deliberate stop: don't resume these on next start
    await ctx.send(
        f"Shutting down.{note}"
        + " Restart from the desktop shortcut when you're back."
    )
    await bot.close()


@bot.command()
async def roll(ctx: commands.Context, *, expression: str = ""):
    """Rolls dice, e.g. !roll 2d6, !roll 1d20+5, !roll 4d6kh3, !roll 1d20 adv.

    Supports modifiers (`+3`/`-1`), several terms at once (`2d6+1d4+3`),
    keep/drop (`4d6kh3` keeps the highest 3), and advantage/disadvantage.
    """
    # Takes the whole rest of the line rather than one word, so the trailing
    # `adv` in `!roll 1d20+5 adv` reaches the parser.
    try:
        result = dice.roll(expression)
    except dice.DiceError as exc:
        await ctx.send(str(exc))
        return

    await ctx.send(dice.format_result(result))


# ---------------------------------------------------------------------------
# Voice: !record joins your voice channel and transcribes it live to the
# #transcription channel; !stoprecording leaves.
# ---------------------------------------------------------------------------

async def _resolve_guild(ctx: commands.Context) -> discord.Guild | None:
    """Work out which server a DM'd command is about.

    In a DM ctx.guild is None, so anything that reads guild state has to
    find one. Preference order: servers we share with the caller, narrowed
    to those that actually have a transcription channel, and if that still
    leaves a choice, whichever has the most recent transcript -- which is
    almost always the table they just played at.
    """
    if ctx.guild is not None:
        return ctx.guild

    shared = [g for g in bot.guilds if g.get_member(ctx.author.id)]
    candidates = [g for g in shared if _find_transcription_channel(g)]
    if not candidates:
        await ctx.send(
            f"I can't find a `#{TRANSCRIPTION_CHANNEL_NAME}` channel in any "
            f"server we share, so there's nothing for me to read."
        )
        return None
    if len(candidates) == 1:
        return candidates[0]

    # Several tables -- pick the one that was used most recently.
    newest, newest_at = None, None
    for guild in candidates:
        channel = _find_transcription_channel(guild)
        try:
            async for message in channel.history(limit=20):
                if _is_transcript_line(message):
                    if newest_at is None or message.created_at > newest_at:
                        newest, newest_at = guild, message.created_at
                    break
        except discord.HTTPException:
            continue
    if newest is None:
        await ctx.send(
            "We share several servers and none of them have any transcript "
            "yet -- run the command in the server you mean."
        )
        return None
    _log.info("DM command resolved to guild %r (most recent transcript)", newest.name)
    return newest


def _find_transcription_channel(guild: discord.Guild) -> discord.TextChannel | None:
    return discord.utils.get(guild.text_channels, name=TRANSCRIPTION_CHANNEL_NAME)


# Debug helper (disabled): dumps a captured utterance to a WAV file for
# manual inspection. Re-enable this and its call in _handle_utterance
# together when diagnosing capture issues.
# def _dump_debug_wav(pcm_bytes: bytes, display_name: str) -> str:
#     os.makedirs(_DEBUG_AUDIO_DIR, exist_ok=True)
#     path = os.path.join(_DEBUG_AUDIO_DIR, f"{int(time.time() * 1000)}_{display_name}.wav")
#     with wave.open(path, "wb") as wav_file:
#         wav_file.setnchannels(2)
#         wav_file.setsampwidth(2)
#         wav_file.setframerate(48000)
#         wav_file.writeframes(pcm_bytes)
#     return path


async def _close_writer(guild_id: int) -> None:
    writer = _transcript_writers.pop(guild_id, None)
    if writer is None:
        return
    # The sink flushes each speaker's final partial utterance as recording
    # stops, and those are still queued for transcription right now. Let the
    # queue drain first, so the last sentence of the session still gets
    # appended before we close the message off.
    await asyncio.sleep(0.5)
    async with _transcribe_lock:
        pass
    await writer.close()


async def _handle_utterance(
    guild: discord.Guild, writer: TranscriptWriter, user_id: int, pcm_bytes: bytes
):
    member = guild.get_member(user_id)
    display_name = member.display_name if member else f"user {user_id}"

    # Debug audio dump -- writes every utterance to disk, which bloats the
    # drive over a session. Re-enable (with _dump_debug_wav below) only when
    # you need to inspect exactly what was captured.
    # debug_path = await asyncio.to_thread(_dump_debug_wav, pcm_bytes, display_name)
    # _log.info("dumped debug audio to %s", debug_path)

    async with _transcribe_lock:
        text = await asyncio.to_thread(get_transcriber().transcribe_pcm, pcm_bytes)
    _log.info(
        "transcribed utterance for %s: %d bytes -> %r", display_name, len(pcm_bytes), text
    )
    if not text:
        return

    await writer.add(user_id, display_name, text)
    _recent_lines[guild.id].append((display_name, text, datetime.datetime.now()))


@dataclass
class _Capture:
    """One recording session: everything a self-heal rejoin has to carry over.

    The voice connection and py-cord's reader are replaceable; the transcript
    writer, channels and watchdog are not, so a rejoin keeps writing into the
    same transcript messages instead of starting the night over.
    """

    guild: discord.Guild
    voice_channel: discord.abc.Connectable
    transcripts: discord.TextChannel
    chat: discord.abc.Messageable  # where !record was typed -- where people are looking
    writer: TranscriptWriter
    loop: asyncio.AbstractEventLoop
    health: voice_health.CaptureHealth = field(default_factory=voice_health.CaptureHealth)
    sink: UtteranceSink | None = None
    watchdog: asyncio.Task | None = None

    def stop_watchdog(self) -> None:
        # Safe from py-cord's audio thread as well as the event loop.
        if self.watchdog is not None:
            self.loop.call_soon_threadsafe(self.watchdog.cancel)


_captures: dict[int, _Capture] = {}


async def _arm_dave_passthrough(voice_client: discord.VoiceClient) -> None:
    # py-cord only calls DaveSession.set_passthrough_mode() during an
    # explicit DAVE protocol transition (discord/voice/state.py); joining a
    # channel where DAVE is already fully engaged never triggers one, so any
    # participant whose client sends unencrypted audio gets every packet
    # rejected with "DecryptionFailed(UnencryptedWhenPassthroughDisabled)"
    # for the entire session. We don't need DAVE's E2EE guarantee -- we're
    # already a channel member with full audio access -- so force it on.
    dave_session = getattr(voice_client._connection, "dave_session", None)
    _log.info("dave_session at record start: %r", dave_session)
    if dave_session is not None:
        # The DAVE session's MLS key exchange happens asynchronously over
        # the gateway *after* connect() returns, so it's still INACTIVE /
        # not ready right here -- setting passthrough mode on it now gets
        # discarded once the real handshake completes. Wait for it to
        # actually go ready before arming passthrough. py-cord's own calls
        # always pass a numeric expiry (120, 10) in discord/voice/state.py,
        # never None, so None likely doesn't mean "no expiry" -- use a
        # duration long enough to cover a whole session.
        for _ in range(50):  # up to ~5s
            if dave_session.ready:
                break
            await asyncio.sleep(0.1)
        dave_session.set_passthrough_mode(True, 3600)
        _log.info(
            "set_passthrough_mode(True, 3600) called; status=%r ready=%r",
            dave_session.status, dave_session.ready,
        )


async def _start_capture(cap: _Capture) -> None:
    """Join cap.voice_channel and start recording into cap.writer."""
    voice_client = await cap.voice_channel.connect()
    await _arm_dave_passthrough(voice_client)

    async def on_utterance(user_id: int, pcm_bytes: bytes):
        await _handle_utterance(cap.guild, cap.writer, user_id, pcm_bytes)

    sink = UtteranceSink(cap.loop, on_utterance)
    # py-cord 2.8.0 no longer calls sink.init() itself anywhere in the voice
    # module -- without this, self.sink.client stays None and PacketDecoder
    # crashes looking up the speaking member.
    sink.init(voice_client)
    cap.sink = sink
    voice_client.start_recording(sink, _make_finished_callback(cap, sink))
    voice_health.instrument(cap.health, voice_client)


def _make_finished_callback(cap: _Capture, sink: UtteranceSink):
    def on_recording_finished(exception: Exception | None):
        # Runs on py-cord's audio thread; hop back to the event loop for cleanup.
        if cap.sink is not sink:
            # A self-heal rejoin replaced this recording -- the session goes on.
            return
        _active_recordings.discard(cap.guild.id)
        _captures.pop(cap.guild.id, None)
        memory.clear_recording(cap.guild.id)
        cap.stop_watchdog()
        # The sink flushes any half-finished utterance during cleanup, so
        # close the writer behind it rather than here, once those have been
        # transcribed and appended.
        asyncio.run_coroutine_threadsafe(_close_writer(cap.guild.id), cap.loop)
        if exception:
            asyncio.run_coroutine_threadsafe(
                cap.chat.send(f"Recording stopped with an error: {exception}"), cap.loop
            )

    return on_recording_finished


async def _rejoin(cap: _Capture) -> bool:
    """Leave and rejoin voice, carrying the session over. True if recording again."""
    # Detach first, so the old recording's finished callback leaves the
    # session alone instead of closing the transcript.
    cap.sink = None
    old = cap.guild.voice_client
    try:
        if old is not None:
            if old.is_recording():
                old.stop_recording()  # flushes buffered speech into the same transcript
            await old.disconnect(force=True)
        await asyncio.sleep(2)
        await _start_capture(cap)
        _events.warning("rejoined voice in %s / %s", cap.guild.name, cap.voice_channel.name)
        return True
    except Exception as exc:
        _events.error("voice rejoin failed in %s", cap.guild.name, exc_info=exc)
        # Nothing is recording now. Clear the session so `!record` works again
        # rather than answering "Already recording".
        _active_recordings.discard(cap.guild.id)
        _captures.pop(cap.guild.id, None)
        memory.clear_recording(cap.guild.id)
        asyncio.create_task(_close_writer(cap.guild.id))
        return False


def _alert_channels(cap: _Capture) -> list:
    # Chat first: the table mutes the transcripts channel because of how much
    # the bot posts there, so an alert only there would go unseen. Never let
    # an alert start with "**" -- that's how transcript lines are recognised.
    channels = [cap.chat]
    if getattr(cap.chat, "id", None) != cap.transcripts.id:
        channels.append(cap.transcripts)
    return channels


async def _post_alert(cap: _Capture, text: str) -> list:
    sent = []
    for channel in _alert_channels(cap):
        try:
            sent.append(await channel.send(text))
        except discord.HTTPException as exc:
            _events.warning("could not post alert in %s: %s", getattr(channel, "name", channel), exc)
    return sent


async def _edit_alert(sent: list, text: str) -> None:
    for message in sent or []:
        try:
            await message.edit(content=text)
        except discord.HTTPException as exc:
            _events.warning("could not update alert: %s", exc)


def _start_watchdog(cap: _Capture) -> None:
    watchdog = voice_health.Watchdog(
        cap.health,
        get_voice_client=lambda: cap.guild.voice_client,
        rejoin=lambda: _rejoin(cap),
        notify=lambda text: _post_alert(cap, text),
        update=_edit_alert,
        guild_name=cap.guild.name,
    )
    cap.watchdog = asyncio.create_task(watchdog.run())


async def _resume_recording(guild_id: int, info: dict) -> None:
    """Rejoin voice and carry on recording after a crash or a closed window.

    A restart that comes back without recording is just another silent gap,
    which is what this exists to prevent. Says so once in the chat channel.
    """
    await asyncio.sleep(3)  # let _clear_stale_voice_states finish first
    guild = bot.get_guild(guild_id)
    voice = guild.get_channel(info["voice_channel_id"]) if guild else None
    chat = guild.get_channel(info["chat_channel_id"]) if guild else None
    transcripts = _find_transcription_channel(guild) if guild else None
    if not (voice and chat and transcripts) or guild_id in _active_recordings:
        memory.clear_recording(guild_id)
        return
    if not [m for m in voice.members if not m.bot]:
        _events.info("not resuming recording in %s: nobody is in %s", guild.name, voice.name)
        memory.clear_recording(guild_id)
        return

    gap_from = None
    async for message in transcripts.history(limit=30):
        if _is_transcript_line(message):
            gap_from = message.edited_at or message.created_at
            break

    loop = asyncio.get_running_loop()
    try:
        await asyncio.to_thread(get_transcriber)
        writer = TranscriptWriter(transcripts)
        _transcript_writers[guild_id] = writer
        cap = _Capture(guild=guild, voice_channel=voice, transcripts=transcripts,
                       chat=chat, writer=writer, loop=loop)
        _active_recordings.add(guild_id)
        await _start_capture(cap)
        _captures[guild_id] = cap
        _start_watchdog(cap)
        memory.mark_recording(guild_id, voice.id, chat.id)
    except Exception as exc:
        _active_recordings.discard(guild_id)
        _transcript_writers.pop(guild_id, None)
        memory.clear_recording(guild_id)
        _events.error("could not resume recording in %s after a restart", guild.name, exc_info=exc)
        try:
            await chat.send("❌ I restarted after a crash but couldn't rejoin voice. Run `!record` to start again.")
        except discord.HTTPException:
            pass
        return

    gap = f" The transcript has a gap from about {gap_from.astimezone():%H:%M} until now." if gap_from else ""
    _events.warning("resumed recording in %s / %s after a restart", guild.name, voice.name)
    try:
        await chat.send(f"♻️ I crashed and restarted, so I've rejoined voice and I'm recording again.{gap}")
    except discord.HTTPException as exc:
        _events.warning("resumed, but couldn't say so in chat: %s", exc)


@bot.command()
async def record(ctx: commands.Context):
    """Joins your voice channel and transcribes speech to the transcription channel."""
    if ctx.guild is None:
        await ctx.send(
            "Run this one in the server -- I need to see which voice channel "
            "you're in, and a DM doesn't tell me that."
        )
        return
    if ctx.author.voice is None or ctx.author.voice.channel is None:
        await ctx.send("Join a voice channel first.")
        return

    text_channel = _find_transcription_channel(ctx.guild)
    if text_channel is None:
        await ctx.send(
            f"No `#{TRANSCRIPTION_CHANNEL_NAME}` text channel found -- create one, "
            f"or set TRANSCRIPTION_CHANNEL_NAME in .env to an existing channel's name."
        )
        return

    if ctx.guild.id in _active_recordings:
        await ctx.send("Already recording in this server.")
        return

    first_load = _transcriber is None
    if first_load:
        await ctx.send("Loading the transcription model for the first time, hang on...")
    transcriber = await asyncio.to_thread(get_transcriber)

    loop = asyncio.get_running_loop()
    writer = TranscriptWriter(text_channel)
    _transcript_writers[ctx.guild.id] = writer
    cap = _Capture(
        guild=ctx.guild,
        voice_channel=ctx.author.voice.channel,
        transcripts=text_channel,
        chat=ctx.channel,
        writer=writer,
        loop=loop,
    )

    # Recorded just before the first utterance can land, so the recap picks
    # up everything from this session and nothing from the last one.
    _recording_started_at[ctx.guild.id] = discord.utils.utcnow() - datetime.timedelta(seconds=5)

    _active_recordings.add(ctx.guild.id)
    try:
        await _start_capture(cap)
    except Exception:
        _active_recordings.discard(ctx.guild.id)
        _transcript_writers.pop(ctx.guild.id, None)
        raise
    _captures[ctx.guild.id] = cap
    _start_watchdog(cap)
    memory.mark_recording(ctx.guild.id, cap.voice_channel.id, ctx.channel.id)
    _events.info(
        "recording started in %s / %s on %s", ctx.guild.name, cap.voice_channel.name, transcriber.device
    )

    await ctx.send(
        f"Recording on {transcriber.device.upper()} -- transcripts go to "
        f"#{text_channel.name}. Say `!stoprecording` when done and I'll post "
        f"a recap."
    )


@bot.command()
async def stoprecording(ctx: commands.Context):
    """Stops transcribing, leaves the voice channel, and posts a session recap."""
    if ctx.guild is None:
        await ctx.send(
            "Run this one in the server -- I need to see which voice channel "
            "you're in, and a DM doesn't tell me that."
        )
        return
    voice_client = ctx.guild.voice_client
    if voice_client is None or not voice_client.is_connected():
        await ctx.send("Not currently recording.")
        return

    cap = _captures.get(ctx.guild.id)
    if cap is not None:
        cap.stop_watchdog()
    if ctx.guild.id in _active_recordings:
        voice_client.stop_recording()
    await voice_client.disconnect()
    _events.info("recording stopped in %s by %s", ctx.guild.name, ctx.author)
    memory.clear_recording(ctx.guild.id)
    await ctx.send("Stopped recording.")

    _recording_started_at.pop(ctx.guild.id, None)
    text_channel = _find_transcription_channel(ctx.guild)
    if text_channel is None:
        return

    # stop_recording() flushes the last utterances through transcription on
    # its way out; wait for those to land so the recap covers the final
    # moments of the session rather than stopping a sentence short.
    await asyncio.sleep(2.0)
    async with _transcribe_lock:
        pass

    # Deliberately re-derived from the channel rather than using the start
    # time we recorded in memory: that value is lost if the bot restarts
    # mid-session, and the channel still knows where the session began.
    resolved = await _resolve_recap_start(text_channel)
    if resolved is None:
        await ctx.send("Nothing transcribed to recap.")
        return
    await _post_recap(ctx, text_channel, *resolved)


# ---------------------------------------------------------------------------
# Campaign memory: the threads board in the to-do channel, and the quote board.
# ---------------------------------------------------------------------------

def _speaker_label(speaker: str) -> str:
    """Discord name -> 'Thia (Goose)'; already-labelled names pass through."""
    return relabel(speaker)


def _quote_block(speaker: str, text: str, date_iso: str) -> str:
    try:
        stamp = datetime.date.fromisoformat(date_iso).strftime("%d %b %Y").lstrip("0")
    except ValueError:
        stamp = date_iso
    return f"> {text}\n— **{_speaker_label(speaker)}**, {stamp}"


async def _post_quotes(guild: discord.Guild, quotes: list[tuple[str, str]], source: str,
                       message_id: int | None = None) -> int:
    """Post quotes to the quote board, skipping any already on it. Returns how many went up."""
    channel = discord.utils.get(guild.text_channels, name=QUOTES_CHANNEL_NAME)
    if channel is None:
        _events.warning("no #%s channel in %s; quote not posted", QUOTES_CHANNEL_NAME, guild.name)
        return 0
    fresh = [(sp, tx.strip()) for sp, tx in quotes
             if tx.strip() and memory.remember_quote(sp, tx.strip(), source, message_id)]
    if not fresh:
        return 0
    today = datetime.date.today().isoformat()
    chunk, texts = "", []
    for speaker, text in fresh:  # several quotes share a message; Discord caps one at 2000 chars
        block = _quote_block(speaker, text, today)
        if chunk and len(chunk) + len(block) + 2 > 1900:
            posted = await channel.send(chunk)
            memory.set_quote_post(texts, posted.id)
            chunk, texts = "", []
        chunk = f"{chunk}\n\n{block}" if chunk else block
        texts.append(text)
    posted = await channel.send(chunk)
    # Remembered so !correct can re-credit a quote by editing this message.
    memory.set_quote_post(texts, posted.id)
    return len(fresh)


async def _refresh_board(guild: discord.Guild, data: dict) -> discord.Message | None:
    """Edit the one bot-owned threads message in the to-do channel (posting and
    pinning it the first time). Never touches anyone else's messages there."""
    channel = discord.utils.get(guild.text_channels, name=TODO_CHANNEL_NAME)
    if channel is None:
        return None
    text = memory.render_board(data)
    if data.get("board_message_id"):
        try:
            message = await channel.fetch_message(data["board_message_id"])
            await message.edit(content=text)
            return message
        except discord.NotFound:
            pass  # someone deleted it; post a fresh one
    message = await channel.send(text)
    data["board_message_id"] = message.id
    memory.save_threads(data)
    try:
        await message.pin()
    except discord.HTTPException as exc:
        _events.warning("posted the threads board in #%s but couldn't pin it: %s", channel.name, exc)
    return message


async def _update_campaign_memory(chat, guild: discord.Guild, since: datetime.datetime,
                                  transcript: str, recap_text: str) -> None:
    """After a party recap: archive it, then one model call updates threads and quotes."""
    try:
        if recap_text.strip():
            memory.save_recap(recap_text, since)
        data = memory.load_threads()
        result = await asyncio.to_thread(
            claude.update_memory, transcript, recap_text, memory.open_threads(data), _briefing_for(since)
        )
    except Exception as exc:
        _events.error("campaign memory update failed", exc_info=exc)
        return

    added = [t for t in (memory.add_thread(data, x["title"], x["detail"]) for x in result["new_threads"]) if t]
    resolved = [t for t in (memory.resolve_thread(data, x["id"], x["how"], x["quip"]) for x in result["resolved"]) if t]
    memory.save_threads(data)
    try:
        board = await _refresh_board(guild, data)
    except discord.HTTPException as exc:
        _events.warning("couldn't update the threads board: %s", exc)
        board = None

    # Only lines that were really said. The model is told to copy word for
    # word; anything it tidied up or invented is dropped here.
    real = [(q["speaker"], q["quote"]) for q in result["quotes"] if memory.appears_in(q["quote"], transcript)]
    try:
        posted = await _post_quotes(guild, real, source="recap")
    except discord.HTTPException as exc:
        _events.warning("couldn't post recap quotes: %s", exc)
        posted = 0
    _events.info(
        "campaign memory: %d new threads, %d resolved, %d quotes posted (%d not word-for-word, dropped)",
        len(added), len(resolved), posted, len(result["quotes"]) - len(real),
    )

    parts = []
    if added or resolved:
        where = f" in {board.channel.mention}" if board else ""
        parts.append(f"📜 Threads{where}: {len(added)} new, {len(resolved)} resolved")
    if posted:
        parts.append(f"💬 {posted} quote{'s' if posted != 1 else ''} added to #{QUOTES_CHANNEL_NAME}")
    if parts:
        try:
            await chat.send(" · ".join(parts))
        except discord.HTTPException:
            pass


@bot.event
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent):
    """⭐ on a transcript line puts every line in that message on the quote board."""
    if str(payload.emoji) not in QUOTE_EMOJIS or payload.guild_id is None or payload.user_id == bot.user.id:
        return
    guild = bot.get_guild(payload.guild_id)
    channel = guild.get_channel(payload.channel_id) if guild else None
    if channel is None or channel.name != TRANSCRIPTION_CHANNEL_NAME:
        return
    try:
        message = await channel.fetch_message(payload.message_id)
    except discord.HTTPException:
        return
    if not _is_transcript_line(message):
        return
    match = re.match(r"\*\*(.+?):\*\*\s*(.*)", message.content, re.S)
    if not match:
        return
    if await _post_quotes(guild, [(match.group(1), match.group(2))], source="star", message_id=message.id):
        try:
            await message.add_reaction("✅")
        except discord.HTTPException:
            pass


async def _lines_from_channel(guild: discord.Guild) -> list[tuple[str, str, datetime.datetime]]:
    """Recent lines from the transcript channel, for !quote after a restart
    has emptied the in-memory buffer. Coarser: one entry per message."""
    channel = _find_transcription_channel(guild)
    out = []
    if channel is None:
        return out
    async for message in channel.history(limit=60):
        match = re.match(r"\*\*(.+?):\*\*\s*(.*)", message.content, re.S) if _is_transcript_line(message) else None
        if match:
            out.append((match.group(1), match.group(2)[-400:], message.created_at))
    return list(reversed(out))


@bot.command()
async def quote(ctx: commands.Context, *, args: str = ""):
    """Puts a line on the quote board. !quote, !quote Sykth, !quote Thia 2

    No name: the last thing anyone said. A name (Discord name or character):
    that person's last line. A number: go further back (2 = the one before).
    Or react ⭐ to a line in the transcript channel.
    """
    guild = await _resolve_guild(ctx)
    if guild is None:
        return
    tokens = args.split()
    back = int(tokens.pop()) if tokens and tokens[-1].isdigit() else 1
    who = " ".join(tokens).strip().lower()

    lines = list(_recent_lines.get(guild.id, ())) or await _lines_from_channel(guild)
    if who:
        speakers = load_speakers()

        def is_them(name: str) -> bool:
            entry = speakers.get(name.lower(), {})
            return who in name.lower() or who in (entry.get("character", "").lower(), entry.get("player", "").lower())

        lines = [l for l in lines if is_them(l[0])]
    if len(lines) < max(back, 1):
        await ctx.send(f"I haven't heard {'anything from ' + who if who else 'anything'} to quote yet.")
        return
    speaker, text, _ = lines[-max(back, 1)]
    if await _post_quotes(guild, [(speaker, text)], source="!quote"):
        await ctx.message.add_reaction("✅")
    else:
        await ctx.send("That one's already on the quote board.")


@bot.group(invoke_without_command=True)
async def threads(ctx: commands.Context):
    """The campaign's open plot threads. Also: add, resolve, reopen, scan."""
    data = memory.load_threads()
    open_now = memory.open_threads(data)
    if not open_now:
        await ctx.send("No open threads. `!threads scan` builds them from the last session.")
        return
    lines = [f"⬜ **#{t['id']}** {t['title']}" + (f" -- {t['detail']}" if t["detail"] else "") for t in open_now]
    text = "**Open threads**\n" + "\n".join(lines)
    await ctx.send(text[:1990])


@threads.command(name="add")
async def threads_add(ctx: commands.Context, *, text: str):
    """Add a thread by hand: !threads add Find out who hired the scarred man"""
    guild = await _resolve_guild(ctx)
    data = memory.load_threads()
    thread = memory.add_thread(data, text)
    if thread is None:
        await ctx.send("That's already on the list.")
        return
    memory.save_threads(data)
    if guild:
        await _refresh_board(guild, data)
    await ctx.send(f"Added ⬜ **#{thread['id']}** {thread['title']}")


@threads.command(name="resolve")
async def threads_resolve(ctx: commands.Context, thread_id: int, *, how: str = "settled at the table"):
    """Close a thread: !threads resolve 3 We paid the debt in full"""
    guild = await _resolve_guild(ctx)
    data = memory.load_threads()
    thread = memory.resolve_thread(data, thread_id, how)
    if thread is None:
        await ctx.send(f"There's no open thread #{thread_id}.")
        return
    memory.save_threads(data)
    if guild:
        await _refresh_board(guild, data)
    await ctx.send(f"✅ ~~#{thread['id']} {thread['title']}~~ -- {thread['how']}")


@threads.command(name="reopen")
async def threads_reopen(ctx: commands.Context, thread_id: int):
    """Reopen a resolved thread: !threads reopen 3"""
    guild = await _resolve_guild(ctx)
    data = memory.load_threads()
    thread = memory.reopen_thread(data, thread_id)
    if thread is None:
        await ctx.send(f"Thread #{thread_id} isn't resolved.")
        return
    memory.save_threads(data)
    if guild:
        await _refresh_board(guild, data)
    await ctx.send(f"Reopened ⬜ **#{thread['id']}** {thread['title']}")


@threads.command(name="scan")
async def threads_scan(ctx: commands.Context):
    """Build threads and quotes from the last session's transcript and recap."""
    guild = await _resolve_guild(ctx)
    channel = _find_transcription_channel(guild) if guild else None
    if channel is None:
        return
    # Anchored on the latest recap rather than "now", so a session from days
    # ago can still be scanned -- the transcript it was written from, and the
    # recap as corrected.
    group = await _find_last_recap(channel)
    resolved = await _resolve_recap_start(channel, before=group[0]) if group else None
    if resolved is not None:
        start = resolved[0]
        transcript = await _collect_transcript(channel, start, until=group[0])
        recap_text = "\n".join(e.description or "" for m in group for e in m.embeds)
    else:
        start = await _resolve_session_start(channel)
        if start is None:
            await ctx.send("I can't find a recap, or a session in the last day, to scan.")
            return
        transcript = await _collect_transcript(channel, start)
        recap_text = ""
    await ctx.send(f"Scanning {len(transcript):,} characters of the last session...")
    await _update_campaign_memory(ctx.channel, guild, start, transcript, recap_text)


def _briefing_for(session_start: datetime.datetime | None) -> str:
    """Campaign notes, plus anything the table has corrected about this session,
    so re-running its recap or rescanning it doesn't bring the mistakes back."""
    notes = build_recap_briefing()
    fixes = memory.corrections_for(session_start) if session_start else ""
    if fixes:
        notes += ("\n\nThe table corrected an earlier recap of this session. They were "
                  "there; treat these as fact even where the transcript disagrees:\n" + fixes)
    return notes


async def _find_last_recap(channel: discord.TextChannel) -> list[discord.Message]:
    """The newest party recap: its run of consecutive embed messages, oldest first."""
    group: list[discord.Message] = []
    async for message in channel.history(limit=1000):
        # "(continued)" overflow from an earlier correction belongs to the group.
        if message.author == bot.user and any(
                (e.title or "").startswith(RECAP_TITLE) for e in message.embeds):
            group.append(message)
        elif group:
            break
    return list(reversed(group))


async def _rewrite_recap(group: list[discord.Message], text: str, author: str) -> None:
    """Edit the posted recap in place. Same messages, same title -- so it stays
    the boundary marker the next recap scans for -- with a footer saying who
    corrected it."""
    chunks = [text[i:i + 4000] for i in range(0, len(text), 4000)] or [""]
    for i, message in enumerate(group):
        embed = message.embeds[0].copy()
        footer = re.sub(r"\s*·\s*corrected by .*$", "", (embed.footer.text or "") if embed.footer else "")
        embed.set_footer(text=f"{footer} · corrected by {author}" if footer else f"corrected by {author}")
        embed.description = chunks[i] if i < len(chunks) else "(merged into the message above)"
        await message.edit(embed=embed)
    for extra in chunks[len(group):]:
        # Deliberately not RECAP_TITLE: a new boundary marker posted now would
        # make the next recap skip everything said since the original.
        await group[-1].channel.send(embed=discord.Embed(
            title=f"{RECAP_TITLE} (continued)", description=extra, colour=RECAP_COLOUR))


async def _rerender_quote_post(guild: discord.Guild, posted_message_id: int) -> None:
    channel = discord.utils.get(guild.text_channels, name=QUOTES_CHANNEL_NAME)
    if channel is None:
        return
    blocks = [_quote_block(q["speaker"], q["text"], q["date"]) for q in memory.quotes_in_post(posted_message_id)]
    if blocks:
        message = await channel.fetch_message(posted_message_id)
        await message.edit(content="\n\n".join(blocks))


@bot.command(aliases=["fix"])
async def correct(ctx: commands.Context, *, corrections: str = ""):
    """Fix the latest recap. DM me what it got wrong and I'll edit it in place.

    e.g. !correct The DM said "you're terrible at practicing law", not Sykth.
    The five-headed screamer IS the teen.
    """
    if not corrections.strip():
        await ctx.send('Tell me what the recap got wrong, e.g. `!correct The DM said "you\'re '
                       'terrible at practicing law", not Sykth.` Several fixes in one message is fine.')
        return
    guild = await _resolve_guild(ctx)
    if guild is None:
        return
    channel = _find_transcription_channel(guild)
    group = await _find_last_recap(channel) if channel else []
    if not group:
        await ctx.send("I can't find a recap to correct.")
        return
    resolved = await _resolve_recap_start(channel, before=group[0])
    if resolved is None:
        await ctx.send("I can't find the transcript that recap was written from.")
        return
    start = resolved[0]

    recap_text = "\n".join(e.description or "" for m in group for e in m.embeds)
    transcript = await _collect_transcript(channel, start, until=group[0])
    data = memory.load_threads()
    quotes = memory.quotes_since(start.astimezone().date().isoformat())
    status = await ctx.send("Applying your corrections -- this takes a minute or so...")
    try:
        result = await asyncio.to_thread(
            claude.correct_recap, recap_text, corrections, transcript,
            memory.open_threads(data), quotes, _briefing_for(start),
        )
        await _rewrite_recap(group, result["recap"], ctx.author.display_name)
    except Exception as exc:
        _events.error("recap correction failed", exc_info=exc)
        await _say(status, ctx, f"Couldn't apply that ({exc}). The recap hasn't been changed.")
        return

    # Kept, so a re-run of this recap, a rescan, or a question about the
    # session all get it right too -- not just this one message.
    memory.add_correction(start, ctx.author.display_name, corrections)
    memory.save_recap(result["recap"], start)
    threads_changed = [t for t in (memory.update_thread(data, u["id"], u["title"], u["detail"])
                                   for u in result["thread_updates"]) if t]
    if threads_changed:
        memory.save_threads(data)
        try:
            await _refresh_board(guild, data)
        except discord.HTTPException as exc:
            _events.warning("couldn't update the threads board after a correction: %s", exc)
    quotes_fixed = 0
    for fix in result["quote_fixes"]:
        post = memory.fix_quote_speaker(fix["index"], fix["speaker"])
        quotes_fixed += 1
        if post:
            try:
                await _rerender_quote_post(guild, post)
            except discord.HTTPException as exc:
                _events.warning("couldn't edit a quote after a correction: %s", exc)
    _events.info("recap corrected by %s: %d changes, %d threads, %d quotes",
                 ctx.author, len(result["changes"]), len(threads_changed), quotes_fixed)

    extras = []
    if threads_changed:
        extras.append(f"{len(threads_changed)} thread(s) on the board")
    if quotes_fixed:
        extras.append(f"{quotes_fixed} quote(s) re-credited")
    changes = "\n".join(f"• {c}" for c in result["changes"]) or "• Nothing needed changing."
    reply = f"✏️ Recap corrected: {group[0].jump_url}\n{changes}"
    if extras:
        reply += "\nAlso fixed: " + ", ".join(extras) + "."
    await _say(status, ctx, reply[:1990])


def _parse_recap_args(tokens: list[str]) -> tuple[int | None, str | None, str | None]:
    """Split `!recap` arguments by shape rather than position.

    Order doesn't matter, which is the point: a number is a time window, a
    recognised word is a style, a mention is resolved separately as the DM
    target, and whatever is left is a character name. `!recap short 60` and
    `!recap 60 short` are the same command.
    """
    minutes: int | None = None
    style: str | None = None
    focus_parts: list[str] = []
    for token in tokens:
        if token.isdigit():
            minutes = int(token)
        elif _MENTION_RE.fullmatch(token):
            continue  # resolved from ctx.message.mentions by the caller
        elif resolve_style(token):
            style = resolve_style(token)
        else:
            cleaned = token.strip(",.")
            if cleaned:
                focus_parts.append(cleaned)
    return minutes, style, " ".join(focus_parts) or None


@bot.command()
async def recap(ctx: commands.Context, *, args: str = ""):
    """Summarizes the session. !recap [minutes] [style] [character]

    Examples:
      !recap                 everything since the last recap
      !recap 90              the last 90 minutes, literally
      !recap funny           the whole session, played for laughs
      !recap short 60        a tight recap of the last hour
      !recap Sykth           a personal recap for Sykth, sent to your DMs
      !recap @Justin Llyr    Llyr's personal recap, DM'd to Justin
      !recap styles          list the available styles
    """
    tokens = args.split()

    if tokens and tokens[0].lower() in ("styles", "style", "help"):
        listed = "\n".join(
            f"`{name}` -- {text.split('.')[0]}." for name, text in RECAP_STYLES.items()
        )
        await ctx.send(
            f"**Recap styles**\n{listed}\n\n"
            f"Combine freely: `!recap sarcastic 90`. Add a character name for a "
            f"personal recap in your DMs: `!recap Sykth long`."
        )
        return

    minutes, style, focus = _parse_recap_args(tokens)

    if minutes is not None and not (1 <= minutes <= MAX_LOOKBACK_HOURS * 60):
        await ctx.send(f"Pick something between 1 and {MAX_LOOKBACK_HOURS * 60} minutes.")
        return

    # A mention means "DM it to them"; a bare character name means "DM it to
    # me". Deliberately never DMs a third party without being named -- an
    # unexpected DM from a bot is a nuisance.
    target = None
    if ctx.message.mentions:
        target = ctx.message.mentions[0]
    elif focus:
        target = ctx.author

    guild = await _resolve_guild(ctx)
    if guild is None:
        return  # _resolve_guild has already explained why
    text_channel = _find_transcription_channel(guild)
    if text_channel is None:
        await ctx.send(f"No `#{TRANSCRIPTION_CHANNEL_NAME}` channel to read.")
        return

    if minutes is not None:
        # An explicit window is taken literally -- if you asked for the last
        # 60 minutes you get 60 minutes, even if that re-covers old ground.
        since, reason = discord.utils.utcnow() - datetime.timedelta(minutes=minutes), "window"
    elif focus:
        # Personal recaps don't post a boundary marker, so "since the last
        # recap" would give them a sliver of the night. Cover the session.
        resolved = await _resolve_session_start(text_channel)
        if resolved is None:
            await ctx.send("Nothing transcribed yet to recap.")
            return
        since, reason = resolved, "session"
    else:
        resolved = await _resolve_recap_start(text_channel)
        if resolved is None:
            await ctx.send(
                f"Nothing transcribed in #{text_channel.name} yet -- "
                f"use `!record` first, or `!recap <minutes>` to look further back."
            )
            return
        since, reason = resolved

    await _post_recap(ctx, text_channel, since, reason, style=style, focus=focus, target=target)


def _is_recap(message: discord.Message) -> bool:
    return any(embed.title == RECAP_TITLE for embed in message.embeds)


def _is_transcript_line(message: discord.Message) -> bool:
    """True for the bot's own `**Name:** text` transcript messages.

    Recaps are embeds, so they have no content and fail this test -- which
    is what keeps a recap from being read back in as if someone had said it.
    """
    return message.author.id == bot.user.id and message.content.startswith("**")


async def _resolve_session_start(text_channel: discord.TextChannel) -> datetime.datetime | None:
    """Start of the current session, ignoring any recaps already posted.

    Personal recaps don't post a boundary marker, so resolving them against
    the last party recap would hand someone a recap of the four minutes
    since it was written. They want the night.
    """
    resolved = await _resolve_recap_start(text_channel, stop_at_recap=False)
    return resolved[0] if resolved else None


async def _resolve_recap_start(
    text_channel: discord.TextChannel,
    stop_at_recap: bool = True,
    before: discord.Message | None = None,
) -> tuple[datetime.datetime, str] | None:
    """Find where the next recap should begin, reading the channel itself.

    Walks backwards from now and stops at whichever comes first:

      * the last recap we posted -- so a second `!recap` covers only what
        has happened since the first, instead of retelling the whole night;
      * a gap longer than SESSION_GAP_MINUTES -- the start of the current
        session, which is how a recap still covers the *whole* session after
        the bot has been restarted partway through it.

    Whichever boundary is more recent is the one the scan reaches first, so
    a stale recap from last week doesn't win over this week's session start.
    Returns None when there's no transcript in range at all.
    """
    # `before` anchors the scan at an older recap -- "where did *that* recap
    # start?" -- which is what !correct needs to find its transcript.
    anchor = before.created_at if before is not None else discord.utils.utcnow()
    horizon = anchor - datetime.timedelta(hours=MAX_LOOKBACK_HOURS)
    gap = datetime.timedelta(minutes=SESSION_GAP_MINUTES)

    session_start: datetime.datetime | None = None
    newer: datetime.datetime | None = None  # the message examined just before this one

    async for message in text_channel.history(limit=None, after=horizon, before=before, oldest_first=False):
        if not (_is_recap(message) or _is_transcript_line(message)):
            continue
        # The gap check has to come first, and has to apply to recaps too:
        # a recap from *last* week's session sits on the far side of a gap,
        # and must not be mistaken for the boundary of tonight's.
        if newer is not None and (newer - message.created_at) > gap:
            break  # everything older than this belongs to an earlier session
        if _is_recap(message) and stop_at_recap:
            _log.info("recap boundary: previous recap at %s", message.created_at)
            return message.created_at, "recap"
        newer = message.created_at
        session_start = message.created_at

    if session_start is None:
        return None
    # Step back a second so the boundary message itself is included --
    # history(after=...) is exclusive.
    _log.info("recap boundary: session appears to start at %s", session_start)
    return session_start - datetime.timedelta(seconds=1), "session"


async def _collect_transcript(
    text_channel: discord.TextChannel, since: datetime.datetime, until=None
) -> str:
    """Read back the transcript this bot posted since a given time (and,
    optionally, before a given message).

    Only our own transcript lines count -- players chatting in the channel
    aren't part of the session record, and neither are previous recaps. Each
    "**wildgoose:** ..." becomes "Thia (Goose): ...", so the model never has
    to work out who plays whom.
    """
    lines: list[str] = []
    speakers = load_speakers()
    async for message in text_channel.history(limit=None, after=since, before=until, oldest_first=True):
        if not _is_transcript_line(message):
            continue
        content = _labelled_line(message.content, speakers)
        if content:
            lines.append(content)
    return "\n".join(lines)


async def _say(status: discord.Message, ctx: commands.Context, content: str) -> None:
    """Report to the user without assuming the network is up.

    The failure this exists for: a recap died because the machine dropped
    off wifi, the handler tried to edit the status message to say so, and
    *that* call raised too -- so the error handler became the error, and the
    user saw nothing at all. Reporting a network failure over the network
    needs a fallback and a shrug.
    """
    for attempt in (0, 1):
        try:
            await status.edit(content=content)
            return
        except discord.HTTPException:
            if attempt == 0:
                await asyncio.sleep(5)  # a reconnect is often seconds away
    try:
        await ctx.send(content)
    except discord.HTTPException:
        _log.error("could not report to Discord at all: %s", content)


async def _post_recap(
    ctx: commands.Context,
    text_channel: discord.TextChannel,
    since: datetime.datetime,
    reason: str = "window",
    style: str | None = None,
    focus: str | None = None,
    target: discord.abc.User | None = None,
):
    transcript = await _collect_transcript(text_channel, since)
    if len(transcript) < 200:  # nothing meaningful was said
        # Say *why* the window is empty. "Not enough transcript" on its own
        # reads as "the channel is empty" when the real cause is usually
        # that everything before this point has already been recapped.
        mins = (discord.utils.utcnow() - since).total_seconds() / 60
        explanation = {
            "recap": (
                f"I've only got {len(transcript)} characters since the last recap "
                f"({mins:.0f} min ago) -- not enough to write another one.\n"
                f"To recap further back anyway, give me a window: `!recap {max(int(mins) + 60, 120)}`."
            ),
            "session": (
                f"This session only has {len(transcript)} characters of transcript "
                f"so far ({mins:.0f} min) -- not enough to summarize yet."
            ),
        }.get(reason, f"Only {len(transcript)} characters in the last {mins:.0f} minutes -- not enough to summarize.")
        await ctx.send(explanation)
        return

    # A full session runs several Opus passes at max effort and takes
    # minutes. A typing indicator alone looks identical to a hung bot, so
    # post a status message and edit it as each pass lands.
    passes = claude.plan_passes(transcript)
    described = ", ".join(
        p for p in (f"{style} style" if style else "", f"focused on {focus}" if focus else "") if p
    )
    status = await ctx.send(
        f"Reading {len(transcript):,} characters of transcript "
        f"({passes} passes, a few minutes)"
        + (f" -- {described}" if described else "")
        + "..."
    )
    loop = asyncio.get_running_loop()

    def on_progress(done: int, total: int, label: str) -> None:
        # Runs on the worker thread -- hand the edit back to the event loop.
        asyncio.run_coroutine_threadsafe(
            status.edit(content=f"Recap {done}/{total}: {label}..."), loop
        )

    try:
        # Hand the recap everything we know about this campaign: who the
        # Discord display names belong to, and which names the transcriber
        # mangles. The model rewrites the text wholesale anyway, so it can
        # fix "clear" back to "Llyr" from context -- something a blind
        # find-and-replace over the transcript could never do safely.
        summary = await asyncio.to_thread(
            claude.summarize_session,
            transcript,
            on_progress,
            _briefing_for(since),
            style,
            focus,
        )
    except Exception as exc:
        _log.error("session recap failed", exc_info=exc)
        await _say(
            status, ctx,
            f"Couldn't generate the recap: {exc}"
            f"\nThe transcript is still in #{text_channel.name} -- "
            "run `!recap` again once you are back online and it will "
            "pick up the same session.",
        )
        return

    # Belt and braces: _complete already rejects empty output, but an empty
    # summary here would make the send loop below iterate zero times and post
    # nothing at all, with no error -- which is exactly how a recap once
    # disappeared silently.
    if not summary.strip():
        _log.error("recap came back empty from %d chars of transcript", len(transcript))
        await _say(status, ctx, "The recap came back empty -- nothing posted. Check bot.log.")
        return

    # Every chunk carries the title, so any of them works as the boundary
    # marker for the next recap -- the scan finds the most recent one first.
    covered = f"Covers {since:%d %b %H:%M} - {discord.utils.utcnow():%H:%M} UTC"
    if style:
        covered += f" | {style}"

    # A personal recap goes to a DM and is titled differently, deliberately:
    # RECAP_TITLE is the boundary marker the next party recap scans for, and
    # a per-character recap must not move that boundary.
    if target is not None:
        destination, title = target, f"Your session recap — {focus}" if focus else "Your session recap"
    else:
        destination, title = text_channel, RECAP_TITLE

    sent = 0
    try:
        for start in range(0, len(summary), 4000):  # embed descriptions cap at 4096
            embed = discord.Embed(
                title=title,
                description=summary[start:start + 4000],
                colour=RECAP_COLOUR,
            )
            embed.set_footer(text=covered)
            await destination.send(embed=embed)
            sent += 1
    except discord.Forbidden:
        _log.warning("could not DM %s -- DMs closed", getattr(destination, "name", destination))
        await status.edit(
            content=f"I can't DM {getattr(destination, 'display_name', 'them')} -- "
            f"their DMs are closed to server members. Nothing was posted."
        )
        return

    # Logged *after* the send, and reporting what actually went out. The
    # previous version logged "posted recap" before sending anything, so the
    # log claimed success on a night when nothing reached the channel.
    _events.info(
        "posted recap: %d embed(s)%s%s, %d chars of summary from %d chars of transcript",
        sent,
        f", style={style}" if style else "",
        f", focus={focus!r} -> DM" if target is not None else "",
        len(summary), len(transcript),
    )

    # A party recap of the night (not a personal one, not a literal `!recap 60`
    # window) also updates the campaign's memory: the archive, the threads
    # board and the quote board. In the background, so the recap isn't held up.
    if target is None and reason in ("session", "recap"):
        asyncio.create_task(_update_campaign_memory(ctx.channel, text_channel.guild, since, transcript, summary))

    # Always say where it went. Silently posting somewhere the invoker isn't
    # looking is exactly how a working recap reads as a failed one.
    if target is not None:
        who = "your DMs" if target.id == ctx.author.id else f"{target.display_name}'s DMs"
        await status.edit(content=f"Sent {focus or 'the recap'} to {who}.")
    elif ctx.channel.id == text_channel.id:
        await status.delete()
    else:
        await status.edit(content=f"Recap posted in #{text_channel.name}.")
