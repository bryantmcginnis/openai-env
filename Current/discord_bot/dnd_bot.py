# Requires the 'Server Members' and 'Message Content' privileged intents to
# be enabled for this bot in the Discord Developer Portal.

import asyncio
import datetime
import logging
import os
import re
# import time  # only used by the disabled _dump_debug_wav helper
# import wave  # only used by the disabled _dump_debug_wav helper

import discord
from discord.ext import commands
from dotenv import load_dotenv

from claude_caller.claude_caller import RECAP_STYLES, ClaudeChatCaller, resolve_style
from discord_bot import dice
from discord_bot.transcript_writer import TranscriptWriter
from transcription.audio_sink import UtteranceSink
from transcription.dnd_vocabulary import build_recap_briefing
from transcription.transcriber import Transcriber

load_dotenv()

_log = logging.getLogger(__name__)

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
    _log.info("Logged in as %s (ID: %s)", bot.user, bot.user.id)
    # on_ready can fire again on a gateway resume; this is idempotent and
    # only acts when a voice state has no connection behind it.
    await _clear_stale_voice_states()


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
        try:
            reply = await asyncio.to_thread(claude.call, content, message.channel.id)
        except Exception as exc:  # surface API errors at the table instead of dropping them silently
            reply = f"(Claude API error: {exc})"

    for start in range(0, len(reply), 2000):  # Discord's message length limit
        await message.channel.send(reply[start:start + 2000])


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

    voice_client = await ctx.author.voice.channel.connect()

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

    loop = asyncio.get_running_loop()

    writer = TranscriptWriter(text_channel)
    _transcript_writers[ctx.guild.id] = writer

    async def on_utterance(user_id: int, pcm_bytes: bytes):
        await _handle_utterance(ctx.guild, writer, user_id, pcm_bytes)

    def on_recording_finished(exception: Exception | None):
        # Runs on py-cord's audio thread; hop back to the event loop for cleanup.
        _active_recordings.discard(ctx.guild.id)
        # The sink flushes any half-finished utterance during cleanup, so
        # close the writer behind it rather than here, once those have been
        # transcribed and appended.
        asyncio.run_coroutine_threadsafe(_close_writer(ctx.guild.id), loop)
        if exception:
            asyncio.run_coroutine_threadsafe(
                ctx.send(f"Recording stopped with an error: {exception}"), loop
            )

    # Recorded just before the first utterance can land, so the recap picks
    # up everything from this session and nothing from the last one.
    _recording_started_at[ctx.guild.id] = discord.utils.utcnow() - datetime.timedelta(seconds=5)

    sink = UtteranceSink(loop, on_utterance)
    # py-cord 2.8.0 no longer calls sink.init() itself anywhere in the voice
    # module -- without this, self.sink.client stays None and PacketDecoder
    # crashes looking up the speaking member.
    sink.init(voice_client)
    _active_recordings.add(ctx.guild.id)
    voice_client.start_recording(sink, on_recording_finished)

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

    if ctx.guild.id in _active_recordings:
        voice_client.stop_recording()
    await voice_client.disconnect()
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
    horizon = discord.utils.utcnow() - datetime.timedelta(hours=MAX_LOOKBACK_HOURS)
    gap = datetime.timedelta(minutes=SESSION_GAP_MINUTES)

    session_start: datetime.datetime | None = None
    newer: datetime.datetime | None = None  # the message examined just before this one

    async for message in text_channel.history(limit=None, after=horizon, oldest_first=False):
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


async def _collect_transcript(text_channel: discord.TextChannel, since: datetime.datetime) -> str:
    """Read back the transcript this bot posted since a given time.

    Only our own transcript lines count -- players chatting in the channel
    aren't part of the session record, and neither are previous recaps. The
    bold "**Name:**" prefix is stripped back to a plain "Name: line".
    """
    lines: list[str] = []
    async for message in text_channel.history(limit=None, after=since, oldest_first=True):
        if not _is_transcript_line(message):
            continue
        content = message.content.replace("**", "").strip()
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
            build_recap_briefing(),
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
    _log.info(
        "posted recap: %d embed(s)%s%s, %d chars of summary from %d chars of transcript",
        sent,
        f", style={style}" if style else "",
        f", focus={focus!r} -> DM" if target is not None else "",
        len(summary), len(transcript),
    )

    # Always say where it went. Silently posting somewhere the invoker isn't
    # looking is exactly how a working recap reads as a failed one.
    if target is not None:
        who = "your DMs" if target.id == ctx.author.id else f"{target.display_name}'s DMs"
        await status.edit(content=f"Sent {focus or 'the recap'} to {who}.")
    elif ctx.channel.id == text_channel.id:
        await status.delete()
    else:
        await status.edit(content=f"Recap posted in #{text_channel.name}.")
