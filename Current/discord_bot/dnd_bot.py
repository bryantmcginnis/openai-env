# Requires the 'Server Members' and 'Message Content' privileged intents to
# be enabled for this bot in the Discord Developer Portal.

import asyncio
import logging
import os
import random
# import time  # only used by the disabled _dump_debug_wav helper
# import wave  # only used by the disabled _dump_debug_wav helper

import discord
from discord.ext import commands
from dotenv import load_dotenv

from claude_caller.claude_caller import ClaudeChatCaller
from transcription.audio_sink import UtteranceSink
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


@bot.event
async def on_ready():
    _log.info("Logged in as %s (ID: %s)", bot.user, bot.user.id)


@bot.event
async def on_command_error(ctx: commands.Context, error: commands.CommandError):
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

    await bot.process_commands(message)

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


@bot.command()
async def roll(ctx: commands.Context, dice: str):
    """Rolls dice in NdN format, e.g. !roll 2d6."""
    try:
        rolls, limit = map(int, dice.lower().split("d"))
    except ValueError:
        await ctx.send("Format has to be NdN, e.g. `2d6`.")
        return

    if not (1 <= rolls <= 100 and 1 <= limit <= 1000):
        await ctx.send("Keep it reasonable -- up to 100 dice, up to d1000.")
        return

    results = [random.randint(1, limit) for _ in range(rolls)]
    await ctx.send(f"{', '.join(map(str, results))} (total: {sum(results)})")


# ---------------------------------------------------------------------------
# Voice: !record joins your voice channel and transcribes it live to the
# #transcription channel; !stoprecording leaves.
# ---------------------------------------------------------------------------

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


async def _handle_utterance(
    guild: discord.Guild, text_channel: discord.TextChannel, user_id: int, pcm_bytes: bytes
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

    await text_channel.send(f"**{display_name}:** {text}")


@bot.command()
async def record(ctx: commands.Context):
    """Joins your voice channel and transcribes speech to the transcription channel."""
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

    async def on_utterance(user_id: int, pcm_bytes: bytes):
        await _handle_utterance(ctx.guild, text_channel, user_id, pcm_bytes)

    def on_recording_finished(exception: Exception | None):
        # Runs on py-cord's audio thread; hop back to the event loop for cleanup.
        _active_recordings.discard(ctx.guild.id)
        if exception:
            asyncio.run_coroutine_threadsafe(
                ctx.send(f"Recording stopped with an error: {exception}"), loop
            )

    sink = UtteranceSink(loop, on_utterance)
    # py-cord 2.8.0 no longer calls sink.init() itself anywhere in the voice
    # module -- without this, self.sink.client stays None and PacketDecoder
    # crashes looking up the speaking member.
    sink.init(voice_client)
    _active_recordings.add(ctx.guild.id)
    voice_client.start_recording(sink, on_recording_finished)

    await ctx.send(
        f"Recording on {transcriber.device.upper()} -- transcripts go to "
        f"#{text_channel.name}. Say `!stoprecording` when done."
    )


@bot.command()
async def stoprecording(ctx: commands.Context):
    """Stops transcribing and leaves the voice channel."""
    voice_client = ctx.guild.voice_client
    if voice_client is None or not voice_client.is_connected():
        await ctx.send("Not currently recording.")
        return

    if ctx.guild.id in _active_recordings:
        voice_client.stop_recording()
    await voice_client.disconnect()
    await ctx.send("Stopped recording.")
