import asyncio
import concurrent.futures
import logging
from collections import defaultdict
from typing import Awaitable, Callable

import numpy as np
from discord.sinks import Sink

_log = logging.getLogger(__name__)

# Matches discord.opus.Decoder: 48kHz, stereo, 16-bit PCM.
SAMPLE_RATE = 48000
CHANNELS = 2
SAMPLE_WIDTH = 2
BYTES_PER_MS = SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH // 1000  # 192, exact

SILENCE_RMS_THRESHOLD = 300.0  # int16 RMS below this counts as silence
# Pause length that ends an utterance. Lower = lower latency (transcription
# starts sooner after someone stops talking) but chops mid-sentence more
# readily when someone pauses to think, which costs Whisper some context.
# 400ms is roughly a natural sentence-boundary pause -- too tight in
# practice for a DM narrating at length, who pauses for breath and for
# effect mid-thought. 900ms rides through those and only ends the utterance
# on a real stop. Continuations are stitched back together downstream by
# discord_bot/transcript_writer.py, so a higher value here costs a little
# latency but no longer means a separate message per fragment.
SILENCE_HANG_MS = 900
MIN_UTTERANCE_MS = 300  # discard shorter blips (coughs, mic taps)
# Hard cap that force-flushes a buffer even without a pause, so a
# continuous talker still gets transcribed in chunks instead of nothing
# until they stop. This is the worst-case latency ceiling -- keep it low
# enough that a long monologue still shows up reasonably promptly.
MAX_UTTERANCE_MS = 12_000

OnUtterance = Callable[[int, bytes], Awaitable[None]]


def _rms(pcm_bytes: bytes) -> float:
    if not pcm_bytes:
        return 0.0
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0


class UtteranceSink(Sink):
    """A py-cord recording sink that segments each speaker's audio on
    natural pauses instead of fixed-size chunks.

    Feeding Whisper tiny fixed-size windows is a common cause of "mishearing
    everything" -- words get cut in half at chunk boundaries and Whisper has
    no sentence context to disambiguate homophones or proper nouns. Buffering
    per-user audio until a real pause in speech, then transcribing the whole
    utterance at once, fixes most of that.
    """

    # py-cord 2.8.0's SinkEventRouter (discord/voice/receive/router.py) reads
    # these off every sink unconditionally, but discord.sinks.Sink -- the
    # base class, including py-cord's own built-in sinks -- never defines
    # them. Without this, start_recording() crashes before a single packet
    # arrives. We don't need the new per-event-listener system: packets
    # still reach us the old way, via PacketRouter calling write() directly.
    __sink_listeners__ = ()

    def walk_children(self):
        return []

    def is_opus(self) -> bool:
        # Tells discord/opus.py's PacketDecoder whether to hand us raw Opus
        # frames or decode them first -- we want decoded PCM (write() below
        # feeds it straight to numpy/Whisper), so False.
        return False

    def __init__(self, loop: asyncio.AbstractEventLoop, on_utterance: OnUtterance):
        super().__init__()
        self._loop = loop
        self._on_utterance = on_utterance
        self._buffers: dict[int, bytearray] = defaultdict(bytearray)
        self._silence_ms: dict[int, float] = defaultdict(float)

    def write(self, voice_data, member) -> None:
        # Deliberately does not call super().write() -- the base
        # implementation writes to on-disk AudioData files for exporting a
        # full recording, which isn't what we want here.
        #
        # py-cord 2.8.0's PacketRouter passes a VoiceData wrapper plus a
        # Member/User object here instead of the old (pcm_bytes, user_id)
        # pair (discord/voice/receive/router.py) -- unwrap both.
        if member is None:
            return
        user = member.id
        data = voice_data.pcm

        buffer = self._buffers[user]
        chunk_ms = len(data) / BYTES_PER_MS

        if _rms(data) < SILENCE_RMS_THRESHOLD:
            self._silence_ms[user] += chunk_ms
        else:
            self._silence_ms[user] = 0.0

        buffer.extend(data)
        buffer_ms = len(buffer) / BYTES_PER_MS

        should_flush = (
            self._silence_ms[user] >= SILENCE_HANG_MS and buffer_ms >= MIN_UTTERANCE_MS
        ) or buffer_ms >= MAX_UTTERANCE_MS

        if should_flush:
            self._flush(user)

    def _flush(self, user: int) -> None:
        buffer = self._buffers.pop(user, None)
        self._silence_ms.pop(user, None)
        if not buffer:
            return

        # Trim the trailing silence tail so it isn't fed to Whisper as content.
        trailing_bytes = int(SILENCE_HANG_MS * BYTES_PER_MS)
        pcm_bytes = bytes(buffer[:-trailing_bytes]) if len(buffer) > trailing_bytes else bytes(buffer)
        if len(pcm_bytes) < MIN_UTTERANCE_MS * BYTES_PER_MS:
            return

        # write() runs on py-cord's audio-receive thread, not the event
        # loop -- hand the coroutine back to the loop instead of awaiting it.
        future = asyncio.run_coroutine_threadsafe(self._on_utterance(user, pcm_bytes), self._loop)
        # Nobody ever called .result()/.exception() on this Future, so any
        # exception raised inside on_utterance (transcription errors,
        # Discord API errors, etc.) was being silently discarded.
        future.add_done_callback(self._log_utterance_exception)

    @staticmethod
    def _log_utterance_exception(future: concurrent.futures.Future) -> None:
        exc = future.exception()
        if exc is not None:
            _log.error("on_utterance callback raised", exc_info=exc)

    def cleanup(self) -> None:
        for user in list(self._buffers.keys()):
            self._flush(user)
        super().cleanup()
