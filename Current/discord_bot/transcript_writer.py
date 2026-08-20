import asyncio
import logging

import discord

_log = logging.getLogger(__name__)

# How long after a speaker's last utterance their message stays "open" for
# more text to be appended to. Past this, the next thing they say starts a
# fresh message -- otherwise a line from before a ten-minute combat would
# keep growing and read as if it were said just now.
COALESCE_WINDOW_S = 90.0

# Discord's hard limit is 2000 characters per message; leave room for the
# bold "**Name:** " prefix and roll over to a new message before we hit it.
MAX_MESSAGE_CHARS = 1900

# Appends are batched into one edit per this many seconds. Editing on every
# utterance would burn through Discord's per-message edit rate limit during
# a monologue (several utterances can land within a couple of seconds).
EDIT_DEBOUNCE_S = 2.0


class TranscriptWriter:
    """Posts transcribed utterances to a channel, coalescing consecutive
    speech from the same speaker into a single message.

    The sink segments audio on ~1s pauses, which is right for Whisper (it
    wants whole sentences for context) but wrong for the channel: someone
    talking for a minute generates a dozen separate messages. So the first
    utterance posts immediately -- latency is unchanged, text still shows up
    as soon as it's transcribed -- and continuations *edit that message*,
    growing it in place instead of stacking new entries. A different speaker,
    a long gap, or the 2000-character limit rolls over to a new message.
    """

    def __init__(self, channel: discord.abc.Messageable):
        self._channel = channel
        # Guards the open-message state below. _handle_utterance runs as an
        # independent task per utterance, so two can otherwise interleave
        # between the "is this the same speaker" check and the send/edit.
        self._lock = asyncio.Lock()

        self._speaker_id: int | None = None
        self._display_name: str = ""
        self._message: discord.Message | None = None
        self._text: str = ""  # what the open message *should* say
        self._posted_text: str = ""  # what Discord currently shows
        self._last_append: float = 0.0
        self._edit_task: asyncio.Task | None = None

    async def add(self, speaker_id: int, display_name: str, text: str) -> None:
        """Append one transcribed utterance, coalescing where possible."""
        text = text.strip()
        if not text:
            return

        async with self._lock:
            now = asyncio.get_running_loop().time()
            candidate = f"{self._text} {text}" if self._text else text

            can_append = (
                self._message is not None
                and self._speaker_id == speaker_id
                and now - self._last_append <= COALESCE_WINDOW_S
                and len(self._render(display_name, candidate)) <= MAX_MESSAGE_CHARS
            )

            if can_append:
                self._text = candidate
                self._last_append = now
                self._schedule_edit()
                return

            # Rolling over to a new message: make sure whatever is still
            # pending on the old one actually lands before we let go of it.
            await self._flush_locked()
            await self._open(speaker_id, display_name, text, now)

    async def close(self) -> None:
        """Flush any pending edit and stop coalescing. Call when recording ends."""
        async with self._lock:
            await self._flush_locked()
            self._speaker_id = None
            self._message = None
            self._text = ""
            self._posted_text = ""

    # -- internals; all of these assume the caller holds _lock ---------------

    @staticmethod
    def _render(display_name: str, text: str) -> str:
        return f"**{display_name}:** {text}"

    async def _open(self, speaker_id: int, display_name: str, text: str, now: float) -> None:
        self._speaker_id = speaker_id
        self._display_name = display_name
        self._text = text
        self._last_append = now
        self._message = await self._channel.send(self._render(display_name, text))
        self._posted_text = text

    def _schedule_edit(self) -> None:
        # One debounce task at a time -- it reads the latest text when it
        # wakes, so appends that arrive while it sleeps ride along for free.
        if self._edit_task is None or self._edit_task.done():
            self._edit_task = asyncio.create_task(self._edit_after_debounce())

    async def _edit_after_debounce(self) -> None:
        await asyncio.sleep(EDIT_DEBOUNCE_S)
        async with self._lock:
            await self._flush_locked()

    async def _flush_locked(self) -> None:
        if self._edit_task is not None and not self._edit_task.done():
            # Only ever cancelling a task that is still in its sleep -- it
            # takes _lock (held here) before touching anything, so this can't
            # interrupt an in-flight edit.
            self._edit_task.cancel()
        self._edit_task = None

        if self._message is None or self._text == self._posted_text:
            return

        content = self._render(self._display_name, self._text)
        try:
            await self._message.edit(content=content)
        except discord.HTTPException as exc:
            # Losing an edit shouldn't kill the recording session -- fall back
            # to a fresh message so the words still reach the channel.
            _log.warning("transcript edit failed, posting separately instead: %s", exc)
            tail = self._text[len(self._posted_text):].strip()
            if tail:
                self._message = await self._channel.send(self._render(self._display_name, tail))
                self._text = tail
        self._posted_text = self._text
