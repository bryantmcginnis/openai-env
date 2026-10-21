"""Notice when recording has gone deaf, try to fix it, and tell people once.

What this guards against (2026-10-14): about 30 minutes into a session every
voice packet started failing decryption. The bot stayed connected and kept
receiving audio, decoded none of it, and said nothing for 77 minutes.

py-cord's receive reader drops each packet it can't decrypt with no count
and no recovery. It also builds its decryptor once, from the voice key at
the moment recording starts -- the reader has an update_secret_key() method,
but nothing in py-cord calls it -- so if the voice session is re-keyed
mid-recording, every packet after that fails. That is the leading theory;
the logs that would have proven it were lost (see log_setup.py).

Recovery, in order, never repeating a visible step within a session:
  1. Re-sync the reader's key from the live voice connection. Silent and
     instant; if a stale key was the cause, that alone fixes it.
  2. One full voice rejoin per recording session, announced once.
  3. If that doesn't take, flag it once and stop trying, so the group gets
     one clear message rather than a stream of them.
"""

import asyncio
import datetime
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from nacl.exceptions import CryptoError

_log = logging.getLogger(__name__)
_events = logging.getLogger("dndbot.events")

# Recording counts as stalled once packets have been failing for this long
# with none decoding. Long enough to ride out the few seconds of failures a
# normal mid-call encryption change can cause; short enough that a dead
# recording is noticed within a minute instead of at the end of the night.
STALL_SECONDS = 15.0
# ...and at least this many packets must have failed in that time. Silence
# sends no packets, so a quiet table never looks like a stall.
STALL_MIN_FAILURES = 50
CHECK_INTERVAL_S = 5.0
# After each recovery step, how long to wait to see whether audio decodes.
# Any packet decoding means recovered; failures with none decoding means
# not. If nobody speaks at all there's no evidence either way, which is
# treated as recovered -- a recurrence is caught by the next check anyway.
VERDICT_TIMEOUT_S = 60.0


@dataclass
class CaptureHealth:
    """Decrypt counters for one recording, written from py-cord's reader thread."""

    ok: int = 0
    failed: int = 0
    last_ok_at: datetime.datetime = field(default_factory=datetime.datetime.now)

    def take(self) -> tuple[int, int]:
        """Read and reset the counters since the last call."""
        ok, failed = self.ok, self.failed
        self.ok = self.failed = 0
        return ok, failed


def instrument(health: CaptureHealth, voice_client) -> bool:
    """Count successful and failed decrypts on this voice client's reader.

    Wraps the decryptor's decrypt_rtp on the instance: py-cord's reader looks
    it up on every packet, so this sees exactly what it sees without
    changing what it does. Returns False -- stall detection off, recording
    unaffected -- if this py-cord version doesn't have the expected shape.
    """
    decryptor = getattr(getattr(voice_client, "_reader", None), "decryptor", None)
    original = getattr(decryptor, "decrypt_rtp", None)
    if original is None:
        _log.warning("can't watch voice decryption on this py-cord version; stall detection is off")
        return False

    def counted(packet):
        try:
            result = original(packet)
        except CryptoError:
            health.failed += 1
            raise
        health.ok += 1
        health.last_ok_at = datetime.datetime.now()
        return result

    decryptor.decrypt_rtp = counted
    return True


def resync_key(voice_client) -> bool:
    """Hand the reader the voice connection's current key. True if it could."""
    reader = getattr(voice_client, "_reader", None)
    key = getattr(voice_client, "secret_key", None)
    if reader is None or not key or not hasattr(reader, "update_secret_key"):
        return False
    reader.update_secret_key(bytes(key))
    return True


Notify = Callable[[str], Awaitable[object]]          # post a new alert, returns a handle
Update = Callable[[object, str], Awaitable[None]]    # edit that alert in place


class Watchdog:
    """Polls one recording's health and runs the escalation when it stalls."""

    def __init__(
        self,
        health: CaptureHealth,
        *,
        get_voice_client: Callable[[], object],
        rejoin: Callable[[], Awaitable[bool]],
        notify: Notify,
        update: Update,
        guild_name: str = "",
        check_interval: float = CHECK_INTERVAL_S,
        stall_seconds: float = STALL_SECONDS,
        verdict_timeout: float = VERDICT_TIMEOUT_S,
    ):
        self.health = health
        self._get_vc = get_voice_client
        self._rejoin = rejoin
        self._notify = notify
        self._update = update
        self.guild_name = guild_name
        self.check_interval = check_interval
        self.stall_seconds = stall_seconds
        self.verdict_timeout = verdict_timeout
        self.rejoin_used = False
        self.flagged = False

    async def run(self) -> None:
        stalled_for, failures = 0.0, 0
        while not self.flagged:
            await asyncio.sleep(self.check_interval)
            ok, failed = self.health.take()
            if ok or not failed:
                # Decoding fine, or nobody talking: either way, not stalled.
                stalled_for, failures = 0.0, 0
                continue
            stalled_for += self.check_interval
            failures += failed
            if stalled_for >= self.stall_seconds and failures >= STALL_MIN_FAILURES:
                await self._recover(failures)
                stalled_for, failures = 0.0, 0

    async def _decoding_resumed(self) -> bool:
        self.health.take()
        waited = 0.0
        while waited < self.verdict_timeout:
            await asyncio.sleep(min(1.0, self.check_interval))
            waited += min(1.0, self.check_interval)
            if self.health.ok:
                return True
            if self.health.failed >= STALL_MIN_FAILURES:
                return False
        # Out of time. Any failure with no success means still broken; a
        # silent table means no evidence either way, treated as fine (the
        # next check catches a recurrence).
        return self.health.failed == 0

    async def _recover(self, failures: int) -> None:
        since = self.health.last_ok_at.strftime("%H:%M")
        _events.warning(
            "audio stalled in %s: %d packets failed decryption since %s with none decoding",
            self.guild_name, failures, since,
        )

        # 1. Silent: re-sync the decryption key.
        if resync_key(self._get_vc()):
            _events.warning("re-synced the voice decryption key from the live connection")
            if await self._decoding_resumed():
                _events.warning("audio recovered by key re-sync; transcript gap began %s", since)
                return

        # 2. One full rejoin per recording session, announced once.
        notice = None
        if not self.rejoin_used:
            self.rejoin_used = True
            notice = await self._notify(
                f"⚠️ **Transcription problem:** I've been hearing the voice channel since "
                f"{since} but can't decode the audio, so nothing is being transcribed. "
                f"Rejoining voice to fix it -- you'll see me drop out for a few seconds."
            )
            _events.warning("rejoining voice in %s to recover", self.guild_name)
            if await self._rejoin():
                if await self._decoding_resumed():
                    now = datetime.datetime.now().strftime("%H:%M")
                    _events.warning("audio recovered after rejoin; transcript gap %s-%s", since, now)
                    await self._update(
                        notice,
                        f"✅ **Transcription back** at {now} after rejoining voice. "
                        f"The transcript is missing roughly {since}–{now}.",
                    )
                    return
                still_connected = True
            else:
                still_connected = False
        else:
            still_connected = self._get_vc() is not None

        # 3. Flag once, then stop trying.
        self.flagged = True
        action = (
            "Run `!stoprecording` then `!record` to reconnect"
            if still_connected
            else "I'm out of the voice channel -- run `!record` to start again"
        )
        text = (
            f"❌ **Transcription stopped** at about {since} and I couldn't recover it on my own. "
            f"{action}. I won't keep retrying, so this is the only message you'll get about it."
        )
        _events.error("could not recover audio in %s; flagged to the table", self.guild_name)
        if notice is not None:
            await self._update(notice, text)
        else:
            await self._notify(text)
