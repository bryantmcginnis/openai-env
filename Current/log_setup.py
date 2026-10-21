"""Logging for the bot: console, a rolling bot.log, and a slow bot_events.log.

Why this is more than basicConfig: on 2026-10-14 the voice receiver logged a
full traceback for every packet it couldn't decrypt -- about 60 a second --
and filled all 20 MB of bot.log's rotation in six minutes, overwriting the
hour that would have shown what broke. Two defences:

* RateLimitFilter: a repeated WARNING or ERROR is logged in full once, then
  counted, and one summary line every 30 s says how many were suppressed.
  INFO and below always pass -- "transcribed utterance for %s" is a single
  template that must never be throttled.
* bot_events.log keeps only WARNING+ and session milestones (the
  "dndbot.events" logger), rotated on its own, so the next flood in bot.log
  can't take the session timeline with it.
"""

import atexit
import logging
import logging.handlers
import threading
import time
from pathlib import Path

FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
EVENTS_LOGGER = "dndbot.events"
RATE_LIMIT_WINDOW_S = 30.0


class RateLimitFilter(logging.Filter):
    """Let each distinct WARNING/ERROR through once per window; count the rest.

    "Distinct" means same logger, level and message template, so a flood of
    one error is collapsed while a different error still gets through
    immediately. One instance is shared by every handler, and the verdict is
    stored on the record, so each handler sees the same decision for it.
    """

    def __init__(self, window: float = RATE_LIMIT_WINDOW_S):
        super().__init__()
        self.window = window
        self._lock = threading.Lock()
        self._last_emit: dict[tuple, float] = {}
        self._suppressed: dict[tuple, int] = {}
        self._summary = logging.getLogger("dndbot.ratelimit")
        threading.Thread(target=self._flush_loop, name="log-ratelimit", daemon=True).start()

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno < logging.WARNING or getattr(record, "ratelimit_bypass", False):
            return True
        decided = getattr(record, "_ratelimit_pass", None)
        if decided is not None:
            return decided

        key = (record.name, record.levelno, str(record.msg))
        now = time.monotonic()
        with self._lock:
            last = self._last_emit.get(key)
            allow = last is None or now - last >= self.window
            if allow:
                self._last_emit[key] = now
            else:
                self._suppressed[key] = self._suppressed.get(key, 0) + 1
        record._ratelimit_pass = allow
        return allow

    def _flush_loop(self) -> None:
        while True:
            time.sleep(self.window)
            self.flush()

    def flush(self) -> None:
        """Write one summary line per throttled message, then reset the counts."""
        with self._lock:
            pending, self._suppressed = self._suppressed, {}
        for (name, level, msg), count in pending.items():
            self._summary.warning(
                "suppressed %d more of [%s] %s: %s (rate limit: one per %ds)",
                count, logging.getLevelName(level), name, msg[:160], int(self.window),
                extra={"ratelimit_bypass": True},
            )


class _EventsOnly(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.WARNING or record.name.startswith(EVENTS_LOGGER)


def configure_logging(base_dir: Path) -> RateLimitFilter:
    """Set up all logging. Files live next to main.py, not in the cwd -- a
    desktop shortcut and a terminal have different working directories."""
    limiter = RateLimitFilter()
    formatter = logging.Formatter(FORMAT)

    console = logging.StreamHandler()
    main_log = logging.handlers.RotatingFileHandler(
        base_dir / "bot.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    events_log = logging.handlers.RotatingFileHandler(
        base_dir / "bot_events.log", maxBytes=1_000_000, backupCount=5, encoding="utf-8"
    )
    events_log.addFilter(_EventsOnly())

    for handler in (console, main_log, events_log):
        handler.setFormatter(formatter)
        handler.addFilter(limiter)

    # py-cord doesn't configure logging itself -- without this, library
    # messages fall back to logging's bare "handler of last resort".
    logging.basicConfig(level=logging.INFO, handlers=[console, main_log, events_log], force=True)

    # A harmless "Received unexpected rtcp packet type=200" arrives at INFO
    # several times a second while recording. Its ERRORs still come through.
    logging.getLogger("discord.voice.receive.reader").setLevel(logging.WARNING)
    # Two INFO lines per utterance that our own "transcribed utterance" line
    # already covers -- most of the log volume during a session.
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Don't lose the last window's counts on shutdown. Registered after
    # logging's own atexit hook, so it runs first, while handlers are open.
    atexit.register(limiter.flush)
    return limiter
