"""Keep the bot running: restart it after a crash, never after !shutdown.

main.py exits 0 only when someone ran !shutdown; anything else -- a crash,
an unexpected stop, the process being killed -- gets a restart after 10 s,
then 30 s, then 60 s. Five crashes inside 15 minutes is a crash loop rather
than bad luck, so this stops and leaves the error on screen for a human.

Closing the console window still stops everything, supervisor included.
"""

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKOFF_S = (10, 30, 60)
MAX_CRASHES = 5
CRASH_WINDOW_S = 15 * 60


def supervise(command: list[str], *, backoff=BACKOFF_S, max_crashes=MAX_CRASHES,
              window=CRASH_WINDOW_S, sleep=time.sleep, clock=time.monotonic) -> int:
    crashes: list[float] = []
    while True:
        try:
            code = subprocess.call(command, cwd=HERE)
        except KeyboardInterrupt:
            print("\nStopped (Ctrl+C).")
            return 130
        if code == 0:
            print("Bot shut down on request -- not restarting.")
            return 0

        now = clock()
        crashes = [t for t in crashes if now - t < window] + [now]
        if len(crashes) >= max_crashes:
            print(f"\nThe bot has crashed {len(crashes)} times in {window // 60} minutes -- that's a "
                  f"crash loop, not bad luck, so I've stopped restarting it. The error is above; "
                  f"bot.log and bot_events.log have the details.")
            return code
        delay = backoff[min(len(crashes) - 1, len(backoff) - 1)]
        print(f"\nBot exited with code {code}. Restarting in {delay}s "
              f"({len(crashes)} crash(es) in the last {window // 60} min; giving up at {max_crashes}). "
              f"Close this window to stop.")
        try:
            sleep(delay)
        except KeyboardInterrupt:
            print("\nStopped (Ctrl+C).")
            return 130


if __name__ == "__main__":
    sys.exit(supervise([sys.executable, "-u", str(HERE / "main.py")]))
