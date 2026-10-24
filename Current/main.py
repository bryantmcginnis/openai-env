import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from log_setup import configure_logging

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Logging goes up before the bot module is imported, so that what the import
# does -- resolving which Claude models to use -- is in the log too.
configure_logging(BASE_DIR)

from discord_bot import dnd_bot  # noqa: E402

# Exit codes are how run_bot.py tells a deliberate stop from a crash:
# 0 = someone ran !shutdown (don't restart); anything else = restart.
EXIT_UNEXPECTED_STOP = 3

if __name__ == "__main__":
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        raise SystemExit(
            "DISCORD_BOT_TOKEN is not set. Copy .env.example to .env and fill it in."
        )

    try:
        dnd_bot.bot.run(token)
    except Exception:
        # !shutdown can race py-cord's own reconnect on the way out and raise
        # "Session is closed". It was still a requested, clean shutdown.
        if dnd_bot.shutdown_requested.is_set():
            sys.exit(0)
        raise
    sys.exit(0 if dnd_bot.shutdown_requested.is_set() else EXIT_UNEXPECTED_STOP)
