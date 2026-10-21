import os
from pathlib import Path

from dotenv import load_dotenv

from log_setup import configure_logging

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Logging goes up before the bot module is imported, so that what the import
# does -- resolving which Claude models to use -- is in the log too.
configure_logging(BASE_DIR)

from discord_bot.dnd_bot import bot  # noqa: E402

if __name__ == "__main__":
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        raise SystemExit(
            "DISCORD_BOT_TOKEN is not set. Copy .env.example to .env and fill it in."
        )

    bot.run(token)
