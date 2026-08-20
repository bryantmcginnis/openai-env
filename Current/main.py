import logging
import logging.handlers
import os
from pathlib import Path

from dotenv import load_dotenv

from discord_bot.dnd_bot import bot

load_dotenv()

# The log file lives next to this script, not in whatever directory the bot
# happened to be launched from -- double-clicking a desktop shortcut has a
# different working directory than running it from a terminal.
LOG_FILE = Path(__file__).resolve().parent / "bot.log"

# py-cord doesn't configure logging itself (unlike newer discord.py, it has
# no setup_logging() helper) -- without this, library messages fall back to
# logging's bare "handler of last resort" and our own log calls go nowhere.
#
# Writing bot.log from inside the app rather than by shell redirection means
# the log exists however the bot was started, and it stops PowerShell from
# wrapping every stderr line in a NativeCommandError blob.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        ),
    ],
)

# discord.voice.receive.reader logs a harmless "Received unexpected rtcp
# packet type=200" (a normal RTCP Sender Report) at INFO on every one --
# several per second during a recording session, drowning out anything
# else. Quieted now that the DAVE decrypt path is confirmed working; bump
# back to DEBUG here (or per-module) if diagnosing voice/opus again.
logging.getLogger("discord.voice.receive.reader").setLevel(logging.WARNING)

# faster_whisper logs two INFO lines per utterance ("Processing audio with
# duration...", "VAD filter removed..."), which is most of the log volume
# during a session and tells us nothing our own "transcribed utterance" line
# doesn't. Quieting it cuts the log growth rate by roughly two thirds. Set
# back to INFO if you're diagnosing VAD or segmentation.
logging.getLogger("faster_whisper").setLevel(logging.WARNING)

if __name__ == "__main__":
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        raise SystemExit(
            "DISCORD_BOT_TOKEN is not set. Copy .env.example to .env and fill it in."
        )

    bot.run(token)
