import logging
import os

from dotenv import load_dotenv

from discord_bot.dnd_bot import bot

load_dotenv()

# py-cord doesn't configure logging itself (unlike newer discord.py, it has
# no setup_logging() helper) -- without this, library messages fall back to
# logging's bare "handler of last resort" and our own log calls go nowhere.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# discord.voice.receive.reader logs a harmless "Received unexpected rtcp
# packet type=200" (a normal RTCP Sender Report) at INFO on every one --
# several per second during a recording session, drowning out anything
# else. Quieted now that the DAVE decrypt path is confirmed working; bump
# back to DEBUG here (or per-module) if diagnosing voice/opus again.
logging.getLogger("discord.voice.receive.reader").setLevel(logging.WARNING)

if __name__ == "__main__":
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        raise SystemExit(
            "DISCORD_BOT_TOKEN is not set. Copy .env.example to .env and fill it in."
        )

    bot.run(token)
