from collections import defaultdict, deque

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-haiku-4-5"
MAX_TOKENS = 1024
HISTORY_TURNS = 20  # user+assistant messages kept per channel

SYSTEM_PROMPT = (
    "You are a games night assistant for a Dungeons & Dragons 5e table. "
    "You are a rules master, monster codex, and note taker extraordinaire. "
    "Give concise, table-ready answers: cite the specific rule or stat block "
    "when it matters, but don't lecture. When asked to improvise (an NPC line, "
    "a room description, a name), commit fully and stay in character. "
    "If a rules question is ambiguous or house-rule territory, say so briefly "
    "and offer the most common ruling rather than hedging at length. "
    "Keep replies short enough for a Discord message unless asked for detail."
)


class ClaudeChatCaller:
    """Thin wrapper around the Claude API with a per-channel conversation history."""

    def __init__(self):
        self.client = anthropic.Anthropic()
        self._history = defaultdict(lambda: deque(maxlen=HISTORY_TURNS))

    def call(self, message: str, channel_id: int) -> str:
        history = self._history[channel_id]
        history.append({"role": "user", "content": message})

        response = self.client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=list(history),
        )

        reply = "".join(block.text for block in response.content if block.type == "text")
        history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self, channel_id: int) -> None:
        """Clear conversation history for a channel (e.g. on a !forget command)."""
        self._history.pop(channel_id, None)
