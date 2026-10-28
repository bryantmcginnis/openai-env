import json
import logging
import os
import time
from collections import defaultdict, deque

import anthropic
import httpx
from dotenv import load_dotenv

load_dotenv()

_log = logging.getLogger(__name__)

# Network failures worth retrying. httpx.HTTPError is the one that actually
# bit us -- a RemoteProtocolError ("peer closed connection without sending
# complete message body") raised mid-stream when the machine changed
# network. It escapes the SDK unwrapped, so catching only anthropic.* misses
# it entirely.
RETRYABLE = (
    httpx.HTTPError,
    anthropic.APIConnectionError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)
RETRY_ATTEMPTS = 4
# Generous, because the usual cause is a human reconnecting to wifi.
RETRY_BACKOFF_S = (5, 20, 45)

# --- Model selection ---------------------------------------------------------
# Set in .env as CHAT_MODEL / RECAP_MODEL. Each takes either:
#   * a pinned model ID, e.g. "claude-opus-5-5" -- never changes underneath you;
#   * "latest-<family>", e.g. "latest-haiku" -- the newest model of that family
#     the API offers, looked up once at startup and logged.
# The API itself has no floating "latest" alias (a dateless ID like
# claude-haiku-5-5 is its own pinned snapshot), hence the lookup.
#
# Chat follows the newest Haiku by default: it's short, cheap, and has
# little to tune. The recap is pinned by default, because a new version can
# change behaviour under the same settings -- Opus 5.5 at the effort that
# suited Opus 5 burned its whole token budget and wrote nothing. Switch it to
# "latest-opus" only alongside a test recap.
#
# Chat stays on Haiku for latency (nobody wants to wait at the table for a
# rules answer); the recap is once per session and is the artifact people
# actually read, so it gets Opus.
CHAT_MODEL_SETTING = os.getenv("CHAT_MODEL") or "latest-haiku"
RECAP_MODEL_SETTING = os.getenv("RECAP_MODEL") or "claude-opus-5-5"

# Used when a "latest-" lookup can't reach the API at startup.
FAMILY_FALLBACK = {
    "haiku": "claude-haiku-5-5",
    "sonnet": "claude-sonnet-5-5",
    "opus": "claude-opus-5-5",
    "fable": "claude-fable-5-1",
}

# Chat replies are short; this is headroom so thinking on a hard rules
# question can't eat the answer, not a target. Thinking is on by default
# from Haiku 5.5 on, and "low" keeps it brief, which keeps replies fast.
CHAT_MAX_TOKENS = 4096
CHAT_EFFORT = "low"
HISTORY_TURNS = 20  # user+assistant messages kept per channel
# Thinking is always on, and max_tokens caps thinking *plus* response text.
# When reasoning exhausts the budget the response comes back as a lone
# thinking block with *no text at all* and stop_reason="max_tokens" -- an
# HTTP 200, not an exception (_attempt() turns it into one). The recap
# itself is only ~700 tokens, so this ceiling is almost entirely headroom
# for thinking, and unused budget costs nothing.
SUMMARY_MAX_TOKENS = 32_000

# Transcripts are chunked before summarizing. This is far under the model's
# context window -- the limit that actually bites first is quality, not
# capacity: asking for one recap of a five-hour session gives a vague list,
# while summarizing in pieces and stitching those together keeps detail.
# ~4 chars/token puts a chunk around 6k tokens. Deliberately small: a long
# session gets summarized in more, finer passes, which keeps specifics
# (names, numbers, who said what) that one broad pass flattens away.
SUMMARY_CHUNK_CHARS = 25_000

# Reasoning effort for the recap. Do NOT raise this to "max" on Opus 5.5:
# measured on a 326-word sample, "max" spent the entire 32,000-token budget
# on reasoning and wrote no recap in 3 of 3 runs (~$0.64 per pass), where
# Opus 5 used ~1,600 tokens for the same pass. "high" finished every test
# inside budget -- a real 20k-character session in 54s for $0.19, and a
# 5-pass 13k-word session in 145s for $0.47 -- with the sharpest writing of
# the levels tried. "medium" is ~40% cheaper but noticeably flatter.
RECAP_EFFORT = "high"

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


SESSION_QA_PROMPT = (
    "You also have this table's session records below: campaign notes, the "
    "open plot threads, recaps of recent sessions, and (if a session is on or "
    "just finished) tonight's transcript with local times.\n\n"
    "Work out which kind of question you've been asked. If it's about what "
    "happened in their game -- what someone said, found, decided or promised, "
    "who an NPC is, where the party stands -- answer from the records: be "
    "specific, quote the line with its time like [20:41] where you can, and if "
    "the records don't cover it, say you didn't catch that rather than "
    "guessing. Older sessions only exist as recaps, so say when an answer comes "
    "from a recap and might be missing detail. If it's a rules or general D&D "
    "question, answer it normally from your own knowledge and ignore the "
    "records.\n\n"
    "The transcript is machine-generated: names get misheard (the notes list "
    "the usual mishearings) and in-character play is mixed with table talk."
)

MEMORY_SYSTEM_PROMPT = (
    "You keep two things for a D&D group: their list of open plot threads, and "
    "a quote board. You'll get the current open threads, tonight's recap, and "
    "tonight's raw transcript. The transcript is machine-generated, so names "
    "may be misheard -- use the campaign notes to read them correctly."
)

MEMORY_PROMPT = (
    "Update the records for tonight.\n\n"
    "new_threads: hooks, mysteries, quests, debts and promises that are still "
    "unresolved after tonight and are NOT already on the open list (if one is "
    "already there, leave it; don't restate it). Only things that will matter "
    "next session -- usually 1 to 5. title: at most 8 words. detail: one short "
    "line of context, at most 25 words.\n\n"
    "resolved: open threads (by id) that tonight's transcript clearly settles. "
    "If it's merely mentioned or progressed, leave it open. how: one factual "
    "sentence saying how it was resolved. quip: one short dry joke about the "
    "resolution, at most 15 words, affectionate -- never at a player's "
    "expense.\n\n"
    "quotes: 3 to 8 of the night's most memorable lines. Copy each one WORD FOR "
    "WORD from the transcript -- they are checked against it, and anything "
    "reworded is thrown away. speaker: the name exactly as it appears before "
    "the colon. Prefer in-character lines and genuinely funny table talk. "
    "context: the one-line setup that makes the quote land for someone "
    "reading the board weeks later: the one thing that prompted the line, "
    "like a stage direction. One sentence, 6 to 15 words, no quotation marks, "
    "and don't explain or repeat the joke. In-game moments use character "
    "names and \"the DM\"; out-of-character ones use players' real names -- "
    "never both for one person. e.g. *The boss rises again, eight feet tall, "
    "with a mouth in its chest.* or *Joe mentions he leaves for work at 6:45.*\n\n"

    "<open_threads>\n{threads}\n</open_threads>\n\n"
    "<recap>\n{recap}\n</recap>\n\n"
    "<transcript>\n{transcript}\n</transcript>"
)

MEMORY_SCHEMA = {
    "type": "object",
    "properties": {
        "new_threads": {"type": "array", "items": {
            "type": "object",
            "properties": {"title": {"type": "string"}, "detail": {"type": "string"}},
            "required": ["title", "detail"], "additionalProperties": False}},
        "resolved": {"type": "array", "items": {
            "type": "object",
            "properties": {"id": {"type": "integer"}, "how": {"type": "string"}, "quip": {"type": "string"}},
            "required": ["id", "how", "quip"], "additionalProperties": False}},
        "quotes": {"type": "array", "items": {
            "type": "object",
            "properties": {"speaker": {"type": "string"}, "quote": {"type": "string"},
                           "context": {"type": "string"}},
            "required": ["speaker", "quote", "context"], "additionalProperties": False}},
    },
    "required": ["new_threads", "resolved", "quotes"],
    "additionalProperties": False,
}

RELEASE_SYSTEM_PROMPT = (
    "You write the release notes for a D&D group's Discord bot, posted in "
    "their #dndbot-release-notes channel. The readers are the players: smart, "
    "into D&D, mostly not programmers. Make it nerdy and fun -- D&D flavour, a "
    "light joke or two -- but every line must make sense to someone who has "
    "never written code. Lead with what they can now DO, and show the exact "
    "command to type. Translate technical fixes into what the table "
    "experienced (\"the bot stopped going deaf mid-session\"), and put any "
    "genuinely interesting technical detail in a short *Nerd corner* line, "
    "explained in plain words. Describe only what the commits and diff "
    "actually show -- never invent features, numbers or plans. Leave out file "
    "names, function names, internal refactors nobody would notice, and "
    "anything about API keys or secrets."
)

RELEASE_PROMPT = (
    "Write the post for these changes, in Discord markdown.\n\n"
    "Start with one line: `# ` + an emoji + a short, playful title for this "
    "update. Then whichever of these sections have something in them, as "
    "`## ` headings with bullets:\n"
    "- ✨ New -- features, each with how to use it\n"
    "- 🔧 Improved -- existing things that got better\n"
    "- 🐛 Fixed -- bugs squashed, told as what went wrong at the table\n"
    "Tiny changes (one more phantom phrase on the ignore list) get one short "
    "bullet at most. Keep the whole thing under 2500 characters -- shorter is "
    "better for a small update. No sign-off.\n\n"
    "Earlier posts, for voice and so you don't re-announce old features:\n"
    "<earlier_posts>\n{earlier}\n</earlier_posts>\n\n"
    "<commits>\n{log}\n</commits>\n\n"
    "<diff>\n{diff}\n</diff>"
)

QUOTE_CONTEXT_PROMPT = (
    "Someone at a D&D table put this line on the group's quote board:\n\n"
    "{speaker}: \"{quote}\"\n\n"
    "Here is the transcript around it; the quoted line is marked >>>. It's "
    "machine-generated, so names may be "
    "misheard -- the campaign notes list the usual mishearings.\n\n"
    "<transcript>\n{surrounding}\n</transcript>\n\n"
    "Write the one-line setup that makes the quote land for someone "
    "reading the board weeks later: the one thing that prompted the line, "
    "like a stage direction. One sentence, 6 to 15 words, no quotation marks, "
    "and don't explain or repeat the joke. In-game moments use character "
    "names and \"the DM\"; out-of-character ones use players' real names -- "
    "never both for one person. e.g. *The boss rises again, eight feet tall, "
    "with a mouth in its chest.* or *Joe mentions he leaves for work at 6:45.*\n\n"
)

QUOTE_CONTEXT_SCHEMA = {
    "type": "object",
    "properties": {"context": {"type": "string"}},
    "required": ["context"],
    "additionalProperties": False,
}

CORRECT_SYSTEM_PROMPT = (
    "You correct a D&D session recap that the players have found mistakes in. "
    "They were at the table: where a correction disagrees with the transcript, "
    "the correction wins (the machine transcript mislabels speakers and "
    "mishears names). Make every change the corrections ask for, plus anything "
    "that follows from them -- use the transcript for that context. Leave "
    "everything else exactly as it was: same structure, same voice, same "
    "jokes. Call people by their character names and pronouns, and the "
    "Dungeon Master \"the DM\" -- real names only in **IRL:** bullets."
)

CORRECT_PROMPT = (
    "Apply the table's corrections.\n\n"
    "recap: the whole corrected recap, in the same format.\n"
    "changes: one short plain line per change you made.\n"
    "thread_updates: open threads (by id) that the corrections make wrong, "
    "with a corrected title and detail. Empty if none.\n"
    "quote_fixes: quotes (by index) whose speaker the corrections change, with "
    "the corrected label in the transcript's form, e.g. \"DM (Joe)\". Empty if none.\n\n"
    "<corrections>\n{corrections}\n</corrections>\n\n"
    "<recap>\n{recap}\n</recap>\n\n"
    "<open_threads>\n{threads}\n</open_threads>\n\n"
    "<quotes>\n{quotes}\n</quotes>\n\n"
    "<transcript>\n{transcript}\n</transcript>"
)

CORRECT_SCHEMA = {
    "type": "object",
    "properties": {
        "recap": {"type": "string"},
        "changes": {"type": "array", "items": {"type": "string"}},
        "thread_updates": {"type": "array", "items": {
            "type": "object",
            "properties": {"id": {"type": "integer"}, "title": {"type": "string"}, "detail": {"type": "string"}},
            "required": ["id", "title", "detail"], "additionalProperties": False}},
        "quote_fixes": {"type": "array", "items": {
            "type": "object",
            "properties": {"index": {"type": "integer"}, "speaker": {"type": "string"}},
            "required": ["index", "speaker"], "additionalProperties": False}},
    },
    "required": ["recap", "changes", "thread_updates", "quote_fixes"],
    "additionalProperties": False,
}

RECAP_SYSTEM_PROMPT = (
    "You write session recaps for a Dungeons & Dragons 5e group, from raw "
    "voice transcripts of their table.\n\n"
    "The transcript is machine-generated and imperfect: proper nouns are "
    "often misheard, sentences get cut off, and in-character play is mixed "
    "with out-of-character table talk (rules arguments, dice, snacks, "
    "scheduling). Infer the intended proper nouns from context and use them "
    "consistently.\n\n"
    "Each line is labelled with the speaker's character and, in brackets, the "
    "player -- \"Thia (Goose): ...\". \"DM\" is the Dungeon Master. In the story, "
    "always call people by their character's name and use the character's "
    "pronouns from the campaign notes, and call the DM \"the DM\". Never use a player's name for their "
    "character, and never infer a character's gender from the player's. "
    "Players' real names belong only in the IRL bullets.\n\n"
    "Write with wit. The register is a sharp friend who was at the table and "
    "is retelling it well -- dry, affectionate, occasionally deadpan about "
    "the party's worse decisions. The comedy comes from what actually "
    "happened: the failed roll at the worst moment, the plan nobody thought "
    "through, the NPC who saw straight through them. Land it in the framing "
    "and the timing, not in jokes bolted on afterwards. Never be snide about "
    "the players themselves, and never let a laugh cost the reader a fact "
    "they need.\n\n"
    "Accuracy outranks everything, humour included. Report only what actually "
    "happened -- never invent events, embellish a roll, or promote a "
    "throwaway line into a scene. If something is too garbled to follow, "
    "leave it out rather than guessing at it."
)

RECAP_PROMPT = (
    "Here is the transcript of tonight's session. Write a recap for the "
    "players to read before next time.\n\n"
    "Format it as:\n"
    "**Previously on...** -- two or three paragraphs of narrative recap, "
    "written like the cold open of a TV episode. Focus on what the party "
    "did, decided, and discovered.\n"
    "**Loose threads** -- a short bullet list of unresolved hooks, unanswered "
    "questions, and things the party said they'd do next.\n"
    "**Notable** -- a few bullets for memorable rolls, kills, loot, NPCs met, "
    "or quotable lines. Quote people directly where the line is worth it. If "
    "something genuinely funny or interesting happened out of character, add "
    "one or two bullets that start **IRL:**, using the players' real names. "
    "Skip this section if nothing stands out.\n\n"
    "Keep out-of-character table talk out of the narrative -- the best of it "
    "goes in the IRL bullets. Keep the whole thing under 500 words -- brevity "
    "is part of the wit.\n\n"
    "<transcript>\n{transcript}\n</transcript>"
)

# Second pass. The draft is usually accurate but flat in places -- a
# dedicated editing pass sharpens voice far more reliably than asking for
# accuracy and comedy in a single shot, because the model can read its own
# finished draft instead of writing blind.
POLISH_PROMPT = (
    "Below is a draft recap of a D&D session, alongside the transcript it "
    "was written from. Edit it.\n\n"
    "Do:\n"
    "- Sharpen flat sentences and cut anything that reads like filler.\n"
    "- Improve comic timing: put the funny detail at the end of its "
    "sentence, and trust the reader to get it without an explanation.\n"
    "- Make sure the opening line actually hooks, and the last line lands.\n"
    "- Keep every concrete fact, name, and number exactly as it is.\n\n"
    "Don't:\n"
    "- Add events, jokes, or details that aren't in the transcript.\n"
    "- Make it longer. Shorter is better if the cut costs nothing.\n"
    "- Change the section structure.\n\n"
    "Return only the edited recap, with no commentary about your changes.\n\n"
    "<transcript>\n{transcript}\n</transcript>\n\n"
    "<draft>\n{draft}\n</draft>"
)

PARTIAL_PROMPT = (
    "This is part {part} of {total} of one session's transcript. Summarize "
    "just this part in detail: what happened, in order, plus any NPCs, "
    "locations, decisions, and unresolved threads. Note any genuinely funny or "
    "interesting out-of-character moments separately, marked IRL, with the "
    "player's real name. Don't write an intro or "
    "conclusion -- this will be combined with the other parts.\n\n"
    "<transcript>\n{transcript}\n</transcript>"
)

COMBINE_PROMPT = (
    "Below are ordered summaries of consecutive parts of a single D&D "
    "session. Combine them into one recap for the players.\n\n"
    "Format it as:\n"
    "**Previously on...** -- two or three paragraphs of narrative recap, "
    "written like the cold open of a TV episode.\n"
    "**Loose threads** -- a short bullet list of unresolved hooks and things "
    "the party said they'd do next.\n"
    "**Notable** -- a few bullets for memorable moments, NPCs, or loot, plus "
    "one or two **IRL:** bullets for the best out-of-character moments, using "
    "the players' real names. Skip if nothing stands out.\n\n"
    "Give the whole session even weight. The natural failure here is to "
    "write the opening in vivid detail and then compress everything after "
    "it into a summary of a summary -- the last part of the night should be "
    "as specific, and have as many real names and numbers in it, as the "
    "first. Where the session ends is also what the players most need to "
    "remember, so finish on where things actually stand rather than "
    "trailing off.\n\n"
    "Keep the whole thing under 500 words.\n\n"
    "{partials}"
)

# --- Styles -----------------------------------------------------------------
# Appended to the recap system prompt. Kept as directives about *register*
# rather than format, so they compose with the section structure above
# instead of fighting it.
RECAP_STYLES: dict[str, str] = {
    "funny": (
        "Push the comedy harder than usual. Lean into the party's worst "
        "decisions and the timing of bad rolls. Still never invent a joke "
        "that isn't in what happened."
    ),
    "sarcastic": (
        "Adopt a dry, deadpan, faintly exasperated narrator -- the tone of "
        "someone who warned them. Cutting about decisions, never about the "
        "people who made them."
    ),
    "short": (
        "Be ruthless: 150 words maximum for the whole recap. Cut the "
        "narrative to its spine. Every sentence must carry a fact."
    ),
    "long": (
        "Go long: up to 1200 words. Room for scene detail, dialogue, and "
        "the reasoning behind decisions. Do not pad -- earn the length with "
        "specifics from the transcript."
    ),
    "factual": (
        "No jokes, no narrative flourish. A clear chronological account of "
        "what happened, who did it, and what resulted. Plain and useful."
    ),
    "epic": (
        "High-fantasy register, like a chronicle or a saga. Grand without "
        "being purple, and never at the cost of clarity about what happened."
    ),
    "noir": (
        "Hard-boiled detective register: short sentences, world-weary, "
        "atmospheric. The party are people things happen to."
    ),
}

STYLE_ALIASES = {
    "sarcasm": "sarcastic", "sarcy": "sarcastic", "dry": "sarcastic",
    "funnier": "funny", "comedy": "funny", "comedic": "funny",
    "brief": "short", "quick": "short", "tldr": "short",
    "detailed": "long", "full": "long", "verbose": "long",
    "dry-facts": "factual", "facts": "factual", "plain": "factual",
    "dramatic": "epic", "grand": "epic", "saga": "epic",
}


def resolve_style(word: str) -> str | None:
    """Map a user-typed word to a style key, or None if it isn't one."""
    key = word.lower().strip(" ,.")
    key = STYLE_ALIASES.get(key, key)
    return key if key in RECAP_STYLES else None


# --- Per-character recaps ---------------------------------------------------
FOCUS_PROMPT = (
    "Write a personal recap for the player of **{name}**, to be sent to them "
    "directly. This is not the party recap -- it is their character's night, "
    "told to them.\n\n"
    "Format it as:\n"
    "**Your night** -- a narrative account of what {name} did, chose, said "
    "and discovered, in order. Write in second person (\"you\"). Give their "
    "moments the spotlight, including the ones that went badly.\n"
    "**What you know** -- things {name} specifically learned, saw, or was "
    "told, especially anything the rest of the party may not know.\n"
    "**Left hanging** -- what's unresolved for {name} personally: promises "
    "made, debts, questions asked of them, things they said they'd do.\n\n"
    "Where the transcript shows {name} was absent or quiet for a stretch, "
    "say so briefly rather than inventing involvement. If the transcript "
    "doesn't clearly show a moment belonging to {name}, leave it out. "
    "Up to 700 words.\n\n"
    "<transcript>\n{transcript}\n</transcript>"
)


def _supported(capabilities: dict, *path: str) -> bool:
    """Read one capability flag; unknown means "assume yes".

    Unknown only happens when the model list couldn't be fetched, and then
    the model is one of the FAMILY_FALLBACK defaults, all of which support
    effort and adaptive thinking.
    """
    node = capabilities
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return True
        node = node[key]
    return bool(node.get("supported", True)) if isinstance(node, dict) else bool(node)


def resolve_model(setting: str, available: list) -> tuple[str, dict]:
    """Turn a CHAT_MODEL / RECAP_MODEL setting into (model_id, capabilities)."""
    setting = setting.strip()
    by_id = {m.id: m for m in available}
    if setting.startswith("latest-"):
        family = setting[len("latest-"):].lower()
        candidates = [m for m in available if m.id.startswith(f"claude-{family}-")]
        if candidates:
            chosen = max(candidates, key=lambda m: m.created_at).id
        else:
            chosen = FAMILY_FALLBACK.get(family)
            if chosen is None:
                raise ValueError(f"unknown model family in {setting!r}; try latest-haiku or latest-opus")
            _log.warning("could not look up %s; falling back to %s", setting, chosen)
    else:
        chosen = setting
        if available and chosen not in by_id:
            _log.warning("model %r is not in this API key's model list -- calls will likely fail", chosen)
    caps = (by_id[chosen].model_dump().get("capabilities") or {}) if chosen in by_id else {}
    return chosen, caps


class ClaudeChatCaller:
    """Thin wrapper around the Claude API with a per-channel conversation history."""

    def __init__(self):
        self.client = anthropic.Anthropic()
        self._history = defaultdict(lambda: deque(maxlen=HISTORY_TURNS))

        try:
            available = list(self.client.with_options(timeout=15, max_retries=1).models.list(limit=100))
        except Exception as exc:
            _log.warning("could not list models (%s); using fallbacks for any latest- settings", exc)
            available = []
        self.chat_model, chat_caps = resolve_model(CHAT_MODEL_SETTING, available)
        self.recap_model, recap_caps = resolve_model(RECAP_MODEL_SETTING, available)

        # Gate each parameter on what the chosen model accepts, so a pinned
        # older model (Haiku 4.5 rejects effort and adaptive thinking) still
        # works instead of returning a 400 on every call.
        self.chat_effort_ok = _supported(chat_caps, "effort")
        self.recap_effort_ok = _supported(recap_caps, "effort")
        self.recap_adaptive_ok = _supported(recap_caps, "thinking", "types", "adaptive")
        _log.info(
            "chat model: %s (from %r) | recap model: %s (from %r)",
            self.chat_model, CHAT_MODEL_SETTING, self.recap_model, RECAP_MODEL_SETTING,
        )

    def call(self, message: str, channel_id: int, context: tuple[str, str] | None = None) -> str:
        """Answer a chat message. `context` is (stable_records, transcript) when a
        session is on or recent, so questions about the game can be answered
        from what was actually said."""
        history = self._history[channel_id]
        history.append({"role": "user", "content": message})

        system: str | list = SYSTEM_PROMPT
        if context is not None:
            stable, transcript = context
            # Stable records first and cached: follow-up questions in the next
            # few minutes reread them at a fraction of the price. The
            # transcript grows between questions, so it goes last.
            system = [{"type": "text", "text": f"{SYSTEM_PROMPT}\n\n{SESSION_QA_PROMPT}\n\n{stable}",
                       "cache_control": {"type": "ephemeral"}}]
            if transcript:
                system.append({"type": "text", "text": f"<tonights_transcript>\n{transcript}\n</tonights_transcript>"})

        extra = {"output_config": {"effort": CHAT_EFFORT}} if self.chat_effort_ok else {}
        response = self.client.messages.create(
            model=self.chat_model,
            max_tokens=CHAT_MAX_TOKENS,
            system=system,
            messages=list(history),
            **extra,
        )

        reply = "".join(block.text for block in response.content if block.type == "text").strip()
        if not reply:
            # Thinking used the budget, or a refusal. Say so rather than
            # sending Discord an empty message (which it rejects outright).
            return f"(No answer came back -- stop_reason={response.stop_reason}. Try rephrasing?)"
        history.append({"role": "assistant", "content": reply})
        return reply

    def update_memory(self, transcript: str, recap: str, open_threads: list[dict], notes: str = "") -> dict:
        """One structured call after a recap: new threads, resolved ones, quotes.

        Structured output guarantees the reply parses, so a bad reply can't
        corrupt the threads list -- worst case, nothing changes.
        """
        threads = json.dumps([{"id": t["id"], "title": t["title"], "detail": t["detail"]}
                              for t in open_threads], ensure_ascii=False) or "[]"
        system = MEMORY_SYSTEM_PROMPT + (f"\n\n{notes}" if notes else "")
        config = {"format": {"type": "json_schema", "schema": MEMORY_SCHEMA}}
        if self.chat_effort_ok:
            config["effort"] = "medium"  # judging "is this resolved?" earns a little thought
        response = self.client.messages.create(
            model=self.chat_model,
            max_tokens=8000,
            system=system,
            messages=[{"role": "user", "content": MEMORY_PROMPT.format(
                threads=threads, recap=recap or "(no recap)", transcript=transcript)}],
            output_config=config,
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        if response.stop_reason != "end_turn" or not text:
            raise RuntimeError(f"memory update returned no result (stop_reason={response.stop_reason})")
        return json.loads(text)

    def write_release_notes(self, log: str, diff: str, earlier: str = "") -> str:
        """Turn commits + diff into a post for the table. On the recap model
        (it has to read code and explain it plainly); a release is rare, so
        the cost -- typically $0.10-0.50 -- doesn't matter."""
        prompt = RELEASE_PROMPT.format(earlier=earlier or "(none yet)", log=log, diff=diff)
        return self._complete(RELEASE_SYSTEM_PROMPT, prompt, SUMMARY_MAX_TOKENS)

    def quote_context(self, speaker: str, quote: str, surrounding: str, notes: str = "") -> str:
        """One line of scene-setting for a quote, from the lines around it.
        A couple of seconds on the chat model, about $0.002."""
        config = {"format": {"type": "json_schema", "schema": QUOTE_CONTEXT_SCHEMA}}
        if self.chat_effort_ok:
            config["effort"] = "medium"  # at low it grabbed the wrong moment from noisy transcripts
        response = self.client.messages.create(
            model=self.chat_model,
            max_tokens=4000,
            system=notes or SYSTEM_PROMPT,
            messages=[{"role": "user", "content": QUOTE_CONTEXT_PROMPT.format(
                speaker=speaker, quote=quote, surrounding=surrounding or "(nothing nearby)")}],
            output_config=config,
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        if response.stop_reason != "end_turn" or not text:
            raise RuntimeError(f"no quote context came back (stop_reason={response.stop_reason})")
        return json.loads(text)["context"].strip().strip("*_").strip()

    def correct_recap(self, recap: str, corrections: str, transcript: str,
                      open_threads: list[dict], quotes: list[dict], notes: str = "") -> dict:
        """Rewrite a posted recap to the table's corrections, and say what else
        they touch. On the recap model: this is the artifact people read."""
        prompt = CORRECT_PROMPT.format(
            corrections=corrections, recap=recap,
            threads=json.dumps([{"id": t["id"], "title": t["title"], "detail": t["detail"]}
                                for t in open_threads], ensure_ascii=False),
            quotes=json.dumps([{"index": q["index"], "speaker": q["speaker"], "text": q["text"]}
                               for q in quotes], ensure_ascii=False),
            transcript=transcript,
        )
        config = {"format": {"type": "json_schema", "schema": CORRECT_SCHEMA}}
        if self.recap_effort_ok:
            config["effort"] = "high"
        extra = {"thinking": {"type": "adaptive"}} if self.recap_adaptive_ok else {}
        last_error = None
        for attempt in range(RETRY_ATTEMPTS):
            if attempt:
                time.sleep(RETRY_BACKOFF_S[min(attempt - 1, len(RETRY_BACKOFF_S) - 1)])
            try:
                with self.client.beta.messages.stream(
                    model=self.recap_model,
                    max_tokens=SUMMARY_MAX_TOKENS,
                    system=CORRECT_SYSTEM_PROMPT + (f"\n\n{notes}" if notes else ""),
                    messages=[{"role": "user", "content": prompt}],
                    output_config=config,
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                    **extra,
                ) as stream:
                    response = stream.get_final_message()
                break
            except RETRYABLE as exc:
                last_error = exc
        else:
            raise RuntimeError(f"gave up after {RETRY_ATTEMPTS} attempts: {last_error}") from last_error
        text = "".join(b.text for b in response.content if b.type == "text")
        if response.stop_reason != "end_turn" or not text:
            raise RuntimeError(f"the correction came back empty (stop_reason={response.stop_reason})")
        result = json.loads(text)
        if not result["recap"].strip():
            raise RuntimeError("the correction returned an empty recap")
        return result

    def reset(self, channel_id: int) -> None:
        """Clear conversation history for a channel (e.g. on a !forget command)."""
        self._history.pop(channel_id, None)

    # -- session recaps -----------------------------------------------------

    def _complete(self, system: str, prompt: str, max_tokens: int, notify=None) -> str:
        """One-shot recap call with no conversation history attached.

        Runs on Opus with adaptive thinking. `fallbacks="default"` re-runs
        the request server-side on Anthropic's recommended substitute if
        safety classifiers decline it -- a D&D transcript is wall-to-wall
        described violence, so an occasional false positive is plausible and
        this turns it into a slightly different recap rather than a failure.

        Retries transient network failures itself. The SDK's own retries
        cover a request that fails to *start*, but not one that dies partway
        through reading the response body -- and at effort="max" the body
        takes minutes to arrive, which is a wide window for a laptop to
        change wifi. A dropped connection three passes in would otherwise
        throw away every completed pass.
        """
        last_error: Exception | None = None
        for attempt in range(RETRY_ATTEMPTS):
            if attempt:
                delay = RETRY_BACKOFF_S[min(attempt - 1, len(RETRY_BACKOFF_S) - 1)]
                _log.warning(
                    "recap pass failed (%s: %s); retry %d/%d in %ds",
                    type(last_error).__name__, last_error, attempt, RETRY_ATTEMPTS - 1, delay,
                )
                if notify is not None:
                    notify(f"connection lost, retrying in {delay}s")
                time.sleep(delay)
            try:
                return self._attempt(system, prompt, max_tokens)
            except RETRYABLE as exc:
                last_error = exc
        raise RuntimeError(
            f"gave up after {RETRY_ATTEMPTS} attempts -- last error: "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error

    def _attempt(self, system: str, prompt: str, max_tokens: int) -> str:
        # Streamed because max_tokens is well above the ~16k point where a
        # non-streaming request risks an HTTP timeout, and these calls
        # genuinely run for minutes at effort="max".
        extra = {}
        if self.recap_adaptive_ok:
            extra["thinking"] = {"type": "adaptive"}
        if self.recap_effort_ok:
            extra["output_config"] = {"effort": RECAP_EFFORT}
        with self.client.beta.messages.stream(
            model=self.recap_model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            **extra,
        ) as stream:
            response = stream.get_final_message()

        # Check stop_reason before touching content. Both of these come back
        # as an HTTP 200 with no text block, so without these checks they
        # surface as an empty string and the caller posts nothing at all.
        if response.stop_reason == "refusal":
            raise RuntimeError(
                "the model declined to summarize this transcript "
                "(safety classifier); try !recap over a shorter window"
            )
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        if response.stop_reason == "max_tokens" and not text:
            raise RuntimeError(
                f"the model used its entire {max_tokens}-token budget on reasoning "
                f"and produced no recap -- raise SUMMARY_MAX_TOKENS, or drop "
                f"RECAP_EFFORT below {RECAP_EFFORT!r}"
            )
        if not text:
            raise RuntimeError(
                f"the model returned no text (stop_reason={response.stop_reason!r})"
            )
        return text

    def plan_passes(self, transcript: str) -> int:
        """How many model calls summarize_session will make. Lets the caller
        show honest progress instead of an eight-minute silent spinner."""
        chunks = len(_chunk_transcript(transcript, SUMMARY_CHUNK_CHARS))
        return chunks + 1 if chunks == 1 else chunks + 2  # (+polish) or (+reduce +polish)

    def summarize_session(
        self,
        transcript: str,
        on_progress=None,
        campaign_notes: str = "",
        style: str | None = None,
        focus: str | None = None,
    ) -> str:
        """Turn a raw session transcript into a "Previously on..." recap.

        The transcript is a live voice transcription, so it arrives messy:
        mangled proper nouns, interrupted sentences, table talk about pizza
        mixed in with the actual scene. The prompts below lean on that.
        """
        chunks = _chunk_transcript(transcript, SUMMARY_CHUNK_CHARS)
        total = self.plan_passes(transcript)
        done = 0

        # Appended to the system prompt rather than the user turn so they
        # apply to every pass, including the polish -- a style set only on
        # the first pass gets quietly edited back out by the polish.
        system = RECAP_SYSTEM_PROMPT
        if campaign_notes:
            system += f"\n\n{campaign_notes}"
        if style and style in RECAP_STYLES:
            system += f"\n\nSTYLE FOR THIS RECAP: {RECAP_STYLES[style]}"
        if focus:
            system += (
                f"\n\nThis recap is for the player of {focus} personally, not "
                f"the whole party. Their perspective is the through-line."
            )

        def step(label: str) -> None:
            nonlocal done
            done += 1
            if on_progress is not None:
                on_progress(done, total, label)

        def notify(label: str) -> None:
            """Surface a retry in Discord rather than looking frozen."""
            if on_progress is not None:
                on_progress(done, total, label)

        first_pass = FOCUS_PROMPT if focus else RECAP_PROMPT

        if len(chunks) == 1:
            step("reading the session")
            draft = self._complete(
                system,
                first_pass.format(transcript=chunks[0], name=focus or ""),
                SUMMARY_MAX_TOKENS,
                notify,
            )
            evidence = chunks[0]
        else:
            # Map: summarize each slice on its own, keeping more detail than
            # the final recap needs, since the reduce step below only sees
            # these rather than the transcript.
            partials = []
            for index, chunk in enumerate(chunks):
                step(f"reading part {index + 1} of {len(chunks)}")
                partials.append(
                    self._complete(
                        system,
                        PARTIAL_PROMPT.format(part=index + 1, total=len(chunks), transcript=chunk),
                        SUMMARY_MAX_TOKENS,
                        notify,
                    )
                )

            # Reduce: stitch the parts into one narrative.
            evidence = "\n\n".join(f"[Part {i + 1}]\n{p}" for i, p in enumerate(partials))
            step("writing the recap")
            combine = (
                FOCUS_PROMPT.format(transcript=evidence, name=focus)
                if focus
                else COMBINE_PROMPT.format(partials=evidence)
            )
            draft = self._complete(system, combine, SUMMARY_MAX_TOKENS, notify)

        # Polish: a second pass over the finished draft. Editing beats
        # first-drafting for voice, and the polish pass is given the source
        # material too so it can't drift away from what actually happened.
        step("polishing")
        return self._complete(
            system,
            POLISH_PROMPT.format(transcript=evidence, draft=draft),
            SUMMARY_MAX_TOKENS,
            notify,
        )


def _chunk_transcript(transcript: str, chunk_chars: int) -> list[str]:
    """Split on line boundaries so an utterance is never cut in half."""
    lines = transcript.splitlines()
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in lines:
        if size + len(line) > chunk_chars and current:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks or [""]
