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


RECAP_SYSTEM_PROMPT = (
    "You write session recaps for a Dungeons & Dragons 5e group, from raw "
    "voice transcripts of their table.\n\n"
    "The transcript is machine-generated and imperfect: proper nouns are "
    "often misheard, sentences get cut off, and in-character play is mixed "
    "with out-of-character table talk (rules arguments, dice, snacks, "
    "scheduling). Infer the intended proper nouns from context and use them "
    "consistently.\n\n"
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
    "or quotable lines. Quote people directly where the line is worth it. "
    "Skip this section if nothing stands out.\n\n"
    "Ignore out-of-character table talk unless it's genuinely funny and worth "
    "remembering. Keep the whole thing under 500 words -- brevity is part of "
    "the wit.\n\n"
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
    "locations, decisions, and unresolved threads. Don't write an intro or "
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
    "**Notable** -- a few bullets for memorable moments, NPCs, or loot. Skip "
    "if nothing stands out.\n\n"
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

    def call(self, message: str, channel_id: int) -> str:
        history = self._history[channel_id]
        history.append({"role": "user", "content": message})

        extra = {"output_config": {"effort": CHAT_EFFORT}} if self.chat_effort_ok else {}
        response = self.client.messages.create(
            model=self.chat_model,
            max_tokens=CHAT_MAX_TOKENS,
            system=SYSTEM_PROMPT,
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
