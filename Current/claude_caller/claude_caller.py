from collections import defaultdict, deque

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-haiku-4-5"
MAX_TOKENS = 1024
HISTORY_TURNS = 20  # user+assistant messages kept per channel

# The recap runs on a stronger model than the chat does. It's once per
# session rather than once per message, and it's the artifact people
# actually read later -- a four-hour transcript comes to roughly 25k input
# tokens, so Opus costs cents per session. Chat stays on Haiku for latency:
# nobody wants to wait at the table for a rules answer.
RECAP_MODEL = "claude-opus-5"
# Thinking is on by default on Opus 5, and max_tokens caps thinking *plus*
# response text. At effort="max" on a messy three-hour transcript, reasoning
# alone can run past 16k -- and when it exhausts the budget the response
# comes back as a lone thinking block with *no text at all* and
# stop_reason="max_tokens". That is exactly how a recap silently vanished:
# empty text, no exception, nothing posted. The recap itself is only ~700
# tokens, so this ceiling is almost entirely headroom for thinking, and
# unused budget costs nothing.
SUMMARY_MAX_TOKENS = 32_000

# Transcripts are chunked before summarizing. This is far under the model's
# context window -- the limit that actually bites first is quality, not
# capacity: asking for one recap of a five-hour session gives a vague list,
# while summarizing in pieces and stitching those together keeps detail.
# ~4 chars/token puts a chunk around 6k tokens. Deliberately small: a long
# session gets summarized in more, finer passes, which keeps specifics
# (names, numbers, who said what) that one broad pass flattens away.
SUMMARY_CHUNK_CHARS = 25_000

# Reasoning effort for the recap. Cost per session is a few tens of cents
# either way, and this is a once-a-week artifact people actually read, so
# it's worth the ceiling. Drop to "high" to roughly halve the thinking spend.
RECAP_EFFORT = "max"

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

    # -- session recaps -----------------------------------------------------

    def _complete(self, system: str, prompt: str, max_tokens: int) -> str:
        """One-shot recap call with no conversation history attached.

        Runs on Opus with adaptive thinking. `fallbacks="default"` re-runs
        the request server-side on Anthropic's recommended substitute if
        safety classifiers decline it -- a D&D transcript is wall-to-wall
        described violence, so an occasional false positive is plausible and
        this turns it into a slightly different recap rather than a failure.
        """
        # Streamed because max_tokens is well above the ~16k point where a
        # non-streaming request risks an HTTP timeout, and these calls
        # genuinely run for minutes at effort="max".
        with self.client.beta.messages.stream(
            model=RECAP_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            thinking={"type": "adaptive"},
            output_config={"effort": RECAP_EFFORT},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
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

        first_pass = FOCUS_PROMPT if focus else RECAP_PROMPT

        if len(chunks) == 1:
            step("reading the session")
            draft = self._complete(
                system,
                first_pass.format(transcript=chunks[0], name=focus or ""),
                SUMMARY_MAX_TOKENS,
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
            draft = self._complete(system, combine, SUMMARY_MAX_TOKENS)

        # Polish: a second pass over the finished draft. Editing beats
        # first-drafting for voice, and the polish pass is given the source
        # material too so it can't drift away from what actually happened.
        step("polishing")
        return self._complete(
            system,
            POLISH_PROMPT.format(transcript=evidence, draft=draft),
            SUMMARY_MAX_TOKENS,
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
