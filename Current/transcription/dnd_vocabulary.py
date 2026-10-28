import logging
import os
import re
from pathlib import Path

_log = logging.getLogger(__name__)

# Whisper's initial_prompt is hard-capped at half the 448-token decoder
# context, minus one: 223 tokens. faster-whisper doesn't error when you
# exceed it -- it silently keeps only the *last* 223 tokens and drops the
# rest off the front. Measured on this glossary, a token averages ~3.25
# characters, so this is the usable character budget.
#
# This matters more than it looks: the full core list below plus a campaign
# glossary runs well past the cap, so terms have to be prioritised
# deliberately rather than piled up and quietly truncated.
PROMPT_TOKEN_BUDGET = 223
_CHARS_PER_TOKEN = 3.25
MAX_PROMPT_CHARS = int(PROMPT_TOKEN_BUDGET * _CHARS_PER_TOKEN)

_PREFIX = "Dungeons and Dragons 5th edition game session. "

# Where the campaign glossary lives, if it isn't set via CAMPAIGN_VOCAB.
DEFAULT_VOCAB_FILE = Path(__file__).resolve().parent.parent / "campaign_vocab.txt"

# Free-form campaign briefing for the *recap* model. Deliberately a separate
# file from the glossary above, because the two feed different things: the
# glossary is a token-starved hint for a local speech model, while this is
# prose for Claude, which has room for a page of it and can actually reason
# over "the DM is Peter" or "wildgoose plays Llyr".
DEFAULT_CONTEXT_FILE = Path(__file__).resolve().parent.parent / "campaign_context.md"

# Terms that generic speech models routinely mangle because they never show
# up in everyday training data. Feeding these to Whisper as an
# `initial_prompt` biases the decoder toward this vocabulary/spelling when
# it hears something ambiguous -- it does not need to match what's spoken
# verbatim, it just needs to nudge word choice in the right direction.
_CORE_TERMS = [
    # Ability scores & mechanics
    "Strength", "Dexterity", "Constitution", "Intelligence", "Wisdom", "Charisma",
    "armor class", "AC", "saving throw", "advantage", "disadvantage",
    "proficiency bonus", "hit points", "temporary hit points", "death save",
    "short rest", "long rest", "initiative", "opportunity attack",
    "concentration", "spell slot", "cantrip", "ritual",
    # Classes
    "Barbarian", "Bard", "Cleric", "Druid", "Fighter", "Monk", "Paladin",
    "Ranger", "Rogue", "Sorcerer", "Warlock", "Wizard", "Artificer",
    # Races
    "Dragonborn", "Dwarf", "Elf", "Gnome", "Half-Elf", "Halfling",
    "Half-Orc", "Human", "Tiefling", "Aasimar", "Goliath", "Tabaxi", "Firbolg",
    # Common spells (frequently mangled by generic ASR)
    "Fireball", "Eldritch Blast", "Counterspell", "Misty Step", "Cure Wounds",
    "Guidance", "Shield", "Magic Missile", "Hold Person", "Sanctuary",
    "Bless", "Bane", "Polymorph", "Wild Shape", "Sneak Attack",
    "Divine Smite", "Rage", "Second Wind", "Action Surge", "Bardic Inspiration",
    # Common monsters
    "goblin", "kobold", "beholder", "mind flayer", "owlbear", "displacer beast",
    "githyanki", "tarrasque", "lich", "vampire", "werewolf", "dragon wyrmling",
    # Dice / table jargon
    "d20", "d12", "d10", "d8", "d6", "d4", "nat twenty", "natural one",
    "DM", "Dungeon Master", "party", "campaign", "homebrew", "loot",
]


def load_campaign_glossary(
    path: Path | None = None,
) -> tuple[list[str], dict[str, list[str]], dict[str, str]]:
    """Parse campaign_vocab.txt.

    One term per line, `#` starts a comment, blank lines ignored -- so the
    file can be organised into labelled sections and kept under version
    control, which a single comma-separated env var can't manage once it
    grows past a handful of names.

    A line may also declare what a name gets *misheard* as:

        Llyr = clear, *cleer, *lear

    which matters for names that sound like ordinary English. Biasing the
    decoder can't win against a common word -- "clear" is far more probable
    in everyday speech than a Welsh druid -- so mishearings are handled
    after the fact instead. An alias marked with a leading `*` is also
    rewritten in the live transcript; unmarked aliases only inform the
    recap, which is the safe default when the alias is a real word.

    Returns (terms, aliases, rewrites):
      terms    -- canonical names, for the Whisper prompt
      aliases  -- {canonical: [misheard forms]}, for the recap prompt
      rewrites -- {misheard(lower): canonical} for live transcript rewriting
    """
    path = path or Path(os.getenv("CAMPAIGN_VOCAB_FILE") or DEFAULT_VOCAB_FILE)
    if not path.is_file():
        return [], {}, {}

    terms: list[str] = []
    aliases: dict[str, list[str]] = {}
    rewrites: dict[str, str] = {}

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue

        if "=" in line:
            name, _, alias_text = line.partition("=")
            name = name.strip()
            if not name:
                continue
            terms.append(name)
            for alias in (a.strip() for a in alias_text.split(",")):
                if not alias:
                    continue
                live = alias.startswith("*")
                alias = alias.lstrip("*").strip()
                if not alias:
                    continue
                aliases.setdefault(name, []).append(alias)
                if live:
                    rewrites[alias.lower()] = name
            continue

        # Tolerate comma-separated lines as well as one-per-line, so pasting
        # in an old CAMPAIGN_VOCAB value just works.
        terms.extend(part.strip() for part in line.split(",") if part.strip())

    return terms, aliases, rewrites


def load_campaign_vocab(path: Path | None = None) -> list[str]:
    """Canonical campaign terms only -- what goes into the Whisper prompt.

    Deliberately excludes aliases: seeding the decoder with "clear" would
    bias it toward the *wrong* spelling, which is the opposite of the point.
    """
    return load_campaign_glossary(path)[0]


def _rewrite_pattern(rewrites: dict[str, str]) -> "re.Pattern | None":
    if not rewrites:
        return None
    # Longest first so "cleer water" beats "cleer" when both are declared.
    alternatives = sorted((re.escape(a) for a in rewrites), key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(alternatives) + r")\b", re.IGNORECASE)


def apply_live_corrections(text: str, rewrites: dict[str, str] | None = None) -> str:
    """Rewrite starred aliases in a freshly transcribed line.

    Only aliases explicitly marked `*` in campaign_vocab.txt are touched --
    rewriting a real English word like "clear" everywhere it appears would
    corrupt far more lines than it fixed.
    """
    if rewrites is None:
        rewrites = load_campaign_glossary()[2]
    pattern = _rewrite_pattern(rewrites)
    if pattern is None:
        return text
    return pattern.sub(lambda m: rewrites[m.group(0).lower()], text)


def load_campaign_context(path: Path | None = None) -> str:
    """Read campaign_context.md -- free-form notes for the recap model.

    No parsing and no schema: whatever you write is handed to Claude as-is.
    The most valuable thing to put here is who's who, because the transcript
    labels speakers by *Discord display name* ("gossip repeater"), which
    tells the model nothing about who was talking or which character they
    were playing.
    """
    path = path or Path(os.getenv("CAMPAIGN_CONTEXT_FILE") or DEFAULT_CONTEXT_FILE)
    if not path.is_file():
        return ""
    # Strip HTML comments so the template's own instructions don't get sent.
    text = re.sub(r"<!--.*?-->", "", path.read_text(encoding="utf-8"), flags=re.DOTALL)
    return text.strip()


_SPEAKER_RE = re.compile(
    r"^\s*-\s*([^=*\n]+?)\s*=\s*([^(,\n]+?)\s*\(([^)\n]+)\)\s*(?:,\s*([^\n]+?))?\s*$"
)


def load_speakers() -> dict[str, dict]:
    """Discord display name (lowercased) -> who that is at the table.

    Read from campaign_context.md lines like `- wildgoose = Thia (Goose), Thia: she/her, Goose: he/him`.
    The transcript labels lines by Discord name, which left the recap writer
    to work out who plays whom -- and it once wrote "Goose" for Thia and gave
    Goose Thia's pronoun. Relabelling the lines first removes the guesswork.
    """
    speakers = {}
    for line in load_campaign_context().splitlines():
        match = _SPEAKER_RE.match(line)
        if not match:
            continue
        discord_name, character, player, pronouns = (g.strip() if g else "" for g in match.groups())
        label = character if player.lower() == character.lower() else f"{character} ({player})"
        speakers[discord_name.lower()] = {
            "character": character, "player": player, "pronouns": pronouns, "label": label,
        }
    return speakers


def relabel(speaker: str, speakers: dict | None = None) -> str:
    """'wildgoose' -> 'Thia (Goose)'. Unknown names, and labels, pass through."""
    entry = (load_speakers() if speakers is None else speakers).get(speaker.strip().lower())
    return entry["label"] if entry else speaker


def build_recap_briefing() -> str:
    """Everything the recap model should know about this campaign."""
    parts = [p for p in (load_campaign_context(), describe_name_corrections()) if p]
    if not parts:
        return ""
    briefing = "\n\n".join(parts)
    _log.info("recap briefing: %d chars of campaign context", len(briefing))
    return briefing


def describe_name_corrections(aliases: dict[str, list[str]] | None = None) -> str:
    """A note for the recap prompt: which mishearings mean which name.

    The recap is the artifact people actually read, and the model rewrites
    the text wholesale anyway -- so it can fix "clear" back to "Llyr" using
    context (a druid casting a spell) in a way blind substitution never
    could, without touching the raw transcript.
    """
    if aliases is None:
        aliases = load_campaign_glossary()[1]
    if not aliases:
        return ""
    lines = [
        f"- \"{name}\" is often transcribed as: {', '.join(forms)}"
        for name, forms in aliases.items()
    ]
    return (
        "This table's names are regularly misheard by the transcriber. "
        "Where the context fits, read these as the name on the left:\n"
        + "\n".join(lines)
        + "\nUse context to judge -- some of these are ordinary words too, "
        "and should be left alone when they're genuinely being used as words."
    )


def build_prompt(extra_names: str | None = None) -> str:
    """Build the Whisper `initial_prompt` for a D&D session.

    Campaign terms (from campaign_vocab.txt, CAMPAIGN_VOCAB, or the
    `extra_names` argument) are the whole point of this prompt: a generic
    speech model has no chance at "Vesper Kane" or "the Sunken Archive",
    while it already knows what a Paladin is. So campaign terms get first
    claim on the token budget, and the core 5e glossary fills whatever is
    left over.

    They're also placed *last* in the prompt. If the result still overflows
    Whisper's cap, faster-whisper truncates from the front -- so the terms
    at the end are the ones that survive.
    """
    campaign: list[str] = []
    for source in (extra_names, os.getenv("CAMPAIGN_VOCAB")):
        if source:
            campaign.extend(name.strip() for name in source.split(",") if name.strip())
    campaign.extend(load_campaign_vocab())

    # De-duplicate, case-insensitively, keeping first-seen order: the file is
    # ordered most-important-first, and that's the order we want to keep if
    # something has to be dropped.
    seen: set[str] = set()
    campaign = [t for t in campaign if not (t.lower() in seen or seen.add(t.lower()))]

    budget = MAX_PROMPT_CHARS - len(_PREFIX)
    campaign_text = ", ".join(campaign)

    if len(campaign_text) > budget:
        # The glossary alone overflows. Keep as many leading terms as fit and
        # say so loudly -- silently dropping half someone's NPC list is
        # exactly the kind of failure nobody notices until accuracy is bad.
        kept: list[str] = []
        used = 0
        for term in campaign:
            cost = len(term) + 2
            if used + cost > budget:
                break
            kept.append(term)
            used += cost
        _log.warning(
            "campaign vocabulary is too long for Whisper's %d-token prompt: keeping "
            "the first %d of %d terms, dropping %s. Trim campaign_vocab.txt -- "
            "entries near the top are kept first.",
            PROMPT_TOKEN_BUDGET, len(kept), len(campaign), campaign[len(kept):],
        )
        campaign, campaign_text = kept, ", ".join(kept)
        core: list[str] = []
    else:
        # Fill the remaining space with core terms, in order.
        remaining = budget - len(campaign_text)
        core = []
        used = 0
        for term in _CORE_TERMS:
            cost = len(term) + 2
            if used + cost > remaining:
                break
            core.append(term)
            used += cost
        if len(core) < len(_CORE_TERMS):
            _log.info(
                "core 5e glossary trimmed to %d of %d terms to make room for %d "
                "campaign terms (Whisper prompt cap)",
                len(core), len(_CORE_TERMS), len(campaign),
            )

    terms = core + campaign
    prompt = _PREFIX + ", ".join(terms) + "."
    _log.info(
        "whisper prompt: %d chars (~%d tokens), %d core + %d campaign terms",
        len(prompt), int(len(prompt) / _CHARS_PER_TOKEN), len(core), len(campaign),
    )
    return prompt
