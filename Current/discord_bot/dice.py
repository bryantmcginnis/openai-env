"""Dice notation parsing for !roll.

Handles the notation people actually type at a 5e table: `1d20+5`,
`2d6+1d4+3`, `4d6kh3` for stat rolling, and advantage/disadvantage.
"""

import random
import re
from dataclasses import dataclass, field

# Same spirit as the original caps -- generous enough for fireball damage
# (8d6) or rolling a stat array, tight enough that nobody can ask for a
# million dice and stall the bot.
MAX_DICE_PER_TERM = 100
MAX_TOTAL_DICE = 200
MAX_SIDES = 1000
MAX_TERMS = 10


class DiceError(ValueError):
    """Raised for anything we can't parse or won't roll. The message is
    written to be shown directly to the user."""


# [count]d<sides>[kh|kl|dh|dl][n] -- count and the keep/drop count are both
# optional, so `d20`, `4d6kh3` and `2d20kh` all parse.
_TERM_RE = re.compile(r"^(\d*)d(\d+)(?:(kh|kl|dh|dl)(\d*))?$", re.IGNORECASE)

_ADVANTAGE_WORDS = {"adv", "advantage", "a"}
_DISADVANTAGE_WORDS = {"dis", "disadv", "disadvantage", "d"}


@dataclass
class Term:
    """One dice term or flat modifier within an expression."""

    sign: int
    notation: str
    count: int = 0  # 0 marks a flat modifier rather than dice
    sides: int = 0
    keep: int = 0
    keep_high: bool = True
    modifier: int = 0
    rolls: list[int] = field(default_factory=list)
    kept_indices: set[int] = field(default_factory=set)

    @property
    def is_dice(self) -> bool:
        return self.count > 0

    @property
    def subtotal(self) -> int:
        if not self.is_dice:
            return self.modifier
        return sum(self.rolls[i] for i in self.kept_indices)


@dataclass
class RollResult:
    expression: str
    terms: list[Term]
    total: int
    advantage: str | None = None  # "advantage" | "disadvantage" | None


def _parse_term(raw: str, sign: int) -> Term:
    match = _TERM_RE.match(raw)
    if match is None:
        if raw.isdigit():
            return Term(sign=sign, notation=raw, modifier=int(raw))
        raise DiceError(
            f"I don't understand `{raw}`. Try `2d6`, `1d20+5`, `4d6kh3`, or `1d20 adv`."
        )

    count_str, sides_str, keep_mode, keep_str = match.groups()
    count = int(count_str) if count_str else 1
    sides = int(sides_str)

    if count < 1:
        raise DiceError("You have to roll at least one die.")
    if count > MAX_DICE_PER_TERM:
        raise DiceError(f"Keep it reasonable -- up to {MAX_DICE_PER_TERM} dice per term.")
    if not (1 <= sides <= MAX_SIDES):
        raise DiceError(f"Dice need between 1 and {MAX_SIDES} sides.")

    keep = count
    keep_high = True
    if keep_mode:
        keep_mode = keep_mode.lower()
        # kh3 keeps the highest 3; dh1 *drops* the highest 1, i.e. keeps
        # count-1 of them from the other end.
        amount = int(keep_str) if keep_str else 1
        if amount < 1:
            raise DiceError(f"`{raw}`: the keep/drop count has to be at least 1.")
        if amount > count:
            raise DiceError(f"`{raw}`: can't keep or drop more dice than you rolled.")
        if keep_mode in ("kh", "kl"):
            keep, keep_high = amount, keep_mode == "kh"
        else:
            keep, keep_high = count - amount, keep_mode == "dl"
        if keep < 1:
            raise DiceError(f"`{raw}`: that drops every die.")

    return Term(sign=sign, notation=raw, count=count, sides=sides, keep=keep, keep_high=keep_high)


def _split_flags(expression: str) -> tuple[str, str | None]:
    """Pull a trailing adv/dis word off the expression, if present."""
    advantage = None
    words = expression.replace(",", " ").split()
    kept: list[str] = []
    for word in words:
        lowered = word.lower()
        if lowered in _ADVANTAGE_WORDS:
            advantage = "advantage"
        elif lowered in _DISADVANTAGE_WORDS:
            advantage = "disadvantage"
        else:
            kept.append(word)

    # Close up spaces around operators so `1d20 + 5` works, but leave any
    # other whitespace intact -- naively joining every word would silently
    # turn `1d20 5` into `1d205`, a d205 roll nobody asked for.
    expression = re.sub(r"\s*([+-])\s*", r"\1", " ".join(kept))
    if " " in expression:
        raise DiceError(
            f"I don't understand `{expression}` -- terms need to be joined by "
            f"`+` or `-`, like `1d20+5`."
        )
    return expression, advantage


def parse(expression: str) -> tuple[list[Term], str | None]:
    expression, advantage = _split_flags(expression)

    # Bare `!roll adv` is a d20 check with advantage -- the single most
    # common thing anyone wants, so don't make them type the d20.
    if not expression:
        if advantage is None:
            raise DiceError("Roll what? Try `2d6`, `1d20+5`, `4d6kh3`, or `1d20 adv`.")
        expression = "1d20"

    # Split on +/- while keeping the sign attached to the following term.
    pieces = re.split(r"([+-])", expression)
    if pieces and pieces[0] == "":
        pieces = pieces[1:]  # leading sign, e.g. "-2+1d4"

    terms: list[Term] = []
    sign = 1
    expecting_term = False  # set by an operator, cleared by the term after it
    for piece in pieces:
        if piece == "+":
            sign, expecting_term = 1, True
        elif piece == "-":
            sign, expecting_term = -1, True
        elif piece.strip():
            terms.append(_parse_term(piece.strip(), sign))
            expecting_term = False
        elif expecting_term:
            # A dangling `+` at the end -- `1d20+` almost always means the
            # modifier got cut off, so say so rather than rolling without it.
            raise DiceError(f"`{expression}` trails off -- what comes after the sign?")

    if not terms:
        raise DiceError("Roll what? Try `2d6`, `1d20+5`, `4d6kh3`, or `1d20 adv`.")
    if len(terms) > MAX_TERMS:
        raise DiceError(f"That's a lot of terms -- keep it to {MAX_TERMS} or fewer.")
    if sum(term.count for term in terms) > MAX_TOTAL_DICE:
        raise DiceError(f"Keep it reasonable -- up to {MAX_TOTAL_DICE} dice in one roll.")

    if advantage is not None:
        _apply_advantage(terms, advantage)
    return terms, advantage


def _apply_advantage(terms: list[Term], advantage: str) -> None:
    """Advantage rolls the first dice term twice and keeps the better half.

    For the usual `1d20+5` this is exactly 5e's rule (roll two d20s, take the
    higher). Generalising it as "double the dice, keep the original number"
    means `2d6 adv` behaves sensibly too.
    """
    for term in terms:
        if term.is_dice:
            term.keep = term.count
            term.count *= 2
            term.keep_high = advantage == "advantage"
            return
    raise DiceError("Advantage needs dice to roll -- try `1d20 adv`.")


def roll(expression: str) -> RollResult:
    """Parse and roll an expression. Raises DiceError with a user-facing message."""
    terms, advantage = parse(expression)

    total = 0
    for term in terms:
        if term.is_dice:
            term.rolls = [random.randint(1, term.sides) for _ in range(term.count)]
            # Rank by value but keep the original positions, so the breakdown
            # can show the dice in the order they were rolled with the
            # discarded ones struck through in place.
            order = sorted(
                range(term.count), key=lambda i: term.rolls[i], reverse=term.keep_high
            )
            term.kept_indices = set(order[: term.keep])
        total += term.sign * term.subtotal

    return RollResult(expression=expression, terms=terms, total=total, advantage=advantage)


def format_result(result: RollResult) -> str:
    """Render a roll as a Discord message: total first, then the breakdown."""
    parts: list[str] = []
    for index, term in enumerate(result.terms):
        joiner = "" if index == 0 else (" + " if term.sign > 0 else " - ")
        if index == 0 and term.sign < 0:
            joiner = "-"

        if term.is_dice:
            dice = ", ".join(
                str(value) if i in term.kept_indices else f"~~{value}~~"
                for i, value in enumerate(term.rolls)
            )
            parts.append(f"{joiner}{term.notation} [{dice}]")
        else:
            parts.append(f"{joiner}{term.modifier}")

    breakdown = "".join(parts)
    label = f" *({result.advantage})*" if result.advantage else ""
    # A single unmodified die needs no "= [7]" restatement of itself.
    only_die = len(result.terms) == 1 and result.terms[0].is_dice
    if only_die and result.terms[0].count == 1:
        return f"**{result.total}**{label}"
    return f"**{result.total}**{label} = {breakdown}"
