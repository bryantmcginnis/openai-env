"""Release notes: what changed in the bot since it last announced, from git.

The bot reads its own git history at startup. Commits it hasn't announced yet
get turned into notes for the table (by the recap model, from the commit
messages and the code diff) and posted to #dndbot-release-notes. Only
*committed* code counts -- half-finished edits on disk are never announced.

State lives in campaign_memory/release_notes.json; every post is also kept
in release_notes/ as markdown.
"""

import json
import logging
import re
import subprocess
from pathlib import Path

from discord_bot import campaign_memory as memory

_log = logging.getLogger(__name__)

CURRENT = Path(__file__).resolve().parents[1]  # the Current/ folder
ARCHIVE = CURRENT / "release_notes"
HISTORY = ARCHIVE / "history.md"
STATE_FILE = "release_notes.json"

# Big enough for a feature's worth of changes, small enough to stay a cheap
# call. Data files (threads, quotes, recaps) aren't code and are left out.
MAX_DIFF_CHARS = 80_000
DIFF_EXCLUDES = [":(exclude)*campaign_memory/*", ":(exclude)*release_notes/*", ":(exclude)*.json"]


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=CURRENT, capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:2])} failed: {result.stderr.strip()[:200]}")
    return result.stdout


def head() -> str:
    return _git("rev-parse", "HEAD").strip()


def load_state() -> dict:
    return memory._read(STATE_FILE, {})


def save_state(state: dict) -> None:
    memory._write(STATE_FILE, state)


def pending() -> tuple[str, str] | None:
    """(from, to) commits to announce, or None if there's nothing new.

    The first time it ever runs, it marks the current commit as announced:
    everything up to there is covered by the hand-written history post.
    """
    state, now = load_state(), head()
    last = state.get("announced")
    if not last:
        save_state({**state, "announced": now})
        return None
    if last == now:
        return None
    try:
        _git("merge-base", "--is-ancestor", last, now)
    except RuntimeError:
        # History was rewritten; there's no clean "since". Start again from here.
        _log.warning("release notes: %s isn't an ancestor of HEAD; skipping to %s", last[:7], now[:7])
        save_state({**state, "announced": now})
        return None
    return last, now


def changes(since: str, until: str) -> tuple[str, str]:
    """(commit log, code diff) between two commits, for the notes writer."""
    log = _git("log", "--reverse", "--format=## %h, %ad%n%s%n%b", "--date=short", f"{since}..{until}")
    stat = _git("diff", "--stat", f"{since}..{until}", "--", ".", *DIFF_EXCLUDES)
    diff = _git("diff", "--unified=2", f"{since}..{until}", "--", ".", *DIFF_EXCLUDES)
    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS] + "\n\n[diff truncated -- the stat above lists every file]"
    return log.strip(), f"{stat}\n{diff}"


def mark_announced(until: str, notes: str) -> Path:
    """Record the post: state moves on, and the notes are archived."""
    save_state({**load_state(), "announced": until})
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    date = _git("log", "-1", "--format=%ad", "--date=short", until).strip()
    path = ARCHIVE / f"{date}_{until[:7]}.md"
    path.write_text(notes.strip() + "\n", encoding="utf-8")
    return path


def earlier_posts(limit: int = 12_000) -> str:
    """The history post plus the newest archived posts, newest last -- so the
    writer matches the voice and doesn't announce old features as new."""
    posts = sorted(p for p in ARCHIVE.glob("*.md") if p != HISTORY)[-3:]
    text = "\n\n---\n\n".join(p.read_text(encoding="utf-8") for p in [HISTORY, *posts] if p.exists())
    return text[-limit:]


def history_posted() -> bool:
    return bool(load_state().get("history_posted"))


def mark_history_posted() -> None:
    save_state({**load_state(), "history_posted": True})


def split_for_discord(text: str, limit: int = 1900) -> list[str]:
    """Split at headings, then blank lines, so no message cuts a section mid-thought."""
    parts = re.split(r"\n(?=#{1,3} )", text.strip())
    out, chunk = [], ""
    for part in parts:
        for para in (part.split("\n\n") if len(part) > limit else [part]):
            piece = para if len(para) <= limit else para[:limit]
            joiner = "\n\n" if chunk and not piece.startswith("#") else "\n"
            if chunk and len(chunk) + len(joiner) + len(piece) > limit:
                out.append(chunk)
                chunk = ""
            chunk = f"{chunk}{joiner}{piece}" if chunk else piece
    if chunk:
        out.append(chunk)
    return out
