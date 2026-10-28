"""The campaign's long-term memory, kept as plain files in campaign_memory/.

Committed to git on purpose (the table's call): outside Discord, this is the
only copy of the open-threads list and the recap archive.

  recaps/YYYY-MM-DD_HHMM.md  one file per party recap, keyed by the session's
                             start, so re-running a recap overwrites it
  threads.json               open and resolved plot threads, plus the id of
                             the bot's board message in the to-do channel
  quotes.json                quotes already posted, so nothing posts twice
  active_recording.json      written while recording; lets a restart after a
                             crash rejoin voice and carry on
"""

import datetime
import difflib
import json
import os
import re
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "campaign_memory"

_lock = threading.Lock()  # reader thread, event loop and to_thread workers all write here


def _path(name: str) -> Path:
    return ROOT / name


def _read(name: str, default):
    try:
        return json.loads(_path(name).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def _write(name: str, data) -> None:
    path = _path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)  # atomic: a crash mid-write can't leave a half-written list


def _today() -> str:
    return datetime.date.today().isoformat()


def _pretty_date(iso: str) -> str:
    try:
        return datetime.date.fromisoformat(iso).strftime("%d %b").lstrip("0")
    except ValueError:
        return iso


# --- recap archive ----------------------------------------------------------------

def _session_key(session_start: datetime.datetime) -> str:
    return f"{session_start.astimezone():%Y-%m-%d_%H%M}"


def save_recap(text: str, session_start: datetime.datetime) -> Path:
    """Archive a party recap, with the table's corrections for that session
    appended -- which is what lets a later question or re-run get it right."""
    local = session_start.astimezone()
    path = _path("recaps") / f"{_session_key(session_start)}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = f"# Session recap, {local:%d %b %Y}\n\n{text.strip()}\n"
    fixes = corrections_for(session_start)
    if fixes:
        body += f"\n## Corrections from the table\n\n{fixes}\n"
    path.write_text(body, encoding="utf-8")
    return path


def add_correction(session_start: datetime.datetime, author: str, text: str) -> None:
    with _lock:
        data = _read("corrections.json", {})
        data.setdefault(_session_key(session_start), []).append(
            {"author": author, "text": text.strip(), "date": _today()})
        _write("corrections.json", data)


def corrections_for(session_start: datetime.datetime) -> str:
    """Everything the table has corrected about one session, as notes for the model."""
    entries = _read("corrections.json", {}).get(_session_key(session_start), [])
    return "\n".join(f"- ({e['author']}, {_pretty_date(e['date'])}) {e['text']}" for e in entries)


def recent_recaps(count: int = 5) -> list[str]:
    files = sorted(_path("recaps").glob("*.md"))[-count:]
    return [f.read_text(encoding="utf-8") for f in files]


# --- threads ------------------------------------------------------------------------

def load_threads() -> dict:
    return _read("threads.json", {"next_id": 1, "board_message_id": None, "threads": []})


def save_threads(data: dict) -> None:
    with _lock:
        _write("threads.json", data)


def open_threads(data: dict) -> list[dict]:
    return [t for t in data["threads"] if t["status"] == "open"]


def _similar(a: str, b: str) -> bool:
    """Near-identical restatements only. Deliberately strict: merging two real
    threads that differ by one word ("the silver key" / "the golden key")
    silently loses one, which is worse than an occasional duplicate -- and
    the model is already told not to restate threads that are open."""
    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio() >= 0.92


def add_thread(data: dict, title: str, detail: str = "") -> dict | None:
    """Add an open thread. None if an open thread already says the same thing --
    re-running a recap for the same night must not double the list."""
    title = title.strip()
    if not title or any(_similar(title, t["title"]) for t in open_threads(data)):
        return None
    thread = {
        "id": data["next_id"], "title": title, "detail": detail.strip(),
        "status": "open", "opened": _today(),
    }
    data["next_id"] += 1
    data["threads"].append(thread)
    return thread


def _find(data: dict, thread_id: int) -> dict | None:
    return next((t for t in data["threads"] if t["id"] == thread_id), None)


def resolve_thread(data: dict, thread_id: int, how: str, quip: str = "") -> dict | None:
    thread = _find(data, thread_id)
    if thread is None or thread["status"] != "open":
        return None
    thread.update(status="resolved", resolved=_today(), how=how.strip(), quip=quip.strip())
    return thread


def update_thread(data: dict, thread_id: int, title: str, detail: str) -> dict | None:
    thread = _find(data, thread_id)
    if thread is None:
        return None
    thread["title"], thread["detail"] = title.strip() or thread["title"], detail.strip()
    return thread


def reopen_thread(data: dict, thread_id: int) -> dict | None:
    thread = _find(data, thread_id)
    if thread is None or thread["status"] == "open":
        return None
    thread["status"] = "open"
    for key in ("resolved", "how", "quip"):
        thread.pop(key, None)
    return thread


BOARD_LIMIT = 1900  # Discord's hard limit is 2000 per message


def render_board(data: dict, party: str = "The Partnership of the Jade Frog") -> str:
    """The pinned to-do message. Same tick style the channel has always used."""
    head = [
        f"📜 **Open threads: {party}**",
        "*Kept by the bot from each session recap. `!threads` to see them, "
        "`!threads add / resolve / reopen` to change them.*",
        "",
    ]
    open_lines = [
        f"⬜ **#{t['id']} {t['title']}**" + (f" -- {t['detail']}" if t["detail"] else "")
        + f" *(since {_pretty_date(t['opened'])})*"
        for t in open_threads(data)
    ] or ["*Nothing open. Suspicious.*"]
    done = sorted((t for t in data["threads"] if t["status"] == "resolved"),
                  key=lambda t: (t.get("resolved", ""), t["id"]), reverse=True)[:6]
    done_lines = [
        f"✅ ~~#{t['id']} {t['title']}~~ -- {_pretty_date(t['resolved'])}: {t['how']}"
        + (f" *{t['quip']}*" if t.get("quip") else "")
        for t in done
    ]

    out = list(head)
    for i, line in enumerate(open_lines):
        if len("\n".join(out + [line])) > BOARD_LIMIT - 120:
            out.append(f"…and {len(open_lines) - i} more -- `!threads`")
            break
        out.append(line)
    if done_lines:
        out += ["", "**Recently resolved**"]
        for line in done_lines:
            if len("\n".join(out + [line])) > BOARD_LIMIT:
                break
            out.append(line)
    return "\n".join(out)


# --- quotes -------------------------------------------------------------------------

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", text.lower())).strip()


def appears_in(quote: str, transcript: str) -> bool:
    """True only if the quote was really said, near enough word for word.

    Stops the recap model's "notable quotes" from posting a line it tidied
    up or invented. Punctuation and case are ignored; anything else has to
    match a stretch of the transcript at least 90%.
    """
    q, t = _norm(quote), _norm(transcript)
    if len(q) < 8:
        return False
    if q in t:
        return True
    for line in transcript.splitlines():
        body = _norm(line.split(":", 1)[1] if ":" in line else line)
        if len(body) < len(q) * 0.8:
            continue
        for start in range(0, max(1, len(body) - len(q) + 1), 8):
            window = body[start:start + len(q) + 10]
            if difflib.SequenceMatcher(None, q, window).ratio() >= 0.9:
                return True
    return False


def load_quotes() -> dict:
    return _read("quotes.json", {"quotes": []})


def remember_quote(speaker: str, text: str, source: str, message_id: int | None = None,
                   context: str = "") -> bool:
    """Record a quote. False if it's already on the board -- check before posting."""
    with _lock:
        data = load_quotes()
        key = _norm(text)
        if any(_norm(q["text"]) == key or (message_id and q.get("message_id") == message_id)
               for q in data["quotes"]):
            return False
        data["quotes"].append({"speaker": speaker, "text": text, "source": source,
                               "date": _today(), "message_id": message_id, "context": context})
        _write("quotes.json", data)
        return True


def set_quote_post(texts: list[str], posted_message_id: int) -> None:
    """Remember which #quotes message each quote lives in, so a correction can edit it."""
    keys = {_norm(t) for t in texts}
    with _lock:
        data = load_quotes()
        for q in data["quotes"]:
            if _norm(q["text"]) in keys:
                q["posted_message_id"] = posted_message_id
        _write("quotes.json", data)


def set_quote_context(text: str, context: str) -> int | None:
    """Attach the setup line to a quote. Returns its #quotes message, to be re-rendered."""
    key = _norm(text)
    with _lock:
        data = load_quotes()
        for q in data["quotes"]:
            if _norm(q["text"]) == key:
                q["context"] = context.strip()
                _write("quotes.json", data)
                return q.get("posted_message_id")
    return None


def quotes_since(date_iso: str) -> list[dict]:
    """Quotes recorded on or after a date, with their index for corrections."""
    return [dict(q, index=i) for i, q in enumerate(load_quotes()["quotes"]) if q["date"] >= date_iso]


def fix_quote_speaker(index: int, speaker: str) -> int | None:
    """Re-credit a quote. Returns the #quotes message it's in, to be re-rendered."""
    with _lock:
        data = load_quotes()
        if not 0 <= index < len(data["quotes"]):
            return None
        data["quotes"][index]["speaker"] = speaker.strip()
        _write("quotes.json", data)
        return data["quotes"][index].get("posted_message_id")


def quotes_in_post(posted_message_id: int) -> list[dict]:
    return [q for q in load_quotes()["quotes"] if q.get("posted_message_id") == posted_message_id]


# --- active recording (for resume after a crash) ------------------------------------

def mark_recording(guild_id: int, voice_channel_id: int, chat_channel_id: int) -> None:
    with _lock:
        data = _read("active_recording.json", {})
        data[str(guild_id)] = {
            "voice_channel_id": voice_channel_id,
            "chat_channel_id": chat_channel_id,
            "since": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        _write("active_recording.json", data)


def clear_recording(guild_id: int | None = None) -> None:
    """Forget one guild's recording, or all of them."""
    with _lock:
        data = {} if guild_id is None else _read("active_recording.json", {})
        data.pop(str(guild_id), None)
        _write("active_recording.json", data)


def pending_resumes(max_age_hours: float = 6) -> dict[int, dict]:
    """Recordings that were running when the bot last died, young enough to resume."""
    now = datetime.datetime.now(datetime.timezone.utc)
    out = {}
    for gid, info in _read("active_recording.json", {}).items():
        since = datetime.datetime.fromisoformat(info["since"])
        if now - since <= datetime.timedelta(hours=max_age_hours):
            out[int(gid)] = info
    return out
