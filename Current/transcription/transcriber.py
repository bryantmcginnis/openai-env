import importlib.util
import io
import logging
import os
import re
import sys
import threading
import wave
from pathlib import Path

_log = logging.getLogger(__name__)


# Whisper was trained on a very large amount of YouTube audio paired with its
# subtitles, so when it's handed a segment with no real speech in it -- a
# breath, a chair creak, mic hiss that VAD let through -- the likeliest
# "caption" for that kind of audio is a video outro. It hallucinates one.
# Nobody said these; the model invented them from noise. Filtering them here
# is standard practice with Whisper and costs nothing: if a player genuinely
# says "thanks for watching" at the table, it lands inside a longer sentence
# and only an exact whole-utterance match is dropped.
_HALLUCINATION_PHRASES = {
    "thanks for watching",
    "thanks for watching!",
    "thank you for watching",
    "thank you for watching!",
    "thanks for listening",
    "thank you",
    "thank you.",
    "thank you very much",
    "you",
    "bye",
    "bye!",
    "bye bye",
    "okay",
    "so",
    "oh",
    "please subscribe",
    "subscribe to my channel",
    "like and subscribe",
    "don't forget to subscribe",
    "see you next time",
    "see you in the next video",
    "i'll see you in the next video",
    "see you in the next one",
    "i'll see you next time",
    "thanks for watching and i'll see you in the next video",
    "don't forget to like and subscribe",
    "like comment and subscribe",
    "hit the like button",
    "for more information visit www fema gov",
    "transcription by castingwords",
    "subtitles by the amara.org community",
    "amara.org",
    "www.mooji.org",
}

_PUNCTUATION_RE = re.compile(r"[.,!?;:\-—–\"']+")
_WHITESPACE_RE = re.compile(r"\s+")

# Filler left over after the phantom phrases are stripped out. Whisper glues
# its outro phrases together with these ("Thank you for watching AND I'll see
# you in the next video"), so a remainder made only of these is still noise.
# Kept deliberately short: every word here is one that can no longer save an
# utterance from being dropped.
_CONNECTIVES = {
    "and", "then", "so", "but", "also", "um", "uh", "er", "ah", "oh",
    "please", "bye", "thanks", "ok", "okay", "yeah", "well", "right", "hey",
    # Apostrophe fragments left behind by _normalize ("i'll" -> "i ll").
    "i", "ll", "ve", "re",
}

# Extra phrases can be added without touching this file.
DEFAULT_HALLUCINATION_FILE = Path(__file__).resolve().parent.parent / "hallucinations.txt"


def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", _PUNCTUATION_RE.sub(" ", text.lower())).strip()


def _load_extra_phrases(path: Path | None = None) -> set[str]:
    """User-supplied phantom phrases from hallucinations.txt."""
    path = path or Path(os.getenv("HALLUCINATION_FILE") or DEFAULT_HALLUCINATION_FILE)
    if not path.is_file():
        return set()
    phrases = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = _normalize(raw.split("#", 1)[0])
        if line:
            phrases.add(line)
    return phrases


_phrase_pattern: "re.Pattern | None" = None


def _phrase_regex() -> "re.Pattern":
    """One alternation matching any known phantom phrase.

    Built from *normalized* phrases: the text has already had its
    apostrophes stripped by _normalize, so a raw "i'll see you in the next
    video" in the list would never match "i ll see you in the next video".

    Regex rather than repeated str.replace because adjacent repeats
    ("Thanks for watching! Thanks for watching!") defeat a space-padded
    replace -- the first match consumes the separator the second one needs.
    """
    global _phrase_pattern
    if _phrase_pattern is None:
        extra = _load_extra_phrases()
        if extra:
            _log.info("loaded %d extra hallucination phrases", len(extra))
        phrases = {_normalize(p) for p in _HALLUCINATION_PHRASES | extra}
        phrases.discard("")
        # Longest first so "see you in the next video" wins over "you".
        alternation = "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True))
        _phrase_pattern = re.compile(rf"\b(?:{alternation})\b")
    return _phrase_pattern


def _is_hallucination(text: str) -> bool:
    """True if the utterance is nothing but known Whisper phantom phrases.

    Matching a whole utterance against a fixed list isn't enough: Whisper
    happily welds its outros together into new combinations ("Thank you for
    watching and I'll see you in the next video"), and no list of exact
    strings keeps up with that. So instead of matching, this *subtracts* --
    strip every known phantom out of the line, and if what's left is only
    filler, the whole thing was noise.

    Real speech survives because subtracting a phantom from it leaves
    content behind: "thanks for watching the door" keeps "the door".
    """
    normalized = _normalize(text)
    if not normalized:
        return True

    remainder = _phrase_regex().sub(" ", normalized)
    leftover = [word for word in remainder.split() if word not in _CONNECTIVES]
    return not leftover


# os.add_dll_directory() returns a handle that removes the directory from
# the DLL search path when garbage-collected -- it must be kept alive for
# the process's lifetime, or the registration undoes itself as soon as the
# call statement finishes. This is exactly what was happening: whichever
# DLL got touched during model construction (probably cuDNN) was already
# cached in the process by the time the handle was collected, but
# cublas64_12.dll is only ever needed later, during the first real
# inference call -- by then the search path entry was long gone.
_cuda_dll_dirs: list[object] = []


def _register_cuda_dll_dirs() -> None:
    # ctranslate2 loads cuBLAS/cuDNN via the OS DLL search path, which the
    # pip-installed nvidia-cublas-cu12/nvidia-cudnn-cu12 wheels sit outside
    # of. Registering their bin/ dirs here means GPU support only needs
    # `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` -- no system-wide
    # CUDA Toolkit install or manual PATH editing.
    if sys.platform != "win32":
        return
    for pkg in ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_nvrtc", "nvidia.nvjitlink"):
        spec = importlib.util.find_spec(pkg)
        if spec is None or not spec.submodule_search_locations:
            continue
        bin_dir = os.path.join(spec.submodule_search_locations[0], "bin")
        if not os.path.isdir(bin_dir):
            continue
        _cuda_dll_dirs.append(os.add_dll_directory(bin_dir))
        # Belt and suspenders: add_dll_directory() only affects LoadLibraryEx
        # calls that opt into the extended search, and this has failed
        # unpredictably inside the live bot process (but never in an
        # isolated repro with identical code) -- something in that process
        # likely disturbs it after the fact, e.g. a dependency calling the
        # legacy single-slot SetDllDirectory. Prepending to PATH uses the
        # classic search order instead, which virtually everything respects
        # and which nothing here should be able to silently undo.
        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


_register_cuda_dll_dirs()

from faster_whisper import WhisperModel  # noqa: E402 -- must follow DLL dir registration above

from transcription.dnd_vocabulary import (  # noqa: E402 -- follows DLL dir registration
    apply_live_corrections,
    build_prompt,
    load_campaign_glossary,
)

MODEL_SIZE = "large-v3"
SAMPLE_RATE = 48000  # Discord's raw PCM sample rate


class Transcriber:
    """Wraps a local faster-whisper model for D&D-session voice transcription.

    Loading the large-v3 model is what actually fixes most "mishearing"
    complaints -- a lot of bad-Whisper reports turn out to be someone
    accidentally running the tiny/base model, which is dramatically worse at
    accents, cross-talk, and anything not in everyday vocabulary.
    """

    def __init__(self, device: str | None = None, compute_type: str | None = None):
        # device="auto" makes ctranslate2 itself pick CUDA if it's usable,
        # but on Windows a GPU can be *present* while cuBLAS/cuDNN are not
        # correctly on PATH -- that fails at model load, not at detection.
        # Try GPU first and fall back to CPU rather than crashing the bot.
        # .env leaves these blank rather than unset (WHISPER_DEVICE=), and
        # python-dotenv loads a blank value as "", not absent -- os.getenv's
        # default only kicks in when the key is missing entirely, so an "or"
        # chain is needed to treat blank the same as unset.
        device = device or os.getenv("WHISPER_DEVICE") or "auto"
        compute_type = compute_type or os.getenv("WHISPER_COMPUTE_TYPE") or "default"

        try:
            self.model = WhisperModel(MODEL_SIZE, device=device, compute_type=compute_type)
            # device may be "auto" -- read back what ctranslate2 actually
            # resolved it to so callers (e.g. the "Recording on ..." message)
            # report the real device instead of echoing "auto".
            self.device = self.model.model.device
        except Exception as exc:
            if device == "cpu":
                raise
            _log.warning(
                "GPU init failed (%r) -- falling back to CPU. On Windows this usually "
                "means the nvidia-cublas-cu12/nvidia-cudnn-cu12 pip packages aren't "
                "installed, or the DLL directory registration in "
                "_register_cuda_dll_dirs() above didn't find them; see README.md's GPU "
                "setup section. Set WHISPER_DEVICE=cpu in .env to skip this check and "
                "always run on CPU.",
                exc,
            )
            self.model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
            self.device = "cpu"

        _log.info("loaded %s on device=%r compute_type=%r", MODEL_SIZE, self.device, compute_type)
        self.prompt = build_prompt()
        # Read once at load rather than per utterance -- this runs on the
        # hot path for every segment of speech.
        _, self.aliases, self.rewrites = load_campaign_glossary()
        if self.rewrites:
            _log.info("live transcript rewrites active: %s", self.rewrites)

    def transcribe_pcm(self, pcm_bytes: bytes, channels: int = 2, sample_width: int = 2) -> str:
        """Transcribe one utterance of raw PCM audio (as captured by the
        Discord voice sink) and return the recognized text, or "" if
        nothing intelligible was found.

        Audio is wrapped as an in-memory WAV rather than hand-resampled --
        faster-whisper's own decode/resample pipeline is more reliable than
        a homemade one, and this avoids adding a resampling dependency.
        """
        if not pcm_bytes:
            return ""

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(pcm_bytes)
        wav_buffer.seek(0)

        # vad_filter drops silence/non-speech instead of feeding it to the
        # model as if it were words -- another common source of garbage
        # output when a fixed-size chunk lands mostly on silence.
        #
        # This has intermittently failed inside the live bot (but never in
        # an isolated repro with matching threading) with "Library
        # cublas64_12.dll is not found or cannot be loaded", even with the
        # DLL search directories registered and kept alive -- something
        # elsewhere in the live process's thread/library state appears to
        # disturb the search path after the fact. Retry once with the
        # directories re-registered and log enough state to actually
        # diagnose it next time instead of guessing blind again.
        for attempt in range(2):
            try:
                wav_buffer.seek(0)
                segments, _info = self.model.transcribe(
                    wav_buffer,
                    language="en",
                    initial_prompt=self.prompt,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=300),
                    condition_on_previous_text=False,
                    # Raised from the 0.6 default: how confident the model has
                    # to be that a segment contains *no* speech before it's
                    # thrown away. Higher discards more marginal audio, which
                    # is where the "Thanks for watching!" hallucinations come
                    # from -- they're generated from segments that are really
                    # just breath and room noise.
                    no_speech_threshold=0.8,
                    # Drop segments the model itself is unsure about, and ones
                    # whose text compresses suspiciously well -- the signature
                    # of a repetition loop ("Thanks for watching! Thanks for
                    # watching! ...").
                    log_prob_threshold=-0.8,
                    compression_ratio_threshold=2.2,
                )
                text = " ".join(segment.text.strip() for segment in segments).strip()
                if _is_hallucination(text):
                    _log.debug("dropping hallucinated/empty transcript: %r", text)
                    return ""
                # Fix the spellings the decoder can't be biased into getting
                # right -- names that sound like ordinary words.
                return apply_live_corrections(text, self.rewrites)
            except RuntimeError as exc:
                if "cublas" not in str(exc).lower() or attempt == 1:
                    raise
                _log.error(
                    "cuBLAS DLL load failed on thread %r (attempt %d/2); "
                    "registered dll dirs=%r -- re-registering and retrying",
                    threading.current_thread().name, attempt + 1, _cuda_dll_dirs,
                    exc_info=exc,
                )
                _register_cuda_dll_dirs()
        return ""  # unreachable: loop always returns or raises
