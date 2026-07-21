import importlib.util
import io
import logging
import os
import sys
import threading
import wave

_log = logging.getLogger(__name__)


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

from transcription.dnd_vocabulary import build_prompt

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
                )
                return " ".join(segment.text.strip() for segment in segments).strip()
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
