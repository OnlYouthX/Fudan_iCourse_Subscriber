"""Offline LaTeX → PNG rendering for the HTML e-mail.

Formulas used to be rendered by fetching PNGs from CodeCogs
(``latex.codecogs.com``).  That service has been renamed *fxTeX Server* and
now appends an **“Unlicensed. Register at fxeditor.uk”** notice to every
equation requested by a caller it cannot recognise.  A GitHub Actions
runner has no registered domain (and no browser ``Referer``), so every
formula in the daily e-mail arrived with a pink watermark strip underneath.
The strip also enlarges the image, which pushed inline formulas above the
text baseline.

This module renders locally instead, so nothing depends on a third-party
service:

1. **tectonic** — a self-contained XeTeX engine, downloaded once into
   ``bin/tectonic`` (or taken from ``$TECTONIC_BIN`` / ``PATH``).  Full
   LaTeX fidelity, including ``pmatrix`` / ``cases`` / ``aligned``.
2. **matplotlib mathtext** — pure-Python fallback for machines where the
   tectonic binary is unavailable.  A LaTeX subset only; whatever it cannot
   parse falls through to the plain-text fallback in :mod:`src.api.emailer`.

Public API::

    render_batch([("x^2", False), (r"\\frac{a}{b}", True)], dpi=300)
    -> [(width, height, png_bytes) | None, ...]

``width``/``height`` are CSS pixels (96 dpi) because the e-mail HTML is
authored in CSS pixels, while the image itself is rendered at ``dpi``.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import urllib.request
from io import BytesIO

from PIL import Image

_TECTONIC_VERSION = "0.17.0"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BIN_DIR = os.path.join(_REPO_ROOT, "bin")

DEFAULT_DPI = 300

_MISSING = object()

# (latex, is_block, dpi) -> (w, h, png) | None
_CACHE: dict[tuple[str, bool, int], tuple[int, int, bytes] | None] = {}
_CACHE_LOCK = threading.Lock()

_tectonic_path: object = _MISSING

_TARGETS = {
    ("linux", "x86_64"): "x86_64-unknown-linux-musl",
    ("linux", "amd64"): "x86_64-unknown-linux-musl",
    ("linux", "aarch64"): "aarch64-unknown-linux-musl",
    ("linux", "arm64"): "aarch64-unknown-linux-musl",
    ("darwin", "arm64"): "aarch64-apple-darwin",
    ("darwin", "x86_64"): "x86_64-apple-darwin",
}


# --------------------------------------------------------------------------
# tectonic bootstrap
# --------------------------------------------------------------------------

def _download_tectonic() -> str | None:
    """Download the tectonic binary into ``bin/`` once; return its path."""
    machine = platform.machine().lower()
    target = _TARGETS.get((sys.platform, machine))
    if not target:
        print(f"[LaTeX] no tectonic build for {sys.platform}/{machine}")
        return None

    url = (
        "https://github.com/tectonic-typesetting/tectonic/releases/download/"
        f"tectonic%40{_TECTONIC_VERSION}/"
        f"tectonic-{_TECTONIC_VERSION}-{target}.tar.gz"
    )
    dest = os.path.join(_BIN_DIR, "tectonic")
    try:
        os.makedirs(_BIN_DIR, exist_ok=True)
        print(f"[LaTeX] downloading tectonic {_TECTONIC_VERSION} ({target}) ...")
        with urllib.request.urlopen(url, timeout=180) as response:
            archive = response.read()
        with tarfile.open(fileobj=BytesIO(archive), mode="r:gz") as tar:
            member = next(
                (m for m in tar.getmembers()
                 if m.isfile() and os.path.basename(m.name) == "tectonic"),
                None,
            )
            if member is None:
                print("[LaTeX] tectonic archive has no tectonic binary")
                return None
            handle = tar.extractfile(member)
            if handle is None:
                return None
            with open(dest, "wb") as fh:
                shutil.copyfileobj(handle, fh)
        os.chmod(dest, 0o755)
    except Exception as exc:  # noqa: BLE001 - network failures are expected
        print(f"[LaTeX] tectonic download failed: {exc}")
        return None

    return dest


def _ensure_tectonic() -> str | None:
    """Return a usable tectonic binary path, or ``None``."""
    global _tectonic_path
    with _CACHE_LOCK:
        if _tectonic_path is not _MISSING:
            return _tectonic_path  # type: ignore[return-value]
        candidates = [
            os.environ.get("TECTONIC_BIN", ""),
            os.path.join(_BIN_DIR, "tectonic"),
            shutil.which("tectonic") or "",
        ]
        found = next(
            (c for c in candidates
             if c and os.path.isfile(c) and os.access(c, os.X_OK)),
            None,
        )
        if found is None:
            found = _download_tectonic()
        _tectonic_path = found
        return found


# --------------------------------------------------------------------------
# LaTeX source preparation
# --------------------------------------------------------------------------

_TAG_RE = re.compile(r"\\(?:tag|label|nonumber|notag)\*?\{[^{}]*\}")
_PERCENT_RE = re.compile(r"(?<!\\)%")
_USEPACKAGE_RE = re.compile(r"\\(?:usepackage|documentclass)(?:\[[^\]]*\])?\{[^{}]*\}")


def _sanitize(latex: str) -> str:
    """Make an LLM-authored formula safe to drop into a standalone document."""
    text = latex.strip()
    # Packages / preamble fragments must never reach our document body.
    text = _USEPACKAGE_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    # A bare % starts a TeX comment and would swallow the rest of the line.
    text = _PERCENT_RE.sub(r"\\%", text)
    return text.strip()


def _build_tex(equations: list[tuple[str, bool]]) -> str:
    lines = [
        r"\documentclass[multi={eqn},border=2pt]{standalone}",
        r"\usepackage{amsmath}",
        r"\usepackage{amssymb}",
        r"\usepackage{bm}",
        r"\begin{document}",
    ]
    for latex, is_block in equations:
        body = _sanitize(latex)
        if is_block:
            # Display formulas get a larger base size so they do not look
            # miniature next to 15px body text in the e-mail.
            body = r"{\Large $\displaystyle " + body + r"$}"
        else:
            body = "$" + body + "$"
        lines.append(r"\begin{eqn}" + body + r"\end{eqn}")
    lines.append(r"\end{document}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Rendering backends
# --------------------------------------------------------------------------

def _trim(image: Image.Image, pad: int = 2) -> Image.Image:
    """Crop the white margins LaTeX leaves around the formula."""
    grey = image.convert("L")
    bbox = grey.point(lambda v: 255 - v).getbbox()
    if not bbox:
        return image
    left, top, right, bottom = bbox
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(image.width, right + pad)
    bottom = min(image.height, bottom + pad)
    return image.crop((left, top, right, bottom))


def _page_to_png(page, dpi: int) -> tuple[int, int, bytes]:
    """Rasterise one PDF page, trim it, and return CSS-pixel dimensions."""
    pixmap = page.get_pixmap(dpi=dpi, alpha=False)
    with Image.open(BytesIO(pixmap.tobytes("png"))) as raw:
        image = _trim(raw.convert("RGB"))
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    scale = 96.0 / dpi
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    return width, height, buffer.getvalue()


def _compile(tex_path: str, out_dir: str) -> str | None:
    """Run tectonic on ``tex_path``; return the produced PDF path or None."""
    binary = _ensure_tectonic()
    if binary is None:
        return None
    try:
        proc = subprocess.run(
            [binary, "-X", "compile", tex_path, "--outdir", out_dir],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[LaTeX] tectonic invocation failed: {exc}")
        return None

    pdf = os.path.join(out_dir, os.path.splitext(os.path.basename(tex_path))[0] + ".pdf")
    if proc.returncode != 0 or not os.path.isfile(pdf):
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        print(f"[LaTeX] tectonic failed: {detail[-1] if detail else 'unknown error'}")
        return None
    return pdf


def _render_with_tectonic(keys: list[tuple[str, bool, int]]) -> dict:
    """Render all formulas in a single tectonic run (one PDF page each)."""
    dpi = keys[0][2]
    work_dir = tempfile.mkdtemp(prefix="icourse-latex-")
    tex_path = os.path.join(work_dir, "formulas.tex")
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write(_build_tex([(latex, is_block) for latex, is_block, _ in keys]))

    results: dict = {}
    pdf = _compile(tex_path, work_dir)
    if pdf:
        try:
            import pymupdf  # noqa: PLC0415 - optional dependency
        except ImportError:  # pragma: no cover - requirements pin it
            import fitz as pymupdf  # type: ignore[no-redef]
        try:
            with pymupdf.open(pdf) as doc:
                if doc.page_count == len(keys):
                    for key, page in zip(keys, doc):
                        results[key] = _page_to_png(page, dpi)
                else:
                    print(f"[LaTeX] page count {doc.page_count} != "
                          f"{len(keys)} formulas; falling back one by one")
        except Exception as exc:  # noqa: BLE001
            print(f"[LaTeX] PDF rasterisation failed: {exc}")

    # A single bad formula aborts the whole batch — retry the stragglers
    # individually so one typo cannot wipe out the entire summary.
    for key in keys:
        if key not in results:
            single = _compile_single(key)
            results[key] = single
    return results


def _compile_single(key: tuple[str, bool, int]) -> tuple[int, int, bytes] | None:
    latex, is_block, dpi = key
    work_dir = tempfile.mkdtemp(prefix="icourse-latex-one-")
    tex_path = os.path.join(work_dir, "formula.tex")
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write(_build_tex([(latex, is_block)]))
    pdf = _compile(tex_path, work_dir)
    if not pdf:
        return None
    try:
        import pymupdf  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        import fitz as pymupdf  # type: ignore[no-redef]
    try:
        with pymupdf.open(pdf) as doc:
            if doc.page_count < 1:
                return None
            return _page_to_png(doc[0], dpi)
    except Exception as exc:  # noqa: BLE001
        print(f"[LaTeX] single-formula rasterisation failed: {exc}")
        return None


_MATH_TEXT_RE = re.compile(r"\\(?:text|operatorname|mathrm)\{([^{}]*)\}")


def _mathtext_compatible(latex: str) -> str:
    """Best-effort rewrite of LaTeX into matplotlib's mathtext subset."""
    text = latex
    text = _MATH_TEXT_RE.sub(lambda m: r"\mathrm{" + m.group(1) + r"}", text)
    # mathtext understands \leq but not the \le alias.
    text = re.sub(r"\\le(?![a-zA-Z])", r"\\leq ", text)
    text = re.sub(r"\\ge(?![a-zA-Z])", r"\\geq ", text)
    text = text.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    return text


def _render_with_mathtext(key: tuple[str, bool, int]) -> tuple[int, int, bytes] | None:
    """Fallback renderer: matplotlib's built-in mathtext engine."""
    latex, is_block, dpi = key
    try:
        import matplotlib  # noqa: PLC0415 - optional dependency
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        print(f"[LaTeX] matplotlib unavailable, using plain text: {exc}")
        return None

    body = _mathtext_compatible(_sanitize(latex))
    try:
        with matplotlib.rc_context({"mathtext.fontset": "cm"}):
            figure = plt.figure(figsize=(0.01, 0.01), dpi=dpi)
            figure.patch.set_facecolor("white")
            figure.text(0, 0, f"${body}$", fontsize=15, color="black")
            buffer = BytesIO()
            figure.savefig(
                buffer,
                format="png",
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=0.02,
                facecolor="white",
            )
            plt.close(figure)
    except Exception as exc:  # noqa: BLE001
        print(f"[LaTeX] mathtext could not parse {latex[:60]!r}: {exc}")
        return None

    buffer.seek(0)
    with Image.open(buffer) as image:
        rgb = image.convert("RGB")
        buffer = BytesIO()
        rgb.save(buffer, format="PNG", optimize=True)
        scale = 96.0 / dpi
        width = max(1, round(rgb.width * scale))
        height = max(1, round(rgb.height * scale))
    return width, height, buffer.getvalue()


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def render_batch(
    equations: list[tuple[str, bool]],
    dpi: int = DEFAULT_DPI,
) -> list[tuple[int, int, bytes] | None]:
    """Render ``(latex, is_block)`` pairs to ``(w, h, png_bytes)``.

    Results are memoised per formula, so re-sending a summary costs nothing.
    Entries that no backend can render come back as ``None``; the caller is
    expected to fall back to a plain-text representation.
    """
    keys = [(latex, is_block, dpi) for latex, is_block in equations]
    missing = [key for key in keys if key not in _CACHE]

    if missing:
        rendered = _render_with_tectonic(missing)
        for key in missing:
            value = rendered.get(key)
            if value is None:
                value = _render_with_mathtext(key)
            _CACHE[key] = value

    return [_CACHE[key] for key in keys]
