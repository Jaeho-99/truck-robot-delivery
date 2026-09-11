"""Build the cumulative per-size CSV summaries for the v2_claude subtree.

Same tables, same columns and the same merge semantics as
``scripts/summarize_results.py`` -- this reads ``output/v2_claude/`` and
``models/v2_claude/`` and writes ``comparison_n{size}.csv`` and
``training_time_n{size}.csv`` next to those results, under
``output/v2_claude/``.

No logic is duplicated: the original summarizer is imported and its two
path globals are repointed at the variant subtree. ``scripts/
summarize_results.py`` is not modified and keeps summarizing the v1
results exactly as before.

Run from the repository root::

    python scripts/summarize_results_v2_claude.py
    python scripts/summarize_results_v2_claude.py --size 20

Every option of the original script is accepted and forwarded unchanged.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
VARIANT = "v2_claude"
ORIGINAL = Path(__file__).with_name("summarize_results.py")


def _load_summarizer():
    """Import scripts/summarize_results.py without touching the file."""
    spec = importlib.util.spec_from_file_location(
        "summarize_results_original", ORIGINAL)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {ORIGINAL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    summarizer = _load_summarizer()
    output_dir = REPO_ROOT / "output" / VARIANT
    models_dir = REPO_ROOT / "models" / VARIANT
    if not output_dir.is_dir():
        raise FileNotFoundError(
            f"no {VARIANT} results under {output_dir}; run the "
            f"src/{VARIANT} experiments first")
    # OUTPUT_DIR and MODELS_DIR are the only path globals the summarizer
    # reads (discovery, checkpoint lookup and table destination).
    summarizer.OUTPUT_DIR = output_dir
    summarizer.MODELS_DIR = models_dir
    print(f"[{VARIANT}] results={output_dir}  checkpoints={models_dir}",
          flush=True)
    summarizer.main()


if __name__ == "__main__":
    main()
