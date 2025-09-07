"""Thin wrapper to launch training via Hydra from scripts/.

Ensures the repository's `src/` is on `sys.path` so we can import
`trainer.launch` without installing the package.
"""
from __future__ import annotations

from dotenv import load_dotenv; load_dotenv()

import sys
from pathlib import Path


# Make `src/` importable when running this script directly
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trainer.launch import main  # noqa: E402


if __name__ == "__main__":
    main()
