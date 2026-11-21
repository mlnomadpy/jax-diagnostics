"""Run the diagnostic CLI directly from a repository checkout."""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from jax_diagnostics.cli import main

    sys.exit(main())
