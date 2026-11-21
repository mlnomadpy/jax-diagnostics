"""Route a JAX error name, traceback, or symptom to a repair guide."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from . import DEFAULT_GUIDE_BASE, diagnose


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", help="error or symptom, with stdin used when omitted")
    parser.add_argument("--json", action="store_true", help="emit structured output")
    args = parser.parse_args()
    if args.text is None and sys.stdin.isatty():
        parser.error("pass an error or pipe a traceback on stdin")
    result = diagnose(args.text if args.text is not None else sys.stdin.read())
    if result is None:
        if args.json:
            print(json.dumps({"match": None}))
        else:
            print("No diagnostic matched. Check the final exception and the failing operation.")
        return 1
    if args.json:
        print(json.dumps({**asdict(result), "guide": result.link}))
    else:
        print(f"[{result.id}] {result.title}")
        checkout = Path(__file__).resolve().parent.parent
        local_guide = checkout / result.link.split("#", 1)[0]
        location = checkout.as_uri() if local_guide.is_file() else DEFAULT_GUIDE_BASE
        print(f"Agent repair guide: {location}/{result.link}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
