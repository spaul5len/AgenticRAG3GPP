#!/usr/bin/env python
"""Run all configured 3GPP public sync jobs once."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.sync.scheduler import sync_all


def main() -> int:
    result = sync_all()
    print(f"Synced {len(result['specs'])} spec archive entries.")
    print(f"Synced {len(result['meeting_lists'])} meeting list entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
