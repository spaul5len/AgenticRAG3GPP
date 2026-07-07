#!/usr/bin/env python
"""Sync selected 3GPP specification archives."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.sync.spec_sync import sync_specs


def main() -> int:
    synced = sync_specs()
    print(f"Synced {len(synced)} spec archive entries.")
    for item in synced:
        print(
            f"{item.get('spec_number')} {item.get('version')} "
            f"downloaded={item.get('downloaded')} {item.get('local_path')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
