#!/usr/bin/env python
"""Sync public 3GPP SA3 meeting directory listings."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.sync.meeting_sync import sync_meeting_lists


def main() -> int:
    synced = sync_meeting_lists()
    print(f"Synced {len(synced)} meeting list entries.")
    for item in synced:
        print(f"{item.get('meeting_list_name')} {item.get('local_path')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
