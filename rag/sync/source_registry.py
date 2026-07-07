"""Public 3GPP file-server source registry."""

from __future__ import annotations

from dataclasses import dataclass


PUBLIC_FILE_SERVER_ROOT = "https://www.3gpp.org/ftp"
USER_AGENT = "local-sa3-agentic-rag/0.1 (+https://www.3gpp.org/ftp; polite local research sync)"
DEFAULT_SYNC_INTERVAL_HOURS = 6


@dataclass(frozen=True)
class SpecSource:
    spec_number: str
    directory_url: str


@dataclass(frozen=True)
class MeetingListSource:
    name: str
    directory_url: str


SPEC_SOURCES = [
    SpecSource(
        spec_number="TS 33.501",
        directory_url=f"{PUBLIC_FILE_SERVER_ROOT}/Specs/archive/33_series/33.501/",
    ),
    SpecSource(
        spec_number="TS 33.210",
        directory_url=f"{PUBLIC_FILE_SERVER_ROOT}/Specs/archive/33_series/33.210/",
    ),
    SpecSource(
        spec_number="TS 33.310",
        directory_url=f"{PUBLIC_FILE_SERVER_ROOT}/Specs/archive/33_series/33.310/",
    ),
]

MEETING_LIST_SOURCES = [
    MeetingListSource(
        name="SA3",
        directory_url=f"{PUBLIC_FILE_SERVER_ROOT}/tsg_sa/WG3_Security/",
    )
]
