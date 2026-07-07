"""Polite background scheduler for 3GPP public sync jobs."""

from __future__ import annotations

from apscheduler.schedulers.background import BackgroundScheduler

from rag.sync.meeting_sync import sync_meeting_lists
from rag.sync.source_registry import DEFAULT_SYNC_INTERVAL_HOURS
from rag.sync.spec_sync import sync_specs


def sync_all() -> dict[str, object]:
    """Run all configured sync jobs once."""

    return {
        "specs": sync_specs(),
        "meeting_lists": sync_meeting_lists(),
    }


def create_scheduler(interval_hours: int = DEFAULT_SYNC_INTERVAL_HOURS) -> BackgroundScheduler:
    """Create a scheduler with a conservative default six-hour interval."""

    if interval_hours <= 0:
        raise ValueError("interval_hours must be greater than 0.")
    scheduler = BackgroundScheduler()
    scheduler.add_job(sync_all, "interval", hours=interval_hours, id="sync_all_3gpp")
    return scheduler
