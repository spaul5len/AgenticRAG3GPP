"""Helpers for fetching and parsing public HTTP directory listings."""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import requests

from rag.sync.source_registry import USER_AGENT


@dataclass(frozen=True)
class DirectoryLink:
    text: str
    href: str
    url: str
    is_directory: bool


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._active_href: str | None = None
        self._active_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attrs_dict = dict(attrs)
        href = attrs_dict.get("href")
        if href:
            self._active_href = href
            self._active_text = []

    def handle_data(self, data: str) -> None:
        if self._active_href is not None:
            self._active_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or self._active_href is None:
            return
        self.links.append((self._active_href, "".join(self._active_text).strip()))
        self._active_href = None
        self._active_text = []


def fetch_directory_listing(url: str, timeout: int = 30) -> str:
    """Fetch one HTTPS directory listing with a clear user-agent."""

    _ensure_https(url)
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.text


def parse_directory_links(html: str, base_url: str) -> list[DirectoryLink]:
    """Parse links from an HTML directory listing."""

    _ensure_https(base_url)
    parser = _LinkParser()
    parser.feed(html)

    links: list[DirectoryLink] = []
    for href, text in parser.links:
        if not href or href.startswith(("#", "?", "mailto:", "javascript:")):
            continue
        if href in {"../", ".."} or text.lower() in {
            "parent directory",
            "[to parent directory]",
        }:
            continue
        absolute = urljoin(base_url, href)
        if urlparse(absolute).scheme != "https":
            continue
        links.append(
            DirectoryLink(
                text=text or href,
                href=href,
                url=absolute,
                is_directory=href.endswith("/"),
            )
        )
    return links


def _ensure_https(url: str) -> None:
    if urlparse(url).scheme != "https":
        raise ValueError(f"Only HTTPS 3GPP URLs are supported: {url}")
