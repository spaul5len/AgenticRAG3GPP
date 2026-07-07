import pytest

from rag.chunking import chunk_pages, chunk_text
from rag.parsers import parse_document, parse_text_file


def test_chunk_text_splits_words_with_overlap():
    chunks = chunk_text("one two three four five six", chunk_size=3, overlap=1)

    assert chunks == [
        {
            "chunk_id": "chunk-0",
            "text": "one two three",
            "start_word": 0,
            "end_word": 3,
        },
        {
            "chunk_id": "chunk-1",
            "text": "three four five",
            "start_word": 2,
            "end_word": 5,
        },
        {
            "chunk_id": "chunk-2",
            "text": "five six",
            "start_word": 4,
            "end_word": 6,
        },
    ]


def test_chunk_text_skips_empty_text():
    assert chunk_text("   \n\t", chunk_size=5, overlap=1) == []


def test_chunk_pages_preserves_page_number_and_skips_empty_pages():
    pages = [
        {"text": "alpha beta gamma delta", "page": 4},
        {"text": "   ", "page": 5},
        {"text": "epsilon zeta", "page": 6},
    ]

    chunks = chunk_pages(pages, chunk_size=2, overlap=0)

    assert [chunk["chunk_id"] for chunk in chunks] == [
        "chunk-0",
        "chunk-1",
        "chunk-2",
    ]
    assert [chunk["text"] for chunk in chunks] == [
        "alpha beta",
        "gamma delta",
        "epsilon zeta",
    ]
    assert [chunk["page"] for chunk in chunks] == [4, 4, 6]


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (10, -1), (10, 10), (10, 11)],
)
def test_invalid_chunk_settings_raise_errors(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=chunk_size, overlap=overlap)


def test_parse_text_file_returns_single_page_and_skips_empty_file(tmp_path):
    text_file = tmp_path / "sample.txt"
    text_file.write_text("hello\nworld", encoding="utf-8")
    empty_file = tmp_path / "empty.md"
    empty_file.write_text("   ", encoding="utf-8")

    assert parse_text_file(text_file) == [{"text": "hello\nworld", "page": 1}]
    assert parse_text_file(empty_file) == []


def test_parse_document_supports_txt_and_md_dispatch(tmp_path):
    text_file = tmp_path / "sample.txt"
    markdown_file = tmp_path / "sample.md"
    text_file.write_text("plain text", encoding="utf-8")
    markdown_file.write_text("# title", encoding="utf-8")

    assert parse_document(text_file) == [{"text": "plain text", "page": 1}]
    assert parse_document(markdown_file) == [{"text": "# title", "page": 1}]


def test_parse_document_rejects_unsupported_extension(tmp_path):
    unsupported = tmp_path / "sample.csv"
    unsupported.write_text("a,b", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported document type"):
        parse_document(unsupported)
