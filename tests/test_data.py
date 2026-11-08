import pytest

from trustedtext.bench.data import AI, HUMAN, clean_text, group_of, load_corpus


def test_clean_text_removes_source_artifacts():
    raw = "## Heading\nSome **bold** text\u200b with a [link](https://example.com).\n"
    assert clean_text(raw) == "Heading\nSome bold text with a link."


def test_clean_text_keeps_ordinary_punctuation():
    text = "Prices rose 5% — a record… “quoted” and 2*3"
    assert clean_text(text) == text


def test_group_of_parses_filename_prefix():
    assert group_of("claude003") == "claude"
    assert group_of("Reuters12") == "reuters"
    assert group_of("123") == "unknown"


def test_load_corpus(tmp_path):
    (tmp_path / "human").mkdir()
    (tmp_path / "ai").mkdir()
    (tmp_path / "human" / "ap001.txt").write_text("A human article.")
    (tmp_path / "human" / "ap002.txt").write_text("   ")
    (tmp_path / "ai" / "gpt001.txt").write_text("An **AI** article.")

    docs = load_corpus(tmp_path)
    assert [(d.label, d.group, d.text) for d in docs] == [
        (HUMAN, "ap", "A human article."),
        (AI, "gpt", "An AI article."),
    ]
    assert docs[0].doc_id.endswith("human/ap001")


def test_load_corpus_requires_both_classes(tmp_path):
    (tmp_path / "human").mkdir()
    (tmp_path / "ai").mkdir()
    (tmp_path / "human" / "ap001.txt").write_text("text")
    with pytest.raises(ValueError):
        load_corpus(tmp_path)
