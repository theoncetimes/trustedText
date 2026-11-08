"""Corpus loading with per-document metadata."""

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List

HUMAN, AI = 0, 1

_ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_EMPHASIS = re.compile(r"(\*\*|__)(.+?)\1", re.S)
_MD_HEADING = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+", re.M)
_GROUP_PREFIX = re.compile(r"^[A-Za-z]+")


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    label: int
    # Generator for AI text, outlet for human text; parsed from the filename
    # prefix, e.g. "claude003.txt" -> "claude", "reuters001.txt" -> "reuters"
    group: str

    @property
    def sha(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def clean_text(text: str) -> str:
    """
    Remove copy-paste and rendering artifacts that identify the source
    (zero-width characters, markdown) rather than the writing itself.
    """
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH.sub("", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_EMPHASIS.sub(r"\2", text)
    text = _MD_HEADING.sub("", text)
    return text.strip()


def group_of(stem: str) -> str:
    match = _GROUP_PREFIX.match(stem)
    return match.group(0).lower() if match else "unknown"


def load_corpus(root: Path, clean: bool = True) -> List[Document]:
    """Load root/human/*.txt and root/ai/*.txt, one document per file."""
    root = Path(root)
    docs: List[Document] = []
    for label_name, label in (("human", HUMAN), ("ai", AI)):
        for path in sorted((root / label_name).glob("*.txt")):
            text = path.read_text(encoding="utf-8")
            text = clean_text(text) if clean else text.strip()
            if text:
                docs.append(
                    Document(
                        doc_id=f"{root.name}/{label_name}/{path.stem}",
                        text=text,
                        label=label,
                        group=group_of(path.stem),
                    )
                )

    labels = {d.label for d in docs}
    if labels != {HUMAN, AI}:
        raise ValueError(f"{root} must contain non-empty human/ and ai/ folders")
    return docs
