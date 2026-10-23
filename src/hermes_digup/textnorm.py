"""Query text handling, ported from DigUp's SearchText.swift and Searcher.swift.

Keyword search has to see a query the way DigUp stored its text, or exact matches
(invoice numbers, names, error codes) silently stop working. So the normalisation,
stopwords and kind words here mirror DigUp's own.
"""

from __future__ import annotations

import re
import unicodedata

# Zero-width space is a word break; joiners, direction marks, soft hyphen and BOM go.
_DROP = (
    {0x00AD, 0x061C, 0xFEFF}
    | set(range(0x200C, 0x2010))
    | set(range(0x202A, 0x202F))
    | set(range(0x2060, 0x2065))
    | set(range(0x2066, 0x206A))
    # Arabic tatweel, harakat and the dagger alef.
    | {0x0640, 0x0670}
    | set(range(0x064B, 0x0660))
)
_ALEF = {0x0622, 0x0623, 0x0625, 0x0671}

_INDIC = range(0x0900, 0x0E00)
_ARABIC = range(0x0600, 0x0700)


def normalized(text: str) -> str:
    """NFKC, invisible formatting dropped, Arabic the way people type it (SearchText.normalized)."""
    out = []
    for ch in unicodedata.normalize("NFKC", text):
        cp = ord(ch)
        if cp == 0x200B:
            out.append(" ")
        elif cp in _DROP:
            continue
        elif cp in _ALEF:
            out.append("ا")
        else:
            out.append(ch)
    return "".join(out)


def inflects(word: str) -> bool:
    """Scripts that attach endings (Bengali, Hindi, Arabic) are matched as prefixes."""
    return any(ord(c) in _INDIC or ord(c) in _ARABIC for c in word)


STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "with", "and", "or", "my", "me", "that", "this",
    "these", "those", "from", "about", "where", "what", "which", "who", "when", "is", "was", "are", "were", "be",
    "it", "its", "some", "any", "find", "show", "file", "files", "photo", "photos", "picture", "image", "images",
    "screenshot", "screenshots", "video", "videos", "pdf", "document", "documents", "recording",
    "এবং", "ও", "এই", "সেই", "একটি", "আমার", "থেকে", "জন্য", "সাথে", "মধ্যে", "কোথায়", "কোন", "কী", "কি", "যে",
    "ছবি", "স্ক্রিনশট", "ভিডিও", "ফাইল", "পিডিএফ",
    "في", "من", "على", "الى", "عن", "مع", "هذا", "هذه", "ذلك", "التي", "الذي", "اين", "ما",
    "صورة", "صور", "لقطة", "شاشة", "فيديو", "ملف", "مستند",
}

KIND_WORDS = {
    "screenshot": "screenshot", "screenshots": "screenshot", "photo": "image", "photos": "image",
    "picture": "image", "pictures": "image", "image": "image", "images": "image", "pdf": "pdf", "pdfs": "pdf",
    "paper": "pdf", "papers": "pdf", "document": "doc", "documents": "doc", "note": "doc", "notes": "doc",
    "video": "video", "videos": "video", "clip": "video", "movie": "video", "audio": "audio", "song": "audio",
    "podcast": "audio", "voice": "audio", "sound": "audio",
    "স্ক্রিনশট": "screenshot", "ছবি": "image", "পিডিএফ": "pdf", "নোট": "doc", "ভিডিও": "video", "অডিও": "audio",
    "গান": "audio", "রেকর্ডিং": "audio",
    "لقطة": "screenshot", "صورة": "image", "صور": "image", "مستند": "doc", "ملاحظات": "doc", "فيديو": "video",
    "مقطع": "video", "صوت": "audio", "تسجيل": "audio", "اغنية": "audio",
}


def _is_alnum(ch: str) -> bool:
    # Foundation's CharacterSet.alphanumerics: letters, marks and numbers.
    return unicodedata.category(ch)[0] in "LMN"


def _words(text: str) -> list[str]:
    words, current = [], []
    for ch in normalized(text).lower():
        if _is_alnum(ch):
            current.append(ch)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return words


def terms(text: str) -> list[str]:
    """A query's content words: normalised, two letters or more, no stopwords, each once."""
    seen: set[str] = set()
    out = []
    for word in _words(text):
        if len(word) >= 2 and word not in STOPWORDS and word not in seen:
            seen.add(word)
            out.append(word)
    return out


def patterns(words: list[str]) -> list[str]:
    """FTS5 phrase patterns; inflecting scripts match as prefixes."""
    return [f'"{w}"' + ("*" if inflects(w) else "") for w in words]


def is_lookup(words: list[str]) -> bool:
    """Codes, IDs and long numbers: keyword search should outweigh meaning."""
    return bool(words) and all(
        any(c.isdigit() for c in w) and (any(c.isalpha() for c in w) or len(w) >= 4) for w in words
    )


def kind_hints(text: str) -> set[str]:
    """File kinds the query asks for ("... in a video"): a soft boost, never a filter."""
    hints = set()
    for word in _words(text):
        kind = KIND_WORDS.get(word)
        if kind is None:
            for key, value in KIND_WORDS.items():
                if inflects(key) and len(key) >= 3 and word.startswith(key):
                    kind = value
                    break
        if kind:
            hints.add(kind)
    return hints


_SPACE = re.compile(r"\s+")


def one_line(text: str, limit: int = 300) -> str:
    line = _SPACE.sub(" ", text[: limit * 8]).strip()
    return line[:limit] + "…" if len(line) > limit or len(text) > limit * 8 else line


def snippet(text: str, words: list[str], context: int = 12) -> str | None:
    """A line around the first query word in `text`, matches marked « », or None."""
    if not words:
        return None
    pattern = re.compile(
        "|".join(
            (r"(?<![\w])" + re.escape(w) + (r"\w*" if inflects(w) else r"(?![\w])"))
            for w in sorted(words, key=len, reverse=True)
        ),
        re.IGNORECASE,
    )
    haystack = normalized(text)
    first = pattern.search(haystack)
    if not first:
        return None
    before = haystack[: first.start()].split()
    after = haystack[first.start():].split()
    start_words = before[-(context // 3):] if context // 3 else []
    piece = " ".join(start_words + after[: context - len(start_words)])
    marked = pattern.sub(lambda m: f"«{m.group(0)}»", piece)
    prefix = "…" if len(before) > len(start_words) else ""
    suffix = "…" if len(after) > context - len(start_words) else ""
    return prefix + marked + suffix
