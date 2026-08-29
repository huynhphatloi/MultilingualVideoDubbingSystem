"""The fixed test set. Same lines for every engine, or the numbers mean nothing.

Two design decisions worth defending in the write-up:

**Ordered by length, shortest first.** The viXTTS model card states plainly:
*"Subpar performance for input sentences under 10 words in Vietnamese (yielding
inconsistent output and odd trailing sounds)."* Film subtitles are mostly under
10 words. So the length axis is not a nice-to-have dimension of the benchmark -
it is the primary hypothesis being tested. An engine that only works on long
sentences does not work on this pipeline.

**Real dialogue, not read-aloud prose.** Benchmarks built from Wikipedia
sentences flatter every model, because that is what they were trained on. These
are the shapes that actually appear in subtitles: interjections, questions,
imperatives, interrupted clauses, and names.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Line:
    text: str
    #: Word count, precomputed so the report can bucket without re-splitting.
    words: int
    #: What this line is here to catch.
    probe: str


def _line(text: str, probe: str) -> Line:
    return Line(text=text, words=len(text.split()), probe=probe)


#: Vietnamese - the target language of this project, so the deepest set.
VIETNAMESE: tuple[Line, ...] = (
    _line("Không.", "1 word: the worst case, and it happens constantly"),
    _line("Đi thôi.", "2 words: imperative, listen for a trailing artefact"),
    _line("Anh nói gì cơ?", "4 words: question intonation on a short line"),
    _line("Cẩn thận! Phía sau anh!", "5 words: two clauses, exclamation"),
    _line("Tôi không nghĩ đó là ý hay.", "7 words: the boundary of the model card's warning"),
    _line("Chúng ta không còn nhiều thời gian đâu.", "7 words: flat declarative"),
    _line("Thưa ngài, đội hai đã vào vị trí rồi ạ.",
          "9 words: honorifics and sentence-final particles"),
    _line("Nghe đây, cho đến khi chúng ta đóng được cánh cổng đó, "
          "không ai được rời khỏi vị trí.",
          "~20 words: subordinate clause, the length these models were tuned for"),
    _line("Tôi đã nói với anh rồi, nếu chúng ta không hành động ngay bây giờ "
          "thì đến sáng mai sẽ chẳng còn gì để cứu nữa cả.",
          "~28 words: long form, where the cloning models should look best"),
    _line("Mã truy cập là 4 7 2 9, nhắc lại, 4 7 2 9.",
          "digits: read as numbers or as digits? subtitles are full of these"),
    _line("Anh Minh, chị Lan và bác sĩ Hoà đang đợi ở tầng ba.",
          "proper nouns: names are where g2p usually breaks"),
    _line("Nó… nó không thể nào như thế được.",
          "ellipsis and repetition: hesitation, extremely common in dialogue"),
)

#: English - the source language of the test clips, and the only language every
#: engine here can speak. Used for the cross-engine sanity column.
ENGLISH: tuple[Line, ...] = (
    _line("No.", "1 word"),
    _line("Let's go.", "2 words"),
    _line("What did you say?", "4 words"),
    _line("We don't have much time left.", "6 words"),
    _line("Listen to me, until we close that portal nobody leaves their post.",
          "12 words"),
    _line("I already told you, if we don't move right now there will be "
          "nothing left to save by morning.", "19 words"),
)

BY_LANGUAGE: dict[str, tuple[Line, ...]] = {"vi": VIETNAMESE, "en": ENGLISH}


def get(language: str, *, max_lines: int | None = None) -> tuple[Line, ...]:
    lines = BY_LANGUAGE.get((language or "").lower().split("-")[0], ENGLISH)
    return lines[:max_lines] if max_lines else lines


#: Length buckets for the summary table. The first two are the ones that decide
#: whether an engine is usable for subtitles at all.
BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("very short (1-3 w)", 1, 3),
    ("short (4-9 w)", 4, 9),
    ("medium (10-19 w)", 10, 19),
    ("long (20+ w)", 20, 10_000),
)


def bucket_of(words: int) -> str:
    for label, low, high in BUCKETS:
        if low <= words <= high:
            return label
    return BUCKETS[-1][0]
