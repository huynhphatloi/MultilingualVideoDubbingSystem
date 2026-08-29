"""Glossary round trip: protect -> translate -> restore."""
from __future__ import annotations

import json

import pytest
from app.services.translation import glossary


@pytest.fixture()
def terms():
    return {"Captain": "Captain", "Cap": "Cap", "Apple TV": "Apple TV",
            "Smash": "Smash", "Fury": "Đại tướng Fury"}


def test_protect_replaces_only_listed_terms(terms):
    masked, mapping = glossary.protect("Call it, Captain. And Hulk, Smash.", terms)
    assert "Captain" not in masked and "Smash" not in masked
    assert "Hulk" in masked                      # not listed -> left for the model
    assert set(mapping.values()) == {"Captain", "Smash"}


def test_longest_term_wins(terms):
    """'Apple TV' must not be shredded by a shorter overlapping entry."""
    masked, mapping = glossary.protect("free on Apple TV", terms)
    assert list(mapping.values()) == ["Apple TV"]
    assert "Apple" not in masked


def test_matching_is_whole_word_and_case_sensitive(terms):
    masked, mapping = glossary.protect("a stark, uncapped capital", terms)
    assert mapping == {}
    assert masked == "a stark, uncapped capital"


def test_cap_inside_capital_is_not_matched(terms):
    _, mapping = glossary.protect("Capital letters", terms)
    assert mapping == {}


def test_restore_puts_the_term_back(terms):
    masked, mapping = glossary.protect("Call it, Captain.", terms)
    assert glossary.restore(masked.replace("Call it,", "Cứ gọi đi,"), mapping) \
        == "Cứ gọi đi, Captain."


def test_restore_tolerates_a_stray_delimiter(terms):
    """NLLB really does emit '@@1@@@' - observed, not hypothetical."""
    _, mapping = glossary.protect("And Hulk, Smash.", terms)
    assert glossary.restore("Và Hulk, @@0@@@.", mapping) == "Và Hulk, Smash."


def test_restore_can_force_a_specific_rendering(terms):
    _, mapping = glossary.protect("Fury said so.", terms)
    assert "Đại tướng Fury" in glossary.restore("@@0@@ đã nói vậy.", mapping)


def test_a_dropped_marker_is_not_bolted_back_on(terms):
    """The output is spoken; a mangled sentence is worse than a missing name."""
    _, mapping = glossary.protect("Call it, Captain.", terms)
    assert glossary.restore("Cứ gọi đi.", mapping) == "Cứ gọi đi."


def test_no_terms_is_a_no_op():
    assert glossary.protect("anything", {}) == ("anything", {})
    assert glossary.restore("anything", {}) == "anything"


# --------------------------------------------------------------- file loading --
def test_missing_file_means_no_protection(tmp_path):
    assert glossary.terms_for("en", str(tmp_path / "nope.json")) == {}


def test_star_entries_apply_to_every_language(tmp_path):
    path = tmp_path / "g.json"
    path.write_text(json.dumps({"*": {"CS50": "CS50"}, "en": {"Captain": "Captain"}}),
                    encoding="utf-8")
    assert glossary.terms_for("en", str(path)) == {"CS50": "CS50", "Captain": "Captain"}
    assert glossary.terms_for("fr", str(path)) == {"CS50": "CS50"}


def test_a_broken_file_does_not_break_the_job(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    assert glossary.terms_for("en", str(path)) == {}


def test_comment_keys_are_ignored(tmp_path):
    path = tmp_path / "g.json"
    path.write_text(json.dumps({"_comment": ["notes"], "en": {"Cap": "Cap"}}),
                    encoding="utf-8")
    assert glossary.terms_for("en", str(path)) == {"Cap": "Cap"}


def test_the_shipped_glossary_parses():
    assert isinstance(glossary.terms_for("en"), dict)


# ------------------------------------------------------------------ casing --
def test_restore_recapitalises_the_opening_word(terms):
    """Masking the first word makes NLLB start the sentence lowercase."""
    _, mapping = glossary.protect("Call it, Captain.", terms)
    assert glossary.restore("gọi nó, @@0@@.", mapping) == "Gọi nó, Captain."


def test_recase_leaves_an_already_capitalised_line_alone(terms):
    _, mapping = glossary.protect("Call it, Captain.", terms)
    assert glossary.restore("Cứ gọi đi, @@0@@.", mapping) == "Cứ gọi đi, Captain."


def test_recase_leaves_a_line_starting_with_the_term_alone(terms):
    _, mapping = glossary.protect("Captain, look out.", terms)
    assert glossary.restore("@@0@@, coi chừng.", mapping) == "Captain, coi chừng."
