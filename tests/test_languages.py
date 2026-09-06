from __future__ import annotations

import pytest

from dubflow_core import languages as L


def test_every_row_is_complete():
    for code, row in L.LANGUAGES.items():
        assert row.code == code
        assert row.name and row.iso3 and row.flores
        assert len(row.iso3) == 3
        assert "_" in row.flores


def test_codes_are_unique_across_alphabets():
    assert len({row.iso3 for row in L.LANGUAGES.values()}) == len(L.LANGUAGES)
    assert len({row.flores for row in L.LANGUAGES.values()}) == len(L.LANGUAGES)


def test_normalise_treats_auto_as_unknown():
    assert L.normalise("auto") is None
    assert L.normalise("") is None
    assert L.normalise(None) is None
    assert L.normalise("  VI ") == "vi"


def test_subset_rejects_codes_outside_the_table():
    assert L.subset("en", "vi") == ("en", "vi")
    with pytest.raises(ValueError):
        L.subset("en", "klingon")


def test_exclude_keeps_table_order():
    remaining = L.exclude("ms", "si")
    assert "ms" not in remaining and "si" not in remaining
    assert len(remaining) == len(L.LANGUAGES) - 2
    assert list(remaining) == [code for code in L.CODES if code not in {"ms", "si"}]
