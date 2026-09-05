"""Pure-stdlib code shared by the Colab AI service and the local FastAPI app.

Nothing in this package imports torch, transformers, FastAPI or any model
library, so both services - and the offline tests - can use it. It holds the
things that used to be copied into three places and drift apart: the language
table, the canonical segment schema, and the duration-alignment arithmetic.
"""
from __future__ import annotations

__all__ = ["languages", "segments", "alignment"]
