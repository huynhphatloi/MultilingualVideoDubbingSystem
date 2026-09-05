"""Translation engines.

NLLB-200 covers all 200 FLORES languages, which includes every code in the
application table. SeamlessM4T's text side covers every code except Sinhala,
which its language table does not list.
"""
from __future__ import annotations

from dubflow_core import languages as L
from providers.base import ModelSpec, ProviderSpec, Registry

REGISTRY = Registry("translation")

REGISTRY.add_provider(ProviderSpec(
    id="nllb",
    task="translation",
    display_name="NLLB-200",
    homepage="https://huggingface.co/facebook/nllb-200-distilled-600M",
    notes="Sentence-level translation; part of the base install.",
))
REGISTRY.add_provider(ProviderSpec(
    id="seamless",
    task="translation",
    display_name="SeamlessM4T",
    homepage="https://huggingface.co/facebook/hf-seamless-m4t-medium",
    notes="Text-to-text mode. Non-commercial licence.",
))

REGISTRY.add(ModelSpec(
    id="nllb",
    task="translation",
    provider="nllb",
    display_name="NLLB-200 distilled 600M",
    repo_id="facebook/nllb-200-distilled-600M",
    languages=L.ALL,
    multilingual=True,
    license="cc-by-nc-4.0",
    notes="Non-commercial licence. The default, and the smallest of the pair.",
    module="providers.translation.nllb",
))

REGISTRY.add(ModelSpec(
    id="nllb_1.3b",
    task="translation",
    provider="nllb",
    display_name="NLLB-200 distilled 1.3B",
    repo_id="facebook/nllb-200-distilled-1.3B",
    languages=L.ALL,
    multilingual=True,
    license="cc-by-nc-4.0",
    notes="Better quality, about twice the memory of the 600M checkpoint.",
    module="providers.translation.nllb",
))

REGISTRY.add(ModelSpec(
    id="seamless",
    task="translation",
    provider="seamless",
    display_name="SeamlessM4T medium (text)",
    repo_id="facebook/hf-seamless-m4t-medium",
    #: Every application language appears in the M4T target-text column except
    #: Sinhala, which is absent from the table.
    languages=L.exclude("si"),
    multilingual=True,
    license="cc-by-nc-4.0",
    notes="Non-commercial licence. Shares its checkpoint with the SeamlessM4T "
          "recogniser.",
    module="providers.translation.seamless",
))

DEFAULT_MODEL = "nllb"
