"""Source separation providers."""
from __future__ import annotations

from providers.base import ModelSpec, ProviderSpec, Registry

REGISTRY = Registry("separation")

REGISTRY.add_provider(ProviderSpec(
    id="demucs",
    task="separation",
    display_name="Demucs v4",
    homepage="https://github.com/adefossez/demucs",
    notes="Hybrid Transformer Demucs, MIT licensed.",
))

REGISTRY.add(ModelSpec(
    id="htdemucs",
    task="separation",
    provider="demucs",
    display_name="Demucs htdemucs (two stems)",
    repo_id="htdemucs",
    languages=(),
    multilingual=True,
    license="mit",
    optional_package="demucs",
    import_name="demucs",
    notes="Splits the original track into vocals and everything else, so the dub "
          "can sit on the real background instead of a quietened original.",
    module="providers.separation.demucs",
))

REGISTRY.add(ModelSpec(
    id="htdemucs_ft",
    task="separation",
    provider="demucs",
    display_name="Demucs htdemucs_ft (fine-tuned, slower)",
    repo_id="htdemucs_ft",
    languages=(),
    multilingual=True,
    license="mit",
    optional_package="demucs",
    import_name="demucs",
    notes="Four times slower than htdemucs for a modest quality gain.",
    module="providers.separation.demucs",
))

DEFAULT_MODEL = "htdemucs"
