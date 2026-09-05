"""Lip-sync providers.

The abstraction and the pipeline stage exist; no provider is registered in this
build. The obvious candidates were left out on purpose rather than shipped
half-working:

* Wav2Lip - the original weights are released for non-commercial research only
  and the reference implementation pins a torch that conflicts with everything
  else in the notebook.
* Video-Retalking, DINet, and the newer diffusion talkers - each needs its own
  repository, its own face-detection weights and several gigabytes of VRAM, and
  none has a maintained pip package.

Registering one is a matter of adding a ModelSpec here plus a module with the
same `cache_key`/`build` pair every other provider has; the stage, the job
config and the frontend flag already exist and will pick it up.
"""
from __future__ import annotations

from providers.base import Registry

REGISTRY = Registry("lipsync")

DEFAULT_MODEL = None
