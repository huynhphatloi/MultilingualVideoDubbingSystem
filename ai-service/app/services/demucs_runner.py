"""Launcher that runs ``demucs.separate`` under torch's pre-2.6 load semantics.

Why this file exists
--------------------
demucs 4.0.1 loads its checkpoint with::

    package = torch.load(path, 'cpu')          # demucs/states.py:59

The package is not a plain state-dict: it also carries ``klass`` (the model
class itself) plus ``args``/``kwargs``. Since **torch 2.6** ``torch.load``
defaults to ``weights_only=True``, which refuses to unpickle arbitrary classes,
so every separation dies before the model is even built. We are on torch 2.11
because pyannote.audio 4.x requires >= 2.8, so there is no going back.

Upstream fixed exactly this in demucs 4.1.0 with a one-word change::

    package = torch.load(path, 'cpu', weights_only=False)

but 4.1.0 also added a hard dependency on ``sphn``, a Rust extension that has
no linux/arm64 wheel in any release - unbuildable in the container. This shim
applies the identical upstream fix without dragging in that dependency.

Safety: the relaxation is scoped to this short-lived subprocess, and the only
thing it ever loads is a demucs checkpoint fetched from Meta's official release
URLs (or the local model cache). No user-supplied file is ever unpickled here.

Invoked as a plain script so it needs no package context:

    python app/services/demucs_runner.py -n htdemucs --two-stems vocals ...
"""
from __future__ import annotations

import sys

import torch

_torch_load = torch.load


def _load_allowing_pickled_classes(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _load_allowing_pickled_classes

from demucs.separate import main  # noqa: E402  (must come after the patch)

if __name__ == "__main__":
    sys.exit(main() or 0)
