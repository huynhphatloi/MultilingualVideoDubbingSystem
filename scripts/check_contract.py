#!/usr/bin/env python3
"""Fail when the form, the local API, and the Colab server disagree.

The three lists drifted apart once already: the form offered seven voices while
the Colab server implemented one, and every extra choice was silently dropped
because FastAPI ignores unknown form fields. Nothing failed loudly, so the
result simply reported a model that had never run.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def literals(path: Path, wanted: set[str]) -> dict:
    """Read module-level constants without importing the heavy ML dependencies."""
    found: dict[str, set] = {}
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name) or target.id not in wanted:
                continue
            if isinstance(node.value, ast.Dict):
                found[target.id] = {ast.literal_eval(key) for key in node.value.keys}
            else:
                found[target.id] = set(ast.literal_eval(node.value))
    missing = wanted - set(found)
    if missing:
        raise SystemExit(f"{path}: could not find {sorted(missing)}")
    return found


def form_options(path: Path) -> dict:
    workflow = json.loads(path.read_text(encoding="utf-8"))
    for node in workflow["nodes"]:
        if node["type"] != "n8n-nodes-base.formTrigger":
            continue
        return {
            field["fieldLabel"]: {
                option["option"].split()[0]
                for option in field["fieldOptions"]["values"]
            }
            for field in node["parameters"]["formFields"]["values"]
            if field.get("fieldOptions", {}).get("values")
        }
    raise SystemExit(f"{path}: no formTrigger node")


def main() -> int:
    app = literals(
        ROOT / "ai-service/app.py",
        {"_WHISPER_MODELS", "_TRANSLATION_ENGINES", "_TTS_ENGINES", "_CLONING_ENGINES"},
    )
    server = literals(
        ROOT / "colab/server.py",
        {"WHISPER_MODELS", "TRANSLATION_ENGINES", "TTS_ENGINES"},
    )
    form = form_options(ROOT / "n8n/workflows/simple-dubbing.json")

    pairs = [
        ("whisper models, app vs colab", app["_WHISPER_MODELS"], server["WHISPER_MODELS"]),
        ("translation engines, app vs colab", app["_TRANSLATION_ENGINES"], server["TRANSLATION_ENGINES"]),
        ("voice engines, app vs colab", app["_TTS_ENGINES"], server["TTS_ENGINES"]),
        ("whisper models, form vs app", form["Whisper Model"], app["_WHISPER_MODELS"]),
        ("translation engines, form vs app", form["Translation Engine"], app["_TRANSLATION_ENGINES"]),
        ("voice engines, form vs app", form["Voice Engine"], app["_TTS_ENGINES"]),
    ]
    failures = 0
    for label, left, right in pairs:
        if left == right:
            continue
        failures += 1
        print(f"MISMATCH {label}")
        if left - right:
            print(f"  only on the left : {sorted(left - right)}")
        if right - left:
            print(f"  only on the right: {sorted(right - left)}")

    unknown = app["_CLONING_ENGINES"] - app["_TTS_ENGINES"]
    if unknown:
        failures += 1
        print(f"MISMATCH _CLONING_ENGINES names an unknown engine: {sorted(unknown)}")

    if failures:
        print(f"\n{failures} contract mismatch(es)")
        return 1
    print("contract check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
