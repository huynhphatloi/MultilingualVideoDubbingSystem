#!/usr/bin/env python3
"""Validate contracts shared by the UI, APIs, workflow, and model registry."""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "colab"))

import providers  # noqa: E402
from providers.asr.mms import ADAPTER_OVERRIDES, UNSUPPORTED  # noqa: E402
from providers.tts.edge import LOCALE_OVERRIDES  # noqa: E402
from providers.tts.registry import MMS_TTS_LANGUAGES, MMS_TTS_OVERRIDES  # noqa: E402

FORM_MODEL_FIELDS = {
    "Speech Recognition Model": "asr",
    "Translation Model": "translation",
    "Voice Model": "tts",
}
FORM_FLAG_FIELDS = (
    "Speaker Diarization", "Duration Alignment", "Source Separation",
)


def failures() -> list:
    problems: list = []
    problems.extend(providers.consistency_problems())
    problems.extend(check_workflow())
    problems.extend(check_local_api())
    problems.extend(check_frontend())
    problems.extend(check_provider_maps())
    return problems


def check_workflow() -> list:
    problems = []
    workflow = json.loads(
        (ROOT / "n8n/workflows/simple-dubbing.json").read_text(encoding="utf-8")
    )
    nodes = workflow["nodes"]

    form = next(
        (node for node in nodes if node["type"] == "n8n-nodes-base.formTrigger"), None
    )
    if form is None:
        return ["n8n workflow has no formTrigger node"]

    fields = {
        field["fieldLabel"]: [
            option["option"].split()[0]
            for option in field.get("fieldOptions", {}).get("values", [])
        ]
        for field in form["parameters"]["formFields"]["values"]
    }
    for label, task in FORM_MODEL_FIELDS.items():
        if label not in fields:
            continue
        known = set(providers.registry(task).ids())
        unknown = [value for value in fields[label] if value not in known]
        if unknown:
            problems.append(
                f"n8n form field '{label}' offers ids the registry does not "
                f"define: {sorted(unknown)}"
            )
    for label in FORM_FLAG_FIELDS:
        values = {value.lower() for value in fields.get(label, [])}
        if not values <= {"on", "off", "automatic"}:
            problems.append(
                f"n8n form field '{label}' must offer On/Off options; got {sorted(values)}"
            )

    served = local_api_stages()
    called = [
        re.sub(r".*/stages/", "", node["parameters"]["url"])
        for node in nodes
        if node["type"] == "n8n-nodes-base.httpRequest"
        and "/stages/" in node["parameters"].get("url", "")
    ]
    if called != served:
        problems.append(
            f"n8n stage nodes {called} do not match the local API's stages {served}"
        )
    return problems


def literals(path: Path, wanted: set) -> dict:
    found: dict = {}
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in wanted:
                found[target.id] = ast.literal_eval(node.value)
    return found


def local_api_stages() -> list:
    return literals(ROOT / "ai-service/app.py", {"STAGES"}).get("STAGES", [])


def check_local_api() -> list:
    problems = []
    source = (ROOT / "ai-service/app.py").read_text(encoding="utf-8")
    for banned in ("_WHISPER_MODELS", "_TRANSLATION_ENGINES", "_TTS_ENGINES",
                   "_CLONING_ENGINES"):
        if banned in source:
            problems.append(
                f"ai-service/app.py defines {banned}: model lists belong to the "
                f"registry, which /capabilities exposes"
            )
    if "from dubflow_core import languages" not in source:
        problems.append("ai-service/app.py no longer imports the shared language table")
    return problems


def check_frontend() -> list:
    problems = []
    source = (ROOT / "frontend/index.html").read_text(encoding="utf-8")
    if '"/capabilities"' not in source:
        problems.append("frontend/index.html does not fetch /capabilities")
    if "Whisper Model" in source:
        problems.append(
            "frontend/index.html still labels the recogniser 'Whisper Model'; it "
            "offers more than Whisper now"
        )
    match = re.search(r"const STAGES = \[(.*?)\];", source, re.S)
    if not match:
        problems.append("frontend/index.html has no STAGES list")
        return problems
    ids = re.findall(r'id:\s*"([a-z_]+)"', match.group(1))
    expected = ["upload"] + local_api_stages()
    if ids != expected:
        problems.append(f"frontend stages {ids} do not match {expected}")
    return problems


def check_provider_maps() -> list:
    problems = []
    languages = providers.registry("asr").get("mms_asr").languages
    for code in languages:
        if code in UNSUPPORTED:
            problems.append(f"mms_asr advertises '{code}', which it maps as unsupported")
    for code in ADAPTER_OVERRIDES:
        if code not in languages:
            problems.append(f"mms adapter override '{code}' is not advertised by mms_asr")

    tts = providers.registry("tts")
    if set(tts.get("mms").languages) != set(MMS_TTS_LANGUAGES):
        problems.append("the mms voice spec and MMS_TTS_LANGUAGES disagree")
    for code in MMS_TTS_OVERRIDES:
        if code not in MMS_TTS_LANGUAGES:
            problems.append(f"MMS-TTS override '{code}' is not an advertised language")
    for code in LOCALE_OVERRIDES:
        if code not in tts.get("edge").languages:
            problems.append(f"Edge locale override '{code}' is not an advertised language")
    return problems


def main() -> int:
    problems = failures()
    for problem in problems:
        print(f"MISMATCH {problem}")
    if problems:
        print(f"\n{len(problems)} contract mismatch(es)")
        return 1
    print("contract check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
