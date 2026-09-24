from __future__ import annotations

import ast
import json
import os
import socket
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def cells() -> list[str]:
    notebook = json.loads((ROOT / "colab/ai_service.ipynb").read_text())
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_run_all_uses_a_fresh_process_and_checks_the_real_diarization_model():
    config, checkout, install, preflight, launch, tunnel = cells()
    for source in (config, checkout, install, preflight, launch, tunnel):
        ast.parse(source)

    code = next(
        ast.literal_eval(node.value)
        for node in ast.parse(preflight).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "probe" for target in node.targets)
    )
    ast.parse(code)
    assert "torchvision.ops import nms" in code
    assert "providers.diarization.load(model)" in code
    assert 'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1"' in preflight
    assert 'subprocess.run([sys.executable, "-c", probe], check=True' in preflight
    assert 'sys.executable, "-m", "uvicorn", "server:app"' in launch
    assert 'f"http://127.0.0.1:{PORT}/capabilities"' in launch
    assert '"Authorization": f"Bearer {AUTH_TOKEN}"' in launch


def test_cuda_wheels_are_installed_last_and_failures_stop_run_all():
    install = cells()[2]
    assert "check=True" in install
    assert install.index('"pyannote.audio==3.4.0"') < install.index('"torchvision==0.23.0"')
    assert '"--force-reinstall", "--no-deps"' in install


def test_api_cell_restarts_a_fresh_process_on_rerun(tmp_path):
    pytest.importorskip("uvicorn")
    pytest.importorskip("fastapi")

    colab = tmp_path / "colab"
    colab.mkdir()
    server = colab / "server.py"

    def write_server(version):
        server.write_text(
            "import os\n"
            "from fastapi import FastAPI, Header, HTTPException\n"
            f'app = FastAPI(version="{version}")\n'
            '@app.get("/health")\n'
            'def health(): return {"version": app.version}\n'
            '@app.get("/capabilities")\n'
            'def capabilities(authorization: str = Header(None)):\n'
            '    if authorization != f"Bearer {os.environ[\'AUTH_TOKEN\']}": raise HTTPException(401)\n'
            '    return {"version": app.version}\n'
        )

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    namespace = {
        "repo": tmp_path,
        "env": {**os.environ, "PYTHONPATH": str(colab)},
        "HF_TOKEN": "test-token",
        "WHISPER_MODEL": "small",
        "PORT": port,
    }
    try:
        for version in ("5.0", "5.1"):
            write_server(version)
            exec(cells()[4], namespace)
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health") as response:
                assert json.load(response)["version"] == version
    finally:
        process = namespace.get("api_process")
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=15)
