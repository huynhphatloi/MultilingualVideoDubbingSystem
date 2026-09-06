from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PORT = 8399

notebook_deps = pytest.importorskip("uvicorn") and pytest.importorskip("fastapi")


def launch_cell() -> str:
    notebook = json.loads((ROOT / "colab/ai_service.ipynb").read_text(encoding="utf-8"))
    matches = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "uvicorn.Server" in "".join(cell["source"])
    ]
    assert len(matches) == 1, f"expected one launch cell, found {len(matches)}"
    return matches[0]


def code_cells():  # noqa: ANN201
    notebook = json.loads((ROOT / "colab/ai_service.ipynb").read_text(encoding="utf-8"))
    return [
        (index, "".join(cell["source"]))
        for index, cell in enumerate(notebook["cells"])
        if cell["cell_type"] == "code"
    ]


def as_python(source: str) -> str:
    lines = []
    for line in source.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith(("!", "%")):
            lines.append(" " * (len(line) - len(stripped)) + "pass")
        else:
            lines.append(line)
    return "\n".join(lines)


BORROWABLE = {
    "subprocess", "re", "os", "sys", "time", "threading", "secrets", "json",
    "uvicorn", "Path", "shutil", "textwrap",
}


class TestEveryCell:
    def test_every_code_cell_is_valid_python(self):
        import ast

        for index, source in code_cells():
            try:
                ast.parse(as_python(source))
            except SyntaxError as failure:
                raise AssertionError(f"cell {index} does not parse: {failure}") from failure

    def test_no_cell_borrows_an_import_from_another(self):
        import ast

        for index, source in code_cells():
            tree = ast.parse(as_python(source))
            imported, assigned, used = set(), set(), set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported.add((alias.asname or alias.name).split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        imported.add(alias.asname or alias.name)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            assigned.add(target.id)
                elif isinstance(node, (ast.For, ast.comprehension)):
                    target = getattr(node, "target", None)
                    if isinstance(target, ast.Name):
                        assigned.add(target.id)
                elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                    used.add(node.value.id)
            missing = (used & BORROWABLE) - imported - assigned
            assert not missing, f"cell {index} uses {sorted(missing)} without importing it"


class TestLaunchCellStatics:
    def test_it_stops_the_previous_server_before_importing(self):
        source = launch_cell()
        assert "_previous_server.should_exit = True" in source
        assert "_previous_thread.join" in source
        assert source.index("should_exit") < source.index("import server")

    def test_it_purges_the_project_modules(self):
        source = launch_cell()
        assert "del sys.modules[" in source
        for module in ("server", "providers", "core", "dubflow_core"):
            assert f'"{module}"' in source

    def test_it_sets_up_sys_path_before_importing_anything_of_ours(self):
        source = launch_cell()
        assert "sys.path.insert" in source
        assert source.index("sys.path.insert") < source.index("import providers")

    def test_it_refuses_to_run_beside_a_server_it_cannot_stop(self):
        source = launch_cell()
        assert "older version of this notebook" in source
        assert "Restart session" in source

    def test_it_reports_the_version_it_started(self):
        assert "server.app.version" in launch_cell()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="the API needs FFmpeg tooling")
def test_the_cell_reloads_a_pulled_checkout_without_a_restart(tmp_path):
    shutil.copytree(ROOT / "colab", tmp_path / "colab")
    shutil.copytree(ROOT / "dubflow_core", tmp_path / "dubflow_core")

    driver = textwrap.dedent(
        """
        import json, os, sys, urllib.request
        os.chdir({colab!r})
        sys.path.insert(0, {colab!r})
        try:
            import dubflow_core
        except ImportError:
            pass
        else:
            raise SystemExit("PREMISE FAILED: the repository root was already importable")

        PORT = {port}
        WHISPER_MODEL = "small"
        HF_TOKEN = ""
        def version():
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT) as reply:
                return json.load(reply).get("version")

        CELL = {cell!r}
        exec(CELL, globals())
        print("VERSION_1", version())

        target = {server!r}
        source = open(target).read()
        assert 'version="5.0"' in source, "the checkout to modify was not found"
        with open(target, "w") as handle:
            handle.write(source.replace('version="5.0"', 'version="5.1-pulled"'))

        exec(CELL, globals())
        print("VERSION_2", version())
        """
    ).format(
        colab=str(tmp_path / "colab"),
        server=str(tmp_path / "colab" / "server.py"),
        cell=launch_cell(),
        port=PORT,
    )

    environment = {name: value for name, value in os.environ.items() if name != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-c", driver],
        capture_output=True, text=True, timeout=300, env=environment,
    )
    output = result.stdout + result.stderr
    assert "PREMISE FAILED" not in output, output
    assert "VERSION_1 5.0" in output, output
    assert "VERSION_2 5.1-pulled" in output, output[-2000:]
