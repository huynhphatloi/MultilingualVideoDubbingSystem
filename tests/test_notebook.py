"""The Colab notebook's launch cell, exercised the way Colab runs it.

These tests exist because two bugs shipped inside this one cell. The first made
`Run all` silently keep the previous build: `import` is a no-op once a module is
cached, and the old server thread was still alive, so a pulled checkout sat on
disk unused while the tunnel got a fresh URL. The second was the fix's own
fault - reordering the imports exposed that `sys.path` was only ever set up as a
side effect of importing `server`, so importing anything else first failed with
`No module named 'dubflow_core'`.

The integration test therefore reproduces Colab's environment rather than the
test runner's: the working directory is the repository's `colab/` folder and the
repository root is *not* importable until the cell makes it so.
"""
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
    return "".join(notebook["cells"][4]["source"])


def code_cells():  # noqa: ANN201
    notebook = json.loads((ROOT / "colab/ai_service.ipynb").read_text(encoding="utf-8"))
    return [
        (index, "".join(cell["source"]))
        for index, cell in enumerate(notebook["cells"])
        if cell["cell_type"] == "code"
    ]


def as_python(source: str) -> str:
    """A cell as plain Python: `!shell` and `%magic` lines become `pass`.

    The indentation has to survive, because those lines appear inside `if`
    blocks in the checkout cell.
    """
    lines = []
    for line in source.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith(("!", "%")):
            lines.append(" " * (len(line) - len(stripped)) + "pass")
        else:
            lines.append(line)
    return "\n".join(lines)


#: Names that must be imported by the cell that uses them, rather than borrowed
#: from whichever earlier cell happened to import them.
BORROWABLE = {
    "subprocess", "re", "os", "sys", "time", "threading", "secrets", "json",
    "uvicorn", "Path", "shutil", "textwrap",
}


class TestEveryCell:
    """A cell must stand on its own: notebook cells get re-run individually."""

    def test_every_code_cell_is_valid_python(self):
        import ast

        for index, source in code_cells():
            try:
                ast.parse(as_python(source))
            except SyntaxError as failure:
                raise AssertionError(f"cell {index} does not parse: {failure}") from failure

    def test_no_cell_borrows_an_import_from_another(self):
        """The tunnel cell once used `subprocess` because an earlier cell had
        imported it; removing that unused import from the other cell broke it."""
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
        assert "api_server.should_exit = True" in source
        assert "api_thread.join" in source
        assert source.index("should_exit") < source.index("import server")

    def test_it_purges_the_project_modules(self):
        source = launch_cell()
        assert "del sys.modules[" in source
        for module in ("server", "jobs", "pipeline", "providers", "core", "dubflow_core"):
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


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="pipeline import needs FFmpeg tooling")
def test_the_cell_reloads_a_pulled_checkout_without_a_restart(tmp_path):
    """Run the cell, change the checkout underneath it, run it again.

    This is what `git pull` + `Run all` does inside a live Colab session. Before
    the fix the second run reported the first run's version.
    """
    shutil.copytree(ROOT / "colab", tmp_path / "colab")
    shutil.copytree(ROOT / "dubflow_core", tmp_path / "dubflow_core")

    driver = textwrap.dedent(
        """
        import json, os, sys, urllib.request
        os.chdir({colab!r})
        # Colab's situation exactly: the colab/ folder is importable because it
        # is the working directory; the repository root above it is not.
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
        os.environ["JOBS_ROOT"] = {jobs!r}

        def version():
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT) as reply:
                return json.load(reply).get("version")

        CELL = {cell!r}
        exec(CELL, globals())
        print("VERSION_1", version())

        # A `git pull` lands new code on disk while the session keeps running.
        # Read before opening for write: "w" truncates on open.
        target = {server!r}
        source = open(target).read()
        assert 'version="4.0"' in source, "the checkout to modify was not found"
        with open(target, "w") as handle:
            handle.write(source.replace('version="4.0"', 'version="4.1-pulled"'))

        exec(CELL, globals())
        print("VERSION_2", version())
        """
    ).format(
        colab=str(tmp_path / "colab"),
        jobs=str(tmp_path / "jobs"),
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
    assert "VERSION_1 4.0" in output, output
    assert "VERSION_2 4.1-pulled" in output, output[-2000:]
