#!/usr/bin/env python3
"""Write one notebook backend's URL and token into .env.

The notebook prints a ready-to-run `make colab URL=... TOKEN=...` line, so a
new session costs one paste instead of editing a file by hand. Existing keys,
comments and unrelated settings are preserved.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / ".env"
EXAMPLE = ROOT / ".env.example"


def assign(lines: list[str], key: str, value: str) -> list[str]:
    """Replace the line defining `key`, or append one if it is absent."""
    replacement = f"{key}={value}"
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(f"{key}=") and not stripped.startswith("#"):
            lines[index] = replacement
            return lines
    lines.append(replacement)
    return lines


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("usage: set_backend.py <name> <url> [token]", file=sys.stderr)
        return 2
    name, url = argv[1].strip().lower(), argv[2].strip().rstrip("/")
    token = (argv[3] if len(argv) > 3 else "").strip()

    if not url.startswith(("http://", "https://")):
        print(
            f"URL must start with http:// or https://, got {url!r}.\n"
            f"The URL and the token are easy to swap - check the notebook output.",
            file=sys.stderr,
        )
        return 1

    if not ENV.exists():
        ENV.write_text(EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
        print("Created .env from .env.example.")

    lines = ENV.read_text(encoding="utf-8").splitlines()
    lines = assign(lines, f"{name.upper()}_API_URL", url)
    lines = assign(lines, f"{name.upper()}_API_TOKEN", token)
    ENV.write_text("\n".join(lines) + "\n", encoding="utf-8")

    shown = f"{token[:4]}...{token[-4:]}" if len(token) > 8 else "(empty)"
    print(f"{name}: {url}  token={shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
