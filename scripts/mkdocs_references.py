"""Render the agent skill's existing reference guides without duplicating them."""

from pathlib import Path

from mkdocs.structure.files import File


def on_files(files, config):
    source = Path(__file__).resolve().parent.parent / "references"
    for path in sorted(source.glob("*.md")):
        files.append(File.generated(config, f"reference/{path.name}", content=path.read_text(encoding="utf-8")))
    return files
