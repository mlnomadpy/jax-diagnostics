"""Build the GitHub Pages landing page, MkDocs handbook, and combined sitemap."""

from pathlib import Path
from shutil import copytree
import subprocess
import sys
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
NAMESPACE = "http://www.sitemaps.org/schemas/sitemap/0.9"


def main():
    subprocess.run([sys.executable, "-m", "mkdocs", "build"], cwd=ROOT, check=True)
    copytree(ROOT / "docs", SITE, dirs_exist_ok=True)
    landing = ET.parse(SITE / "sitemap.xml")
    handbook = ET.parse(SITE / "docs/sitemap.xml")
    landing.getroot().extend(handbook.getroot())
    ET.register_namespace("", NAMESPACE)
    ET.indent(landing, space="  ")
    landing.write(SITE / "sitemap.xml", encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    main()
