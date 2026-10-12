"""Prepare both novels with identical rules; run: python prepare_cross_novel.py.

Uses the cached ZIP by default. If absent, downloads nine chapters and LICENSE
from the pinned commit. Original Task1-4 inputs and results are not modified.
"""

from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import unicodedata
import urllib.request
import zipfile


COMMIT = "cf04165c84a11bc00bbbb9cceef936e0781740ae"
REPOSITORY = "https://github.com/BlackIQ/the-great-gatsby"
SOURCE_NAMES = [f"Chapter-{i}/README.md" for i in range(1, 10)] + ["LICENSE"]
BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "cross_novel_data"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_sources():
    archive_path = OUTPUT / "gatsby_source.zip"
    if not archive_path.exists():
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in SOURCE_NAMES:
                url = f"https://raw.githubusercontent.com/BlackIQ/the-great-gatsby/{COMMIT}/{name}"
                request = urllib.request.Request(url, headers={"User-Agent": "course-project"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    archive.writestr(name, response.read())
        archive_path.write_bytes(buffer.getvalue())
    with zipfile.ZipFile(archive_path) as archive:
        if set(archive.namelist()) != set(SOURCE_NAMES):
            raise ValueError("Source archive must contain chapters 1-9 and LICENSE only")
        sources = {name: archive.read(name) for name in SOURCE_NAMES}
    return sources, sha256(archive_path.read_bytes())


def clean_chapter(data, chapter):
    text = data.decode("utf-8-sig")
    lines = text.splitlines()
    # This pinned version has exactly two Markdown heading lines per chapter.
    # Brackets in chapter 9 are prose/editorial text, not Markdown links.
    if lines[0].strip() != "### The Greate Gatsby" or lines[2].strip() != f"**Chapter {chapter}**":
        raise ValueError(f"Unexpected chapter {chapter} headings; inspect source before cleaning")
    body = "\n".join(lines[3:]).strip()
    if re.search(r"(?m)^\s*#|[*_`]|\]\(", body):
        raise ValueError(f"Unexpected Markdown in chapter {chapter}; manual review required")
    if not body:
        raise ValueError(f"Empty chapter {chapter}")
    return body


def normalize(text):
    # Same mapping for both books: preserve case; use ASCII punctuation.
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(str.maketrans({
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u2212": "-",
        "\u2026": "...", "\u2060": "", "\ufeff": "",
    }))
    # Accent folding, e.g. é -> e, ô -> o, applied to both inputs.
    text = "".join(ch for ch in unicodedata.normalize("NFKD", text)
                   if unicodedata.category(ch) != "Mn")
    # Paragraphs, CR/LF, tabs, and Unicode spaces become one ASCII space.
    return re.sub(r"\s+", " ", text).strip()


def character_counts(text):
    return {f"U+{ord(ch):04X} {repr(ch)}": n
            for ch, n in sorted(Counter(text).items())}


def main():
    OUTPUT.mkdir(exist_ok=True)
    sources, archive_hash = load_sources()
    manifest_path = OUTPUT / "provenance.json"
    previous = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else None
    source_hashes = {name: sha256(data) for name, data in sources.items()}
    if previous and (previous["commit"] != COMMIT or previous["source_sha256"] != source_hashes):
        raise ValueError("Cached source differs from the recorded commit/hashes")

    chapters = [clean_chapter(sources[f"Chapter-{i}/README.md"], i) for i in range(1, 10)]
    original_a = (BASE / "GameOfThrones.txt").read_bytes()
    raw_a = original_a.decode("gb18030", errors="strict")
    raw_b = "\n\n".join(chapters)
    normalized = {"A": normalize(raw_a), "B": normalize(raw_b)}

    # A common observed alphabet, with '?' as the documented unknown symbol.
    # It already occurs in both texts. Do not delete rare symbols or add fake text.
    common = set(normalized["A"]) & set(normalized["B"])
    if "?" not in common:
        raise ValueError("The common alphabet must contain the '?' fallback")
    processed = {
        name: "".join(ch if ch in common else "?" for ch in text)
        for name, text in normalized.items()
    }
    if set(processed["A"]) != set(processed["B"]):
        raise ValueError("Final character sets differ")

    records = {}
    filenames = {"A": "GameOfThrones.txt", "B": "TheGreatGatsby.txt"}
    for name, text in processed.items():
        data = text.encode("utf-8")
        (OUTPUT / filenames[name]).write_bytes(data)
        replacements = Counter(ch for ch in normalized[name] if ch not in common)
        records[name] = {
            "file": filenames[name], "encoding": "utf-8",
            "characters": len(text), "utf8_bytes": len(data),
            "whitespace_separated_tokens": len(text.split()),
            "sha256": sha256(data),
            "characters_before_normalization": len(raw_a if name == "A" else raw_b),
            "alphabet_before_fallback": "".join(sorted(set(normalized[name]))),
            "fallback_replacements": dict(sorted(replacements.items())),
            "fallback_replacement_count": sum(replacements.values()),
            "character_frequencies": character_counts(text),
        }

    manifest = {
        "repository": REPOSITORY, "commit": COMMIT,
        "retrieved_date": previous["retrieved_date"] if previous else datetime.now(timezone(timedelta(hours=8))).date().isoformat(),
        "retrieved_date_timezone": "Asia/Shanghai",
        "source_archive": "gatsby_source.zip", "source_archive_sha256": archive_hash,
        "source_sha256": source_hashes,
        "chapter_order": list(range(1, 10)),
        "chapter_body_characters": [len(chapter) for chapter in chapters],
        "source_url_pattern": f"https://raw.githubusercontent.com/BlackIQ/the-great-gatsby/{COMMIT}/Chapter-{{i}}/README.md",
        "original_a": {"path": "../GameOfThrones.txt", "encoding": "gb18030", "sha256": sha256(original_a)},
        "rules": [
            "Exclude Chapter-0, repository README and HTML; remove exactly two Markdown headings per chapter",
            "Keep chapter 9 square-bracket prose; concatenate chapters 1-9 in order",
            "Apply Unicode NFKC; curly quotes to ASCII quotes; en/em dashes to ASCII hyphen; ellipsis to three dots",
            "Remove word joiner and BOM; remove decomposed accent marks; preserve letter case",
            "Collapse all whitespace to one ASCII space; strip leading/trailing whitespace",
            "Compute intersection of normalized observed alphabets; map all other characters in both books to '?'",
            "Write UTF-8 without BOM or added trailing newline",
        ],
        "character_sets_equal": True, "alphabet_size": len(common),
        "alphabet": "".join(sorted(common)), "outputs": records,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    for name, record in records.items():
        print(name, record["characters"], "characters; replacements:", record["fallback_replacements"])
    print("Identical alphabet:", len(common), "characters")
    print("Retrieval date:", manifest["retrieved_date"])


if __name__ == "__main__":
    main()
