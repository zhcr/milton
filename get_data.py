"""Download and clean Paradise Lost from Project Gutenberg."""

import re
import urllib.request
from pathlib import Path

GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/26/pg26.txt"
DATA_DIR = Path(__file__).parent / "data"


def download_raw():
    DATA_DIR.mkdir(exist_ok=True)
    raw_path = DATA_DIR / "paradise_lost_raw.txt"
    if not raw_path.exists():
        print("Downloading Paradise Lost from Project Gutenberg...")
        urllib.request.urlretrieve(GUTENBERG_URL, raw_path)
    return raw_path.read_text(encoding="utf-8")


def clean(raw: str) -> str:
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK PARADISE LOST ***"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK PARADISE LOST ***"

    start = raw.index(start_marker) + len(start_marker)
    end = raw.index(end_marker)
    text = raw[start:end]

    # Strip the table of contents and introduction — poem starts at "Book I"
    book_i = text.index("Book I\n")
    text = text[book_i:]

    # Normalize whitespace: collapse runs of 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing whitespace on each line
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = text.strip() + "\n"

    return text


def main():
    raw = download_raw()
    text = clean(raw)

    out_path = DATA_DIR / "paradise_lost.txt"
    out_path.write_text(text, encoding="utf-8")

    words = len(text.split())
    lines = text.count("\n")
    books = len(re.findall(r"^Book [IVX]+", text, re.MULTILINE))
    print(f"Saved {out_path}")
    print(f"  {books} books, {lines} lines, {words} words, {len(text)} chars")


if __name__ == "__main__":
    main()
