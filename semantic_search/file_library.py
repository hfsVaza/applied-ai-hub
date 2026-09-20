"""A drop-anything document library with content-aware organisation.

Files dropped into the library are text-extracted, embedded, and can then be
grouped either by what they are *about* (clustering the embeddings) or by what
they *are* (file type). Extracted text also feeds the search index, so anything
added here becomes searchable.
"""

import json
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

# Subfolder of the search data directory that holds uploaded files.
LIBRARY_DIRNAME = "library"

# Text we will not try to embed: too short to carry meaning.
MIN_USEFUL_CHARS = 20

# Characters of extracted text kept per file. Enough to characterise a
# document's subject without letting one long file dominate the index.
MAX_EXTRACT_CHARS = 200_000

# Distance threshold for agglomerative clustering, in cosine distance.
# Lower splits into more, tighter groups; higher merges more aggressively.
# Calibrated on a mixed pile: two recipes on different dishes sit 0.76 apart
# and should group, while a recipe and an ML paper sit 0.99 apart and must
# not. Above ~0.92 unrelated subjects start collapsing together.
DEFAULT_CLUSTER_DISTANCE = 0.85


# --------------------------------------------------------------- file types

TYPE_CATEGORIES = {
    "Documents": {".pdf", ".docx", ".doc", ".rtf", ".odt", ".epub", ".pages"},
    "Text & Markdown": {".txt", ".md", ".markdown", ".rst", ".log", ".tex"},
    "Spreadsheets & Data": {".xlsx", ".xls", ".csv", ".tsv", ".json", ".jsonl",
                            ".yaml", ".yml", ".xml", ".parquet"},
    "Presentations": {".pptx", ".ppt", ".key"},
    "Code": {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".h", ".cpp",
             ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt",
             ".sh", ".bash", ".zsh", ".sql", ".r", ".m", ".ipynb"},
    "Web": {".html", ".htm", ".css", ".scss", ".vue", ".svelte"},
    "Config": {".ini", ".cfg", ".conf", ".toml", ".env", ".properties"},
    "Images": {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".svg",
               ".heic"},
    "Audio & Video": {".mp3", ".wav", ".flac", ".aac", ".m4a", ".mp4", ".mov",
                      ".avi", ".mkv", ".webm"},
    "Archives": {".zip", ".tar", ".gz", ".bz2", ".7z", ".rar"},
}

_EXTENSION_TO_CATEGORY = {
    ext: category for category, exts in TYPE_CATEGORIES.items() for ext in exts
}

# Extensions read as plain text, whatever their category.
PLAIN_TEXT_EXTENSIONS = (
    TYPE_CATEGORIES["Text & Markdown"]
    | TYPE_CATEGORIES["Code"]
    | TYPE_CATEGORIES["Config"]
    | {".csv", ".tsv", ".json", ".jsonl", ".yaml", ".yml", ".xml", ".svg",
       ".css", ".scss", ".vue", ".svelte"}
)


def category_for(path):
    """Return the human-readable type category for a file."""
    return _EXTENSION_TO_CATEGORY.get(Path(path).suffix.lower(), "Other")


def supported_extensions():
    """Every extension the library knows how to file, sorted."""
    return sorted(_EXTENSION_TO_CATEGORY)


# ---------------------------------------------------------- text extraction

def _read_plain_text(path):
    """Read a text file, guessing the encoding when UTF-8 fails."""
    raw = Path(path).read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    try:
        import chardet
        guess = chardet.detect(raw[:100_000]).get("encoding")
        if guess:
            return raw.decode(guess, errors="replace")
    except Exception:
        pass
    return raw.decode("utf-8", errors="replace")


def _extract_pdf(path):
    from pypdf import PdfReader
    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _extract_docx(path):
    import docx
    document = docx.Document(path)
    parts = [p.text for p in document.paragraphs]
    # Tables often carry the substance in reports, and are skipped entirely
    # if only paragraphs are read.
    for table in document.tables:
        for row in table.rows:
            parts.append("\t".join(cell.text for cell in row.cells))
    return "\n".join(parts)


def _extract_xlsx(path):
    import openpyxl
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    parts = []
    for sheet in workbook.worksheets:
        parts.append(f"# Sheet: {sheet.title}")
        for row in sheet.iter_rows(values_only=True):
            cells = [str(c) for c in row if c is not None]
            if cells:
                parts.append("\t".join(cells))
    workbook.close()
    return "\n".join(parts)


def _extract_pptx(path):
    from pptx import Presentation
    presentation = Presentation(path)
    parts = []
    for i, slide in enumerate(presentation.slides, 1):
        parts.append(f"# Slide {i}")
        for shape in slide.shapes:
            if shape.has_text_frame:
                parts.append(shape.text_frame.text)
    return "\n".join(parts)


def _extract_html(path):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(_read_plain_text(path), "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def _extract_rtf(path):
    from striprtf.striprtf import rtf_to_text
    return rtf_to_text(_read_plain_text(path), errors="ignore")


def _extract_epub(path):
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
    book = epub.read_epub(path)
    parts = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        parts.append(soup.get_text(separator="\n", strip=True))
    return "\n".join(parts)


def _extract_ipynb(path):
    notebook = json.loads(_read_plain_text(path))
    parts = []
    for cell in notebook.get("cells", []):
        parts.append("".join(cell.get("source", [])))
    return "\n".join(parts)


def _describe_binary(path):
    """Fallback description for files whose bytes carry no readable text.

    Images, audio, video, and archives still deserve a place in the library,
    so they are represented by their name and type. Grouping them by content
    then falls back to what the filename says, which is weak but honest — the
    type grouping is the meaningful one for these.
    """
    p = Path(path)
    words = p.stem.replace("_", " ").replace("-", " ")
    description = f"{words}. {category_for(path)} file, {p.suffix.lower()} format."

    if category_for(path) == "Images":
        try:
            from PIL import Image
            with Image.open(path) as img:
                description += f" Image, {img.width}x{img.height} pixels, mode {img.mode}."
        except Exception:
            pass
    return description


_EXTRACTORS = {
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
    ".xlsx": _extract_xlsx, ".xlsm": _extract_xlsx,
    ".pptx": _extract_pptx,
    ".html": _extract_html, ".htm": _extract_html,
    ".rtf": _extract_rtf,
    ".epub": _extract_epub,
    ".ipynb": _extract_ipynb,
}


def _normalise_whitespace(text):
    """Tidy whitespace while keeping line breaks.

    Collapsing everything onto one line would be simpler, but markdown
    headings are what the search chunker splits on, so the newlines have to
    survive extraction.
    """
    lines = [" ".join(line.split()) for line in text.splitlines()]
    cleaned, blank = [], False
    for line in lines:
        if line:
            cleaned.append(line)
            blank = False
        elif not blank:
            cleaned.append("")
            blank = True
    return "\n".join(cleaned).strip()


def extract_text(path):
    """Pull readable text out of a file. Returns (text, status).

    Status is 'ok', 'empty' when the file parsed but held no text (a scanned
    PDF, for instance), 'described' for binary formats represented by their
    metadata, or an error message.
    """
    suffix = Path(path).suffix.lower()

    try:
        if suffix in _EXTRACTORS:
            text = _EXTRACTORS[suffix](path)
        elif suffix in PLAIN_TEXT_EXTENSIONS:
            text = _read_plain_text(path)
        else:
            return _describe_binary(path), "described"
    except Exception as e:
        # A corrupt or password-protected file should not take the batch down.
        return _describe_binary(path), f"could not read ({type(e).__name__})"

    text = _normalise_whitespace(text)
    if len(text) < MIN_USEFUL_CHARS:
        return _describe_binary(path), "empty"
    return text[:MAX_EXTRACT_CHARS], "ok"


# ------------------------------------------------------------- the library

@dataclass
class LibraryFile:
    path: str
    name: str
    category: str
    size: int
    added: str
    status: str = "ok"
    text: str = field(default="", repr=False)

    @property
    def extension(self):
        return Path(self.name).suffix.lower() or "(none)"

    @property
    def size_label(self):
        size = float(self.size)
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024 or unit == "GB":
                return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
            size /= 1024


class FileLibrary:
    """Holds uploaded files, their extracted text, and their embeddings."""

    def __init__(self, directory, model=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.files = []
        self._embeddings = None

    # -- contents ----------------------------------------------------------

    def load_existing(self):
        """Pick up files already sitting in the library folder."""
        self.files = []
        for path in sorted(self.directory.iterdir()):
            if path.is_file() and not path.name.startswith("."):
                self._register(path)
        self._embeddings = None
        return len(self.files)

    def add(self, source_paths):
        """Copy files into the library and extract their text.

        Returns (added, skipped) so the UI can report duplicates.
        """
        added, skipped = [], []
        for source in source_paths or []:
            source = Path(source)
            if not source.is_file():
                skipped.append((source.name, "not a file"))
                continue

            destination = self._unique_destination(source.name)
            try:
                shutil.copy2(source, destination)
            except OSError as e:
                skipped.append((source.name, str(e)))
                continue

            added.append(self._register(destination))

        self._embeddings = None
        return added, skipped

    def _unique_destination(self, name):
        """Avoid silently overwriting a file already in the library."""
        destination = self.directory / name
        if not destination.exists():
            return destination
        stem, suffix = Path(name).stem, Path(name).suffix
        for n in range(2, 1000):
            candidate = self.directory / f"{stem} ({n}){suffix}"
            if not candidate.exists():
                return candidate
        return self.directory / f"{stem}-{datetime.now():%H%M%S}{suffix}"

    def _register(self, path):
        text, status = extract_text(path)
        entry = LibraryFile(
            path=str(path),
            name=path.name,
            category=category_for(path),
            size=path.stat().st_size,
            added=datetime.now().strftime("%Y-%m-%d %H:%M"),
            status=status,
            text=text,
        )
        self.files.append(entry)
        return entry

    def remove(self, names):
        """Delete named files from the library folder."""
        targets = set(names or [])
        kept, removed = [], 0
        for entry in self.files:
            if entry.name in targets:
                try:
                    os.remove(entry.path)
                    removed += 1
                    continue
                except OSError:
                    pass
            kept.append(entry)
        self.files = kept
        self._embeddings = None
        return removed

    def clear(self):
        count = self.remove([f.name for f in self.files])
        return count

    # -- organisation ------------------------------------------------------

    def embeddings(self):
        """Embed every file's extracted text, cached until the set changes."""
        if self._embeddings is not None or not self.files:
            return self._embeddings
        if self.model is None:
            raise RuntimeError("FileLibrary needs an embedding model to group by content.")

        vectors = self.model.encode([f.text for f in self.files], convert_to_numpy=True)
        # Normalising lets cosine distance be computed as a plain dot product,
        # and is what the clustering threshold is calibrated against.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self._embeddings = vectors / np.clip(norms, 1e-9, None)
        return self._embeddings

    def group_by_type(self):
        """Group files by what they are. Returns [(label, [files])]."""
        groups = defaultdict(list)
        for entry in self.files:
            groups[entry.category].append(entry)

        ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        return [
            (f"{label} ({', '.join(sorted({f.extension for f in items}))})",
             sorted(items, key=lambda f: f.name.lower()))
            for label, items in ordered
        ]

    def group_by_content(self, distance=DEFAULT_CLUSTER_DISTANCE):
        """Group files by subject matter. Returns [(label, [files])].

        Uses agglomerative clustering with a distance threshold rather than a
        fixed number of clusters, because the right number of topics in a pile
        of dropped files is not known ahead of time — it depends entirely on
        what was dropped.
        """
        if not self.files:
            return []
        if len(self.files) == 1:
            return [(self._label([0]), list(self.files))]

        from sklearn.cluster import AgglomerativeClustering

        vectors = self.embeddings()
        labels = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance,
            metric="cosine",
            linkage="average",
        ).fit_predict(vectors)

        clusters = defaultdict(list)
        for index, label in enumerate(labels):
            clusters[label].append(index)

        groups = []
        for indices in clusters.values():
            # Order within a cluster by closeness to its centre, so the most
            # representative file of each topic is listed first.
            centroid = vectors[indices].mean(axis=0)
            centroid /= max(float(np.linalg.norm(centroid)), 1e-9)
            ranked = sorted(indices, key=lambda i: -float(vectors[i] @ centroid))
            cohesion = float(np.mean([vectors[i] @ centroid for i in ranked]))
            groups.append((self._label(ranked), [self.files[i] for i in ranked], cohesion))

        # Biggest, most coherent topics first; loose odds-and-ends last.
        groups.sort(key=lambda g: (-len(g[1]), -g[2]))
        return [(label, items) for label, items, _ in groups]

    def _label(self, indices):
        """Name a cluster from the words that distinguish it from the rest."""
        if len(self.files) < 3:
            return self._fallback_label(indices)

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
            matrix = vectorizer.fit_transform([f.text for f in self.files])
            vocabulary = np.array(vectorizer.get_feature_names_out())

            inside = np.asarray(matrix[indices].mean(axis=0)).ravel()
            outside_idx = [i for i in range(len(self.files)) if i not in set(indices)]
            outside = (np.asarray(matrix[outside_idx].mean(axis=0)).ravel()
                       if outside_idx else np.zeros_like(inside))

            # Subtracting the rest of the corpus keeps the label on what makes
            # this group different, not on words every file happens to share.
            distinctive = inside - outside
            # Bare numbers score highly in spreadsheets but say nothing about
            # the subject, so a label term has to contain actual letters.
            ranked = vocabulary[np.argsort(distinctive)[::-1]]
            terms = [t for t in ranked
                     if len(t) > 2 and any(ch.isalpha() for ch in t)][:3]
            if terms:
                return ", ".join(terms)
        except Exception:
            pass
        return self._fallback_label(indices)

    def _fallback_label(self, indices):
        categories = {self.files[i].category for i in indices}
        return " / ".join(sorted(categories)) if categories else "Ungrouped"

    # -- search integration -------------------------------------------------

    def as_search_sources(self):
        """Paths whose text the search engine can index directly."""
        return [f.path for f in self.files if f.status == "ok"]
