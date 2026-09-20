"""Semantic document search over the hub's own documentation.

Indexes markdown and text documents twice: as dense sentence embeddings in a
FAISS vector index, and as a sparse TF-IDF matrix. That makes it possible to
compare meaning-based retrieval against classic keyword matching, or to fuse
the two.
"""

import glob
import os
import re

import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from file_library import LIBRARY_DIRNAME, extract_text

DEFAULT_MODEL = 'all-MiniLM-L6-v2'

# Chunk sizing. Long documents produce "diluted" embeddings that average away
# every specific concept, so text is split into passages. The overlap keeps a
# sentence that straddles a boundary retrievable from either side.
CHUNK_WORDS = 120
CHUNK_OVERLAP_WORDS = 30

SEARCH_METHODS = ["Semantic", "Keyword", "Hybrid"]

# Weight given to the semantic ranking when fusing the two result lists.
HYBRID_SEMANTIC_WEIGHT = 0.65

_HEADING_RE = re.compile(r'^(#{1,6})\s+(.*)$')


def project_readmes(base_dir):
    """Locate the four project READMEs that make up the default corpus.

    The engine indexes the live README files rather than copies of them. An
    earlier version kept duplicates inside this folder, which meant every edit
    to a README silently left the search index describing an older version of
    the project.
    """
    candidates = [
        os.path.join(base_dir, "face_anonymizer", "README.md"),
        os.path.join(base_dir, "Depth Map Visualization", "README.md"),
        os.path.join(base_dir, "pose_estimator", "README.md"),
        os.path.join(base_dir, "semantic_search", "README.md"),
        os.path.join(base_dir, "README.md"),
    ]
    return [p for p in candidates if os.path.exists(p)]


class SemanticSearchEngine:
    def __init__(self, sources=None, data_dir=None, model_name=DEFAULT_MODEL, model=None):
        """
        sources:  explicit list of files to index (the project READMEs).
        data_dir: extra folder scanned for documents to index alongside them,
                  including everything in its file library subfolder.
        model:    an already-loaded SentenceTransformer to share, avoiding a
                  second copy of the weights when the file library has one.
        """
        self.model = model or SentenceTransformer(model_name)
        self.sources = list(sources) if sources else []
        self.data_dir = data_dir

        self.documents = []      # the text of each chunk
        self.doc_metadata = []   # {file, title, section, chunk_id} per chunk

        self.faiss_index = None
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None

    # ---------------------------------------------------------------- loading

    def _collect_files(self):
        """Return every file to index, de-duplicated, in a stable order."""
        files = list(self.sources)
        if self.data_dir and os.path.isdir(self.data_dir):
            for pattern in ("*.md", "*.txt"):
                # A leading underscore marks a note about the folder itself
                # rather than a document meant to be searchable.
                files.extend(sorted(
                    p for p in glob.glob(os.path.join(self.data_dir, pattern))
                    if not os.path.basename(p).startswith("_")
                ))

            # Everything dropped into the file library, whatever its format.
            library = os.path.join(self.data_dir, LIBRARY_DIRNAME)
            if os.path.isdir(library):
                files.extend(sorted(
                    p for p in glob.glob(os.path.join(library, "*"))
                    if os.path.isfile(p) and not os.path.basename(p).startswith(".")
                ))

        seen, unique = set(), []
        for path in files:
            key = os.path.realpath(path)
            if key not in seen:
                seen.add(key)
                unique.append(path)
        return unique

    def chunk_text(self, text, max_words=CHUNK_WORDS, overlap=CHUNK_OVERLAP_WORDS):
        """Split markdown into overlapping passages, tagged with their heading.

        Splitting on headings first keeps each passage on a single topic, and
        carrying the heading along means a result can say *which section* it
        came from instead of just quoting a paragraph out of context.
        """
        sections = []
        current_heading, buffer = "Introduction", []

        for line in text.splitlines():
            match = _HEADING_RE.match(line)
            if match:
                if buffer:
                    sections.append((current_heading, "\n".join(buffer)))
                    buffer = []
                current_heading = match.group(2).strip()
            else:
                buffer.append(line)
        if buffer:
            sections.append((current_heading, "\n".join(buffer)))

        chunks = []
        step = max(max_words - overlap, 1)
        for heading, body in sections:
            words = body.split()
            if not words:
                continue
            for start in range(0, len(words), step):
                passage = " ".join(words[start:start + max_words])
                if len(passage.strip()) > 10:
                    chunks.append((heading, passage))
                if start + max_words >= len(words):
                    break
        return chunks

    def load_documents(self):
        """Read, chunk, and record metadata for every source document."""
        self.documents = []
        self.doc_metadata = []

        for filepath in self._collect_files():
            # extract_text handles PDFs, Office documents, HTML and the rest,
            # so anything added to the file library is searchable too, not
            # just plain text and markdown.
            content, status = extract_text(filepath)
            if status != "ok":
                print(f"Skipping {os.path.basename(filepath)}: {status}")
                continue

            # Prefer the document's own H1 as its title, falling back to the
            # folder name, which is more meaningful than four "README.md"s.
            first_heading = next(
                (m.group(2).strip() for m in
                 (_HEADING_RE.match(line) for line in content.splitlines()) if m),
                None,
            )
            folder = os.path.basename(os.path.dirname(os.path.abspath(filepath)))
            basename = os.path.basename(filepath)
            # A README is best identified by its project folder; anything else
            # by its own filename, since "library" tells the reader nothing.
            if first_heading:
                title = first_heading
            elif basename.lower().startswith("readme"):
                title = folder or basename
            else:
                title = os.path.splitext(basename)[0]
            # Every project file is called README.md, so show it with its
            # parent folder to keep the four of them distinguishable.
            display_path = os.path.join(folder, os.path.basename(filepath))

            for i, (section, chunk) in enumerate(self.chunk_text(content)):
                self.documents.append(chunk)
                self.doc_metadata.append({
                    "file": display_path,
                    "title": title,
                    "section": section,
                    "chunk_id": i + 1,
                })

        return len(self.documents)

    def build_indexes(self):
        """Build the FAISS vector index and the TF-IDF keyword matrix."""
        if not self.documents:
            return

        embeddings = self.model.encode(self.documents, convert_to_numpy=True)
        # L2-normalising turns FAISS's inner product into cosine similarity.
        faiss.normalize_L2(embeddings)
        self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        self.faiss_index.add(embeddings)

        self.tfidf_matrix = self.tfidf.fit_transform(self.documents)

    # --------------------------------------------------------------- querying

    def _result(self, idx, score):
        return {
            "index": int(idx),
            "score": float(score),
            "text": self.documents[idx],
            "metadata": self.doc_metadata[idx],
        }

    def _semantic(self, query, top_k):
        query_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        # Asking for more neighbours than exist makes FAISS pad with -1.
        scores, indices = self.faiss_index.search(query_emb, min(top_k, len(self.documents)))
        return [self._result(idx, score)
                for score, idx in zip(scores[0], indices[0]) if idx != -1]

    def _keyword(self, query, top_k):
        query_vec = self.tfidf.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        ranked = similarities.argsort()[::-1][:top_k]
        return [self._result(idx, similarities[idx]) for idx in ranked if similarities[idx] > 0]

    def _hybrid(self, query, top_k):
        """Blend both rankings with reciprocal rank fusion.

        Scores from cosine similarity and TF-IDF are not on a comparable scale,
        so they are fused by *rank* instead of by value. This catches documents
        that one method alone would miss: an exact model name that embeddings
        gloss over, or a paraphrased question that shares no keywords.
        """
        pool = min(top_k * 3, len(self.documents))
        rankings = [
            (HYBRID_SEMANTIC_WEIGHT, self._semantic(query, pool)),
            (1 - HYBRID_SEMANTIC_WEIGHT, self._keyword(query, pool)),
        ]

        fused = {}
        for weight, results in rankings:
            for rank, result in enumerate(results):
                entry = fused.setdefault(result["index"], {"result": result, "score": 0.0})
                entry["score"] += weight / (60 + rank)  # 60 damps the top-rank spike

        best = sorted(fused.values(), key=lambda e: e["score"], reverse=True)[:top_k]
        return [dict(e["result"], score=e["score"]) for e in best]

    def search(self, query, method="semantic", top_k=3):
        """Search the corpus. method is 'semantic', 'keyword', or 'hybrid'."""
        if not self.documents or not query.strip():
            return []

        method = method.lower()
        if method == "semantic":
            return self._semantic(query, top_k)
        if method == "keyword":
            return self._keyword(query, top_k)
        if method == "hybrid":
            return self._hybrid(query, top_k)
        raise ValueError(
            f"Unknown search method {method!r}. Choose one of: {', '.join(SEARCH_METHODS)}"
        )


def build_default_engine(base_dir=None, data_dir=None, model=None):
    """Create an engine indexing the project READMEs plus any extra documents."""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if data_dir is None:
        data_dir = os.path.join(base_dir, "semantic_search", "data")

    engine = SemanticSearchEngine(sources=project_readmes(base_dir), data_dir=data_dir,
                                  model=model)
    engine.load_documents()
    engine.build_indexes()
    return engine


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Semantic document search")
    parser.add_argument("query", nargs="*", help="Search query (omit for interactive mode)")
    parser.add_argument("--method", default="semantic",
                        choices=[m.lower() for m in SEARCH_METHODS])
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    print("Building index...")
    engine = build_default_engine()
    print(f"Indexed {len(engine.documents)} chunks from "
          f"{len({m['file'] for m in engine.doc_metadata})} documents.\n")

    def show(query):
        results = engine.search(query, method=args.method, top_k=args.top_k)
        if not results:
            print("No results found.\n")
            return
        for i, res in enumerate(results, 1):
            meta = res["metadata"]
            print(f"{i}. {meta['title']} > {meta['section']}  (score {res['score']:.3f})")
            print(f"   {res['text'][:280]}...\n")

    if args.query:
        show(" ".join(args.query))
        return

    print("Interactive mode. Type a query, or press Enter to quit.")
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            break
        show(query)


if __name__ == "__main__":
    main()
