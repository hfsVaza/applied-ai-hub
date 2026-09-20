# Project 4: Semantic Document Search Engine

A mini-application that indexes a collection of documents and searches them by
*meaning* rather than by exact keyword. This project fulfils the "Semantic
Search" learning goals and practical requirements for the Applied AI
Programming course.

## Overview

The engine loads markdown documents, splits them into overlapping passages,
and embeds each passage with a HuggingFace sentence-transformer. The
embeddings live in a **FAISS vector index**. A query is embedded into the same
vector space, and FAISS returns the passages with the highest cosine
similarity.

Three retrieval modes are available so their behaviour can be compared
directly:

| Mode | How it ranks | Best at |
|---|---|---|
| **Semantic** | Cosine similarity between dense embeddings | Paraphrases, vague questions, synonyms |
| **Keyword** | TF-IDF cosine similarity | Exact identifiers, model names, acronyms |
| **Hybrid** | Reciprocal rank fusion of both | General use; catches what either alone misses |

**Meta-feature:** the corpus is the documentation for this very hub. The engine
indexes the four project READMEs **live, in place**, so the index can never
describe an out-of-date version of the documentation. Anything added through
the Document Library is indexed alongside them.

## The Document Library

The **Document Library** tab is a drop-anything pile: upload any number of
mixed files and have them organised for you. Text is extracted, embedded, and
added to the search index, so an uploaded PDF or spreadsheet becomes
searchable immediately.

**Sort by content similarity** clusters files by what they are *about*,
ignoring format entirely. A pile of ML notes gets grouped together whether
they arrived as `.md`, `.pptx`, `.py`, `.html` or `.rtf`. Each group is
labelled with the words that distinguish it from the rest of the pile, and the
most representative file in each group is listed first.

**Sort by file type** groups by what the files *are*, which is the useful view
for formats with no readable text.

**Grouping breadth** controls how aggressively topics merge. Lower splits the
pile into more, tighter groups; higher merges them into fewer, broader ones.

### Supported formats

| Extracted | Formats |
|---|---|
| Full text | `.pdf`, `.docx`, `.xlsx`, `.pptx`, `.html`, `.epub`, `.rtf`, `.ipynb` |
| Plain text | `.txt`, `.md`, `.csv`, `.tsv`, `.json`, `.yaml`, `.xml`, and 25+ code and config extensions |
| Name and type only | Images, audio, video, archives, and anything else |

Extraction is deliberately forgiving. A corrupt, encrypted, or scanned file
does not fail the batch: it falls back to being filed by name and type, and
the UI says which files that happened to, so a silently empty result is never
mistaken for a successful one. Word tables and every sheet of a workbook are
included, because in real documents that is often where the substance is.

## Running it

Inside the hub, open the **Semantic Search** project. Standalone:

```bash
python semantic_search/search_engine.py "how do I blur a face"
python semantic_search/search_engine.py --method hybrid --top-k 5 "3d point cloud export"
python semantic_search/search_engine.py          # interactive mode
```

## Report & Answers to Evaluation Criteria

### 1. What embedding model and libraries were used?
- **Embedding model:** `all-MiniLM-L6-v2` via `sentence-transformers`. It
  produces 384-dimensional vectors and is small and fast enough to index the
  whole corpus in about a second on a laptop CPU.
- **Vector index:** `faiss-cpu`, using `IndexFlatIP`. Flat means exhaustive:
  every vector is compared, so results are exact rather than approximate. At
  this corpus size (tens of chunks) an approximate index such as IVF or HNSW
  would add complexity and recall loss for no measurable speed gain.
- **Keyword baseline:** `scikit-learn`'s `TfidfVectorizer`, to demonstrate
  concretely where exact matching succeeds and where it fails.

### 2. How were documents prepared and chunked?
Long documents produce "diluted" embeddings: a single vector asked to
represent an entire README averages every concept in it and matches nothing
sharply. Text is therefore chunked, in two stages:

1. **Split on markdown headings first.** This keeps each passage on one topic
   and preserves the heading as metadata, so a result can report *which
   section* it came from rather than quoting a paragraph out of context.
2. **Then split long sections into ~120-word passages with a 30-word
   overlap.** The overlap matters: without it, a sentence that happens to
   straddle a chunk boundary is split across two vectors and retrieved well by
   neither. The overlap guarantees every span of text appears intact in at
   least one chunk.

Each chunk carries its source file, document title, section heading, and index.

### 3. How was similarity calculated and ranked?
- **Semantic:** vectors are L2-normalised, then compared by inner product,
  which for normalised vectors is mathematically identical to **cosine
  similarity**. A score near 1.0 means near-identical meaning.
- **Keyword:** query and documents are mapped into a sparse bag-of-words TF-IDF
  space and compared by cosine similarity, based purely on word overlap.
- **Hybrid:** the two score scales are not comparable, so a raw weighted sum
  would be meaningless. The modes are fused by **rank** instead, using
  reciprocal rank fusion: each result contributes `weight / (60 + rank)` from
  each ranking it appears in. The constant 60 is the standard damping term; it
  stops a single first-place hit from dominating the fused ordering.

### 4. What kinds of queries work well vs. are difficult?
- **Semantic search works well with:** vague natural-language questions ("how
  do I blur a video?"), concept searches, and synonyms. It correctly connects
  "blur" to "anonymize" even when the exact word never appears.
- **Keyword search works well with:** exact identifiers, model names, code
  symbols, and acronyms, which embeddings tend to smooth over.
- **Semantic search struggles with:** queries that demand exact precision.
  Searching for `Depth-Anything-V2-Small-hf` may surface passages about other
  models, because all model names occupy a similar region of the vector space.
  This is exactly the failure that the keyword and hybrid modes exist to show.
- **All modes struggle with:** questions whose answer is not in the corpus at
  all. The engine always returns its closest matches, so a confident-looking
  result is not evidence that an answer exists. Scores are shown alongside every
  result so a weak match is visible as a weak match.

### 5. Privacy, Copyright, and Data Quality
**Privacy.** The MiniLM model is downloaded once and then run locally under
PyTorch, so **no document text is ever sent to OpenAI, Google, or any cloud
API**. This architecture could index confidential medical or corporate
documents without a compliance breach, which is the main practical argument for
a local embedding model over a hosted embeddings API.

**Copyright.** The corpus is self-authored project documentation, which avoids
the copyright questions raised by indexing scraped or proprietary text.

**Data quality.** The index is only as good as the corpus. Because the engine
reads the live READMEs, fixing an error in the documentation fixes the search
result too, on the next start. The earlier design kept duplicate copies of the
READMEs in `data/`, which meant the two could silently disagree; that
duplication has been removed for exactly this reason.

**Uploaded content.** Files added to the library are copied into
`semantic_search/data/library/` and stay on disk until cleared. They never
leave the machine, but they are readable by anyone with access to the folder,
so treat the library as no more private than the computer it runs on. Clearing
the library deletes the copies and shrinks the index back.
