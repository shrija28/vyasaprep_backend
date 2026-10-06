"""FAISS vector stores for the RAG pipeline.

Per-subject isolation contract (REQ-5.1, REQ-8.5, design.md §2 / §2.1 / §2.2):

    Uploads scoped to a subject MUST never disturb any other subject's
    FAISS index or chunk list.  ``SubjectVectorStores`` enforces this by
    keeping a separate :class:`VectorStore` per :class:`~smartkcet.db.models.Subject`
    and persisting each one to its own pair of files under
    ``backend/data/faiss/{subject}.index`` (FAISS binary) and
    ``backend/data/faiss/{subject}.chunks.json`` (JSON list of chunk
    strings).

The ``embedder`` (``fastembed`` MiniLM L6 v2) is shared across all
subjects since it is a stateless encoder.  Only the FAISS index and the
parallel ``chunks`` list are per-subject.

fastembed uses ONNX Runtime instead of PyTorch, keeping the deployment
bundle lightweight (~100 MB vs ~3 GB for sentence-transformers + torch).

NOTE: Python 3.14 compatibility
-------

Embedder initialization is deferred until first use via a lazy loader.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import faiss

from ..db.models import Subject

# Lazy embedder initialization — deferred until first use.
# Uses fastembed (ONNX Runtime) instead of sentence-transformers (PyTorch)
# to keep the deployment bundle under 500 MB.
_embedder: Optional[object] = None
_embedder_loading_attempted = False

# fastembed model name for all-MiniLM-L6-v2 (384-dim, same as before)
_FASTEMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _get_embedder():
    """Lazy load the fastembed TextEmbedding model on first use."""
    global _embedder, _embedder_loading_attempted

    if _embedder is not None:
        return _embedder

    if _embedder_loading_attempted and _embedder is None:
        # Already tried and failed — don't retry
        raise RuntimeError(
            "fastembed not available. "
            "Embedding/FAISS functionality will not work."
        )

    _embedder_loading_attempted = True
    try:
        from fastembed import TextEmbedding
        _embedder = TextEmbedding(model_name=_FASTEMBED_MODEL)
        return _embedder
    except Exception as e:
        raise RuntimeError(f"Failed to load fastembed model: {e}")


class _EmbedderProxy:
    """Proxy that lazy-loads the fastembed embedder on first access.

    fastembed's ``embed()`` returns a generator of numpy arrays (one per
    text).  This proxy collects them into a single 2-D float32 array so
    the rest of the codebase can call ``.encode()`` exactly as before.
    """

    def encode(self, texts, show_progress_bar: bool = False, **kwargs):
        import numpy as np
        model = _get_embedder()
        # fastembed.TextEmbedding.embed() accepts an iterable and yields
        # one numpy array per input text.
        embeddings = list(model.embed(texts))
        return np.array(embeddings, dtype="float32")


embedder = _EmbedderProxy()

# ``backend/data/faiss/`` resolved relative to the backend root, mirroring
# the path-resolution pattern used by ``smartkcet.db.session``.
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_FAISS_DIR = _BACKEND_ROOT / "data" / "faiss"
_SUPABASE_BUCKET = os.getenv("SUPABASE_FAISS_BUCKET", "vyasaprep-faiss")
_supabase_client: Optional[object] = None
_supabase_load_attempted = False

# Type alias for inputs that select a subject.  Callers may pass either a
# ``Subject`` enum value or its string name; both are normalised internally.
SubjectLike = Union[Subject, str]


def _get_supabase_client():
    """Return a configured Supabase client, or ``None`` when disabled."""

    global _supabase_client, _supabase_load_attempted
    if _supabase_client is not None:
        return _supabase_client
    if _supabase_load_attempted:
        return None

    _supabase_load_attempted = True
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        return None

    try:
        from supabase import create_client
        _supabase_client = create_client(url, key)
    except Exception:
        _supabase_client = None
    return _supabase_client


class VectorStore:
    """In-memory FAISS L2 index plus the original chunk text.

    This class is the per-subject building block used by
    :class:`SubjectVectorStores`.  It carries no knowledge of which
    subject it represents — that mapping is owned by the parent store.
    """

    def __init__(self)-> None:
        self.index = None  # ``faiss.IndexFlatL2`` once initialised
        self.chunks: List[str] = []
        self.dim = 384

    def reset(self)-> None:
        self.index = faiss.IndexFlatL2(self.dim)
        self.chunks = []

    def add(self, texts: Iterable[str])-> None:
        if self.index is None:
            self.reset()
        texts = list(texts)
        if not texts:
            return
        vecs = embedder.encode(texts, show_progress_bar=False).astype("float32")
        self.index.add(vecs)
        self.chunks.extend(texts)

    def search(self, query: str, k: int = 20)-> List[str]:
        if not self.chunks:
            return []
        vec = embedder.encode([query]).astype("float32")
        k = min(k, len(self.chunks))
        _, ids = self.index.search(vec, k)
        return [self.chunks[i] for i in ids[0] if i < len(self.chunks)]


class SubjectVectorStores:
    """Per-subject FAISS stores with lazy load + on-write persistence.

    Each :class:`Subject` gets its own :class:`VectorStore`.  Mutations
    (``add``/``reset``) are scoped strictly to the requested subject; no
    code path here ever touches another subject's index or chunk list,
    which preserves the isolation contract from REQ-5.1.

    Persistence layout under ``self.data_dir`` (default
    ``backend/data/faiss/``):

        Biology.index, Biology.chunks.json
        Physics.index, Physics.chunks.json
        Chemistry.index, Chemistry.chunks.json
        Mathematics.index, Mathematics.chunks.json
    """

    def __init__(self, data_dir: Path | None = None)-> None:
        self.data_dir = data_dir if data_dir is not None else _DEFAULT_FAISS_DIR
        self._stores: Dict[Subject, VectorStore] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(subject: SubjectLike)-> Subject:
        """Accept a ``Subject`` enum or its string name, return the enum."""

        if isinstance(subject, Subject):
            return subject
        if isinstance(subject, str):
            try:
                return Subject(subject)
            except ValueError as exc:  # pragma: no cover - defensive
                raise ValueError(
                    f"Unknown subject {subject!r}; expected one of "
                    f"{[s.value for s in Subject]}"
                ) from exc
        raise TypeError(
            f"subject must be Subject or str, got {type(subject).__name__}"
        )

    def _index_path(self, subject: Subject)-> Path:
        return self.data_dir / f"{subject.value}.index"

    def _chunks_path(self, subject: Subject)-> Path:
        return self.data_dir / f"{subject.value}.chunks.json"

    def _ensure_data_dir(self)-> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _remote_paths(subject: Subject)-> tuple[str, str]:
        prefix = subject.value
        return f"{prefix}.index", f"{prefix}.chunks.json"

    def _load_local(self, subject: Subject)-> Optional[VectorStore]:
        idx_path = self._index_path(subject)
        chunks_path = self._chunks_path(subject)
        if not (idx_path.exists() and chunks_path.exists()):
            return None

        try:
            vs = VectorStore()
            vs.index = faiss.read_index(str(idx_path))
            with chunks_path.open("r", encoding="utf-8") as fp:
                chunks = json.load(fp)
            if not isinstance(chunks, list) or vs.index.ntotal != len(chunks):
                return None
            vs.chunks = [str(c) for c in chunks]
            return vs
        except (OSError, ValueError, RuntimeError):
            return None

    def _load_remote(self, subject: Subject)-> Optional[VectorStore]:
        client = _get_supabase_client()
        if client is None:
            return None

        index_name, chunks_name = self._remote_paths(subject)
        try:
            storage = client.storage.from_(_SUPABASE_BUCKET)
            index_bytes = storage.download(index_name)
            chunks_bytes = storage.download(chunks_name)
            with tempfile.TemporaryDirectory() as temp_dir:
                index_path = Path(temp_dir) / index_name
                index_path.write_bytes(index_bytes)
                vs = VectorStore()
                vs.index = faiss.read_index(str(index_path))
            chunks = json.loads(chunks_bytes.decode("utf-8"))
            if not isinstance(chunks, list) or vs.index.ntotal != len(chunks):
                return None
            vs.chunks = [str(c) for c in chunks]
            return vs
        except Exception:
            return None

    def _load(self, subject: Subject)-> VectorStore:
        """Read the persisted index + chunks for ``subject`` from disk.

        Returns a fresh :class:`VectorStore` populated from disk if both
        files exist; otherwise returns a fresh empty store.
        """

        return self._load_local(subject) or self._load_remote(subject) or VectorStore()

    def _persist(self, subject: Subject, vs: VectorStore)-> None:
        """Persist a subject locally, and to Supabase when configured."""

        client = _get_supabase_client()
        if client is None:
            self._ensure_data_dir()
            if vs.index is not None:
                faiss.write_index(vs.index, str(self._index_path(subject)))
            with self._chunks_path(subject).open("w", encoding="utf-8") as fp:
                json.dump(vs.chunks, fp, ensure_ascii=False)
            return

        index_name, chunks_name = self._remote_paths(subject)
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = Path(temp_dir) / index_name
            faiss.write_index(vs.index, str(index_path))
            chunks_bytes = json.dumps(vs.chunks, ensure_ascii=False).encode("utf-8")
            storage = client.storage.from_(_SUPABASE_BUCKET)
            options = {"upsert": "true"}
            storage.upload(index_name, index_path.read_bytes(), options)
            storage.upload(chunks_name, chunks_bytes, options)

    def _get(self, subject: Subject)-> VectorStore:
        """Return the cached store for ``subject``, lazy-loading on miss."""

        vs = self._stores.get(subject)
        if vs is None:
            vs = self._load(subject)
            self._stores[subject] = vs
        return vs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, subject: SubjectLike, texts: Iterable[str])-> None:
        """Append ``texts`` to ``subject``'s index. Other subjects untouched."""

        s = self._normalize(subject)
        vs = self._get(s)
        before = len(vs.chunks)
        vs.add(texts)
        # Skip the disk write if nothing changed (empty ``texts``) so we
        # don't churn timestamps on no-op calls.
        if len(vs.chunks) != before:
            self._persist(s, vs)

    def search(self, subject: SubjectLike, query: str, k: int = 20)-> List[str]:
        """Search only ``subject``'s index; never reads other subjects."""

        s = self._normalize(subject)
        return self._get(s).search(query, k)

    def reset(self, subject: SubjectLike)-> None:
        """Clear ``subject``'s in-memory state and remove its persisted files."""

        s = self._normalize(subject)
        vs = self._get(s)
        vs.reset()
        client = _get_supabase_client()
        if client is not None:
            try:
                client.storage.from_(_SUPABASE_BUCKET).remove(list(self._remote_paths(s)))
            except Exception:
                pass
        for path in (self._index_path(s), self._chunks_path(s)):
            path.unlink(missing_ok=True)

    def reset_all(self)-> None:
        """Clear every subject's state. Useful for tests and admin reset."""

        for s in Subject:
            self.reset(s)

    def chunk_count(self, subject: SubjectLike)-> int:
        """Return the number of indexed chunks for ``subject``."""

        s = self._normalize(subject)
        return len(self._get(s).chunks)


# Module-level singleton used by the per-subject upload/generate routes
# introduced in tasks 4.3 and 4.5.
stores = SubjectVectorStores()

# Backwards-compat alias for the legacy single-store routes in
# ``smartkcet.routes.legacy``.  Those endpoints predate per-subject
# isolation and operate on a single, in-memory vector store; they will be
# retired in later tasks (5.x / 7.x) when the new admin upload/generate
# endpoints replace them.  Keeping ``store`` as a separate
# :class:`VectorStore` instance is the simplest way to leave that legacy
# path untouched while the new ``stores`` singleton owns all
# subject-scoped state.
store = VectorStore()


__all__ = [
    "embedder",
    "VectorStore",
    "SubjectVectorStores",
    "SubjectLike",
    "stores",
    "store",
]
