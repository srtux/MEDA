"""Durable, local cross-run learning for MEDA: a lessons + skills library.

This is the procedural-memory layer that lets the agent stop repeating
mistakes. It is deliberately **cloud-free**: a single local SQLite file holds
two tables —

* ``lessons``  – (error signature -> root cause -> corrective fix) records,
  born when a failure is later fixed within a run (Reflexion / ExpeL style).
* ``skills``   – parameterized CadQuery snippets that reached reward 1.0,
  retrievable by sub-goal similarity (Voyager / Agent-Workflow-Memory style).

Embeddings (Gemini ``gemini-embedding-001``) power semantic retrieval when a
genai client + API key are available. The store degrades gracefully to keyword
+ signature matching when embeddings are unavailable (e.g. an offline seed DB,
or no API key), so a committed seed database with NULL embeddings still works
and is back-filled lazily on first use.

References: Voyager (arXiv:2305.16291), Reflexion (arXiv:2303.11366),
Generative Agents (arXiv:2304.03442), ExpeL (arXiv:2308.10144),
Agent Workflow Memory (arXiv:2409.07429).
"""

from __future__ import annotations

import math
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

_DB_DEFAULT = Path("memory/meda_memory.db")
_EMBED_MODEL = "gemini-embedding-001"
_EMBED_DIM = 768            # Matryoshka-truncated; manual L2-normalization applied
_CONF_FLOOR = 0.4          # retrieval confidence threshold
_DECAY_LAMBDA = 0.005      # per-day exponential decay on effective confidence
_DEDUP_SIM = 0.92          # cosine above which two lessons are merged, not duplicated
_SIG_BOOST = 0.25          # similarity bump for an exact signature-class match
_MAX_ROWS = 2000           # per-table soft cap before eviction


def _now() -> float:
    return time.time()


class LearningStore:
    """SQLite-backed lessons + skills library with optional embedding retrieval."""

    def __init__(self, genai_client: Any = None, db_path: Path | str = _DB_DEFAULT):
        self.client = genai_client
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(self.db_path), check_same_thread=False)
        try:
            self.con.execute("PRAGMA journal_mode=WAL")
        except sqlite3.Error:
            pass
        self._init_schema()

    # ------------------------------------------------------------------ schema
    def _init_schema(self) -> None:
        self.con.execute(
            """CREATE TABLE IF NOT EXISTS lessons(
                id TEXT PRIMARY KEY,
                error_signature TEXT,
                error_detail TEXT,
                root_cause TEXT,
                corrective_fix TEXT,
                prompt_context TEXT,
                embedding BLOB,
                confidence REAL,
                uses INTEGER,
                hits INTEGER,
                misses INTEGER,
                created_ts REAL,
                last_used_ts REAL)"""
        )
        self.con.execute(
            """CREATE TABLE IF NOT EXISTS skills(
                id TEXT PRIMARY KEY,
                name TEXT,
                goal_description TEXT,
                signature TEXT,
                code_template TEXT,
                embedding BLOB,
                success_count INTEGER,
                confidence REAL,
                created_ts REAL,
                last_used_ts REAL)"""
        )
        self.con.commit()

    # --------------------------------------------------------------- embeddings
    def _embed(self, text: str, query: bool = False):
        """Return a unit-normalized embedding vector, or ``None`` if unavailable."""
        if not self.client or not text:
            return None
        try:
            import numpy as np
            from google.genai import types

            cfg = types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY" if query else "RETRIEVAL_DOCUMENT",
                output_dimensionality=_EMBED_DIM,
            )
            resp = self.client.models.embed_content(
                model=_EMBED_MODEL, contents=text, config=cfg
            )
            vec = np.asarray(resp.embeddings[0].values, dtype="float32")
            norm = np.linalg.norm(vec)
            return vec / norm if norm else vec
        except Exception:
            return None

    @staticmethod
    def _to_blob(vec) -> Optional[bytes]:
        return vec.tobytes() if vec is not None else None

    @staticmethod
    def _from_blob(blob):
        if not blob:
            return None
        import numpy as np
        return np.frombuffer(blob, dtype="float32")

    @staticmethod
    def _conf_eff(confidence: float, last_used_ts: float) -> float:
        age_days = max(0.0, (_now() - last_used_ts) / 86400.0)
        return confidence * math.exp(-_DECAY_LAMBDA * age_days)

    @staticmethod
    def _keyword_overlap(a: str, b: str) -> float:
        """Cheap fallback similarity when embeddings are unavailable."""
        ta = {w for w in a.lower().split() if len(w) > 3}
        tb = {w for w in b.lower().split() if len(w) > 3}
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    # -------------------------------------------------------------- lessons I/O
    def record_lesson(
        self,
        error_signature: str,
        error_detail: str,
        root_cause: str,
        corrective_fix: str,
        prompt_context: str,
    ) -> str:
        """Insert a lesson, or merge into an existing near-duplicate."""
        emb = self._embed(f"{error_signature}\n{prompt_context}\n{root_cause}")

        # Dedup: same signature + (embedding near-dup OR no-embedding fallback).
        for row in self.con.execute(
            "SELECT id, embedding, prompt_context FROM lessons WHERE error_signature=?",
            (error_signature,),
        ).fetchall():
            other = self._from_blob(row[1])
            is_dup = False
            if emb is not None and other is not None:
                is_dup = float((emb * other).sum()) >= _DEDUP_SIM
            else:
                is_dup = self._keyword_overlap(prompt_context, row[2] or "") >= 0.6
            if is_dup:
                self.con.execute(
                    "UPDATE lessons SET uses=uses+1, last_used_ts=?, corrective_fix=? WHERE id=?",
                    (_now(), corrective_fix, row[0]),
                )
                self.con.commit()
                return row[0]

        lid = str(uuid.uuid4())
        now = _now()
        self.con.execute(
            "INSERT INTO lessons VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (lid, error_signature, error_detail, root_cause, corrective_fix,
             prompt_context, self._to_blob(emb), 0.5, 0, 0, 0, now, now),
        )
        self.con.commit()
        self._evict_if_needed("lessons")
        return lid

    def retrieve_lessons(
        self, query_text: str, k: int = 3, signature: Optional[str] = None,
        require_proven: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return up to ``k`` relevant lessons ranked by similarity * effective confidence.

        ``require_proven`` filters to lessons that helped at least once (uses>=1),
        used for run-start injection to avoid surfacing unvetted one-offs.
        """
        q = self._embed(query_text, query=True)
        rows = self.con.execute(
            "SELECT id,error_signature,error_detail,root_cause,corrective_fix,"
            "embedding,confidence,uses,last_used_ts,prompt_context FROM lessons"
        ).fetchall()
        scored = []
        for r in rows:
            conf_eff = self._conf_eff(r[6], r[8])
            if conf_eff < _CONF_FLOOR:
                continue
            if require_proven and (r[7] or 0) < 1:
                continue
            other = self._from_blob(r[5])
            if q is not None and other is not None:
                sim = float((q * other).sum())
            else:
                sim = self._keyword_overlap(query_text, r[9] or "")
            if signature and r[1] == signature:
                sim += _SIG_BOOST
            scored.append((sim * conf_eff, r))
        scored.sort(key=lambda x: -x[0])
        out = []
        for _, r in scored[:k]:
            out.append({
                "id": r[0], "error_signature": r[1], "error_detail": r[2],
                "root_cause": r[3], "corrective_fix": r[4],
            })
        return out

    def feedback(self, lesson_id: str, helped: bool) -> None:
        """Update a lesson's Laplace-smoothed confidence after it was applied."""
        col = "hits" if helped else "misses"
        self.con.execute(f"UPDATE lessons SET {col}={col}+1 WHERE id=?", (lesson_id,))
        row = self.con.execute(
            "SELECT hits,misses FROM lessons WHERE id=?", (lesson_id,)
        ).fetchone()
        if row:
            h, m = row
            conf = (h + 1) / (h + m + 2)
            self.con.execute(
                "UPDATE lessons SET confidence=?, last_used_ts=? WHERE id=?",
                (conf, _now(), lesson_id),
            )
            self.con.commit()

    # --------------------------------------------------------------- skills I/O
    def record_skill(
        self, name: str, goal_description: str, signature: str, code_template: str,
    ) -> str:
        """Insert a reusable skill, or bump an existing one with the same name/goal."""
        emb = self._embed(goal_description)
        for row in self.con.execute(
            "SELECT id, embedding, goal_description FROM skills WHERE name=?",
            (name,),
        ).fetchall():
            other = self._from_blob(row[1])
            is_dup = False
            if emb is not None and other is not None:
                is_dup = float((emb * other).sum()) >= _DEDUP_SIM
            else:
                is_dup = self._keyword_overlap(goal_description, row[2] or "") >= 0.6
            if is_dup:
                self.con.execute(
                    "UPDATE skills SET success_count=success_count+1, last_used_ts=?,"
                    " code_template=? WHERE id=?",
                    (_now(), code_template, row[0]),
                )
                self.con.commit()
                return row[0]

        sid = str(uuid.uuid4())
        now = _now()
        self.con.execute(
            "INSERT INTO skills VALUES(?,?,?,?,?,?,?,?,?,?)",
            (sid, name, goal_description, signature, code_template,
             self._to_blob(emb), 1, 0.6, now, now),
        )
        self.con.commit()
        self._evict_if_needed("skills")
        return sid

    def retrieve_skills(self, query_text: str, k: int = 3) -> List[Dict[str, Any]]:
        q = self._embed(query_text, query=True)
        rows = self.con.execute(
            "SELECT id,name,goal_description,signature,code_template,embedding,"
            "confidence,success_count,last_used_ts FROM skills"
        ).fetchall()
        scored = []
        for r in rows:
            conf_eff = self._conf_eff(r[6], r[8])
            other = self._from_blob(r[5])
            if q is not None and other is not None:
                sim = float((q * other).sum())
            else:
                sim = self._keyword_overlap(query_text, r[2] or "")
            scored.append((sim * max(conf_eff, 0.1), r))
        scored.sort(key=lambda x: -x[0])
        out = []
        for score, r in scored[:k]:
            if score <= 0:
                continue
            out.append({
                "id": r[0], "name": r[1], "goal_description": r[2],
                "signature": r[3], "code_template": r[4],
            })
        return out

    # ---------------------------------------------------------------- eviction
    def _evict_if_needed(self, table: str) -> None:
        count = self.con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if count <= _MAX_ROWS:
            return
        # Evict lowest effective-confidence rows.
        rows = self.con.execute(
            f"SELECT id, confidence, last_used_ts FROM {table}"
        ).fetchall()
        rows.sort(key=lambda r: self._conf_eff(r[1], r[2]))
        to_drop = count - _MAX_ROWS
        for rid, _, _ in rows[:to_drop]:
            self.con.execute(f"DELETE FROM {table} WHERE id=?", (rid,))
        self.con.commit()

    def counts(self) -> Dict[str, int]:
        return {
            "lessons": self.con.execute("SELECT COUNT(*) FROM lessons").fetchone()[0],
            "skills": self.con.execute("SELECT COUNT(*) FROM skills").fetchone()[0],
        }

    def close(self) -> None:
        try:
            self.con.close()
        except sqlite3.Error:
            pass
