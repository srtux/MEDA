"""Lightweight trajectory memory for self-improving CAD generation.

The store is intentionally dependency-free: it persists compact lessons from each
run and retrieves the most relevant lessons for future prompts with simple token
overlap. This gives MEDA a production-safe foothold for self-improvement before
adding embeddings, vector databases, or learned reward models.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


@dataclass
class CADMemory:
    """A reusable lesson extracted from a CAD generation trajectory."""

    prompt: str
    outcome: str
    tip: str
    metrics: Optional[Dict[str, Any]]
    failed_constraints: List[str]
    created_at: float

    @property
    def tokens(self) -> set[str]:
        return _tokenize(" ".join([self.prompt, self.tip, self.outcome]))


def _tokenize(text: str) -> set[str]:
    """Normalize text to a small keyword set for retrieval."""
    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "in", "into", "is", "it", "of", "on", "or", "the", "to", "with",
    }
    return {tok.lower() for tok in _TOKEN_RE.findall(text) if tok.lower() not in stopwords}


class CADMemoryStore:
    """Persist and retrieve compact CAD generation lessons."""

    def __init__(self, path: str | Path = "data/memory/cad_trajectory_memory.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> List[CADMemory]:
        """Load all valid memories, skipping malformed records."""
        if not self.path.exists():
            return []

        memories: List[CADMemory] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                memories.append(CADMemory(**payload))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        return memories

    def retrieve(self, prompt: str, limit: int = 5) -> List[CADMemory]:
        """Retrieve lessons with highest keyword overlap for a new prompt."""
        query_tokens = _tokenize(prompt)
        if not query_tokens:
            return []

        scored = []
        for memory in self.load():
            overlap = len(query_tokens & memory.tokens)
            if overlap:
                scored.append((overlap, memory.created_at, memory))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [memory for _, _, memory in scored[:limit]]

    def append(self, memory: CADMemory) -> None:
        """Append one memory record as JSONL."""
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(memory), sort_keys=True) + "\n")

    def record_run(
        self,
        prompt: str,
        success: bool,
        metrics: Optional[Dict[str, Any]],
        failed_constraints: Iterable[str],
    ) -> CADMemory:
        """Extract and persist a compact learning from one completed run."""
        failures = list(failed_constraints)
        if success:
            tip = (
                "Successful pattern: preserve named parameters, keep features surgical, "
                "and verify B-Rep metrics before visual acceptance. Reuse this structure "
                "for prompts with similar geometry."
            )
            outcome = "success"
        else:
            failure_text = "; ".join(failures) if failures else "unknown failure"
            tip = (
                "Recovery pattern: before adding more features, fix the first reported "
                f"compile/constraint/visual failure ({failure_text}); prefer modifying "
                "the smallest timeline step instead of regenerating the full script."
            )
            outcome = "failure"

        memory = CADMemory(
            prompt=prompt,
            outcome=outcome,
            tip=tip,
            metrics=metrics,
            failed_constraints=failures,
            created_at=time.time(),
        )
        self.append(memory)
        return memory
