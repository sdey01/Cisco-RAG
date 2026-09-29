"""Small fault-tolerant JSON semantic cache."""

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


class SemanticCache:
    def __init__(self, path: Path, threshold: float, embeddings):
        self.path = path
        self.threshold = threshold
        self.embeddings = embeddings
        self.entries = self._load()

    def _load(self):
        try:
            return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []
        except (OSError, ValueError):
            return []

    def lookup(self, question: str):
        try:
            vector = np.asarray(self.embeddings.embed_query(question), dtype=float)
            best = None
            for entry in self.entries:
                candidate = np.asarray(entry["embedding"], dtype=float)
                similarity = float(np.dot(vector, candidate) / (np.linalg.norm(vector) * np.linalg.norm(candidate)))
                if best is None or similarity > best[0]:
                    best = (similarity, entry)
            return best[1] if best and best[0] >= self.threshold else None
        except Exception:
            return None

    def put(self, question: str, answer: str, sources: list, route: str) -> None:
        try:
            self.entries.append({
                "question": question,
                "embedding": self.embeddings.embed_query(question),
                "answer": answer,
                "sources": sources,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "route": route,
            })
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.entries, indent=2), encoding="utf-8")
        except Exception:
            return


def cache_summary(path: Path) -> list[dict]:
    """Return safe display metadata without exposing stored embeddings or answers."""
    try:
        entries = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except (OSError, ValueError):
        return []
    return [
        {
            "question": entry.get("question", ""),
            "route": entry.get("route", "unknown"),
            "timestamp": entry.get("timestamp", ""),
            "sources": len(entry.get("sources", [])),
        }
        for entry in reversed(entries)
    ]
