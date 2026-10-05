"""Runtime view of the extensible, versioned label taxonomy (stored in Postgres)."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.db.repository import Repository
from app.services.embedding import EmbeddingService

log = logging.getLogger(__name__)
DIMENSIONS = ("intent", "product", "severity", "sentiment")


@dataclass
class Label:
    label_id: str
    description: str = ""
    keywords: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    rank: int | None = None
    team: str | None = None


@dataclass
class Taxonomy:
    version: int = 0
    labels: dict[str, dict[str, Label]] = field(default_factory=lambda: {d: {} for d in DIMENSIONS})
    prototypes: dict[str, dict[str, np.ndarray]] = field(default_factory=lambda: {d: {} for d in DIMENSIONS})

    def ids(self, dimension: str) -> list[str]:
        return list(self.labels[dimension])

    def has(self, dimension: str, label_id: str) -> bool:
        return label_id in self.labels[dimension]


class TaxonomyService:
    """Loads labels from the DB, builds zero-shot prototype embeddings, and supports adding classes at runtime."""

    def __init__(self, repo: Repository, embedder: EmbeddingService):
        self.repo, self.embedder = repo, embedder
        self.current = Taxonomy()

    async def refresh(self) -> Taxonomy:
        rows = await self.repo.load_taxonomy()
        tax = Taxonomy(version=await self.repo.taxonomy_version())
        for r in rows:
            tax.labels[r["dimension"]][r["label_id"]] = Label(
                r["label_id"], r["description"], list(r["keywords"]), list(r["examples"]), r["rank"], r["team"])
        # prototype = mean embedding of description + examples -> lets a brand-new class work zero-shot
        texts, owners = [], []
        for dim in DIMENSIONS:
            for lab in tax.labels[dim].values():
                for t in [lab.description, *lab.examples]:
                    if t:
                        texts.append(t)
                        owners.append((dim, lab.label_id))
        if texts:
            vecs = await self.embedder.embed_batch(texts)
            buckets: dict[tuple[str, str], list[np.ndarray]] = {}
            for o, v in zip(owners, vecs):
                buckets.setdefault(o, []).append(v)
            for (dim, lid), vs in buckets.items():
                m = np.mean(vs, axis=0)
                tax.prototypes[dim][lid] = m / (np.linalg.norm(m) or 1.0)
        self.current = tax
        log.info("taxonomy loaded", extra={"version": tax.version, **{d: len(tax.labels[d]) for d in DIMENSIONS}})
        return tax

    async def add_label(self, dimension: str, label_id: str, description: str, keywords: list[str], examples: list[str],
                        team: str | None = None) -> Taxonomy:
        await self.repo.add_taxonomy_labels(
            [{"dimension": dimension, "label_id": label_id, "description": description, "keywords": keywords,
              "examples": examples, "rank": None, "team": team}],
            note=f"add {dimension}:{label_id}")
        return await self.refresh()

    @staticmethod
    def load_seed(path: Path) -> list[dict]:
        data = json.loads(path.read_text(encoding="utf-8"))
        out = []
        for dim in DIMENSIONS:
            for i, lab in enumerate(data[dim]):
                out.append({"dimension": dim, "label_id": lab["id"], "description": lab.get("description", ""),
                            "keywords": lab.get("keywords", []), "examples": lab.get("examples", []),
                            "rank": i if dim == "severity" else None, "team": lab.get("team")})
        return out
