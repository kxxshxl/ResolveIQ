"""Evaluation data loading and ground-truth relevance derived from scenario ids stored in ticket/article metadata."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from app.db.repository import Repository


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


@dataclass
class EvalSets:
    test: list[dict]
    val: list[dict]
    gold: list[dict]
    ood: list[dict]
    evolving: list[dict]
    blind: list[dict] = field(default_factory=list)


def load_eval_sets(data_dir: Path) -> EvalSets:
    q = load_jsonl(data_dir / "eval" / "queries.jsonl")
    ev = data_dir / "eval"
    optional = lambda name: load_jsonl(ev / name) if (ev / name).exists() else []  # noqa: E731
    return EvalSets(
        test=[r for r in q if r["split"] == "test"], val=[r for r in q if r["split"] == "val"],
        # gold = hand-written complaints (v1 + v2); blind = written after the affect model was frozen, never used for selection
        gold=load_jsonl(ev / "gold.jsonl") + optional("gold_v2.jsonl"), blind=optional("gold_blind.jsonl"),
        ood=load_jsonl(ev / "ood.jsonl"), evolving=load_jsonl(ev / "evolving_queries.jsonl"))


@dataclass
class Corpus:
    """scenario -> relevant ids, built from the live (active) corpus so eval always matches what is indexed."""
    tickets: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    articles: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    scenario_of: dict[str, str] = field(default_factory=dict)
    article_steps: dict[str, list[str]] = field(default_factory=dict)
    deprecated: set[str] = field(default_factory=set)

    def relevant(self, kind: str, scenario: str) -> set[str]:
        return (self.tickets if kind == "ticket" else self.articles).get(scenario, set())


async def build_corpus(repo: Repository) -> Corpus:
    c = Corpus()
    for r in await repo.all_docs("ticket"):
        sid = (r["metadata"] or {}).get("scenario_id")
        if sid:
            c.tickets[sid].add(r["source_id"])
            c.scenario_of[r["source_id"]] = sid
    for r in await repo.all_docs("article"):
        sid = (r["metadata"] or {}).get("scenario_id")
        if sid:
            c.articles[sid].add(r["source_id"])
            c.scenario_of[r["source_id"]] = sid
            c.article_steps[sid] = [str(s) for s in (r["steps"] or [])]
    rows = await repo._fetch("SELECT article_id FROM knowledge_articles WHERE status='deprecated'")
    c.deprecated = {r["article_id"] for r in rows}
    return c
