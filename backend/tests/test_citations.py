"""Citation validation + grounding: the core anti-hallucination guarantee."""
import asyncio

from app.models.schemas import Resolution, RetrievedItem, Step
from app.rag.citations import validate_resolution


def _evidence():
    return [
        RetrievedItem(source_type="ticket", source_id="TKT-1", title="Evening drops", text="Broadband disconnects every evening.",
                      score=0.9, rank=1, retrieval_method="dense", metadata={}, resolution_summary="Stable line profile applied",
                      steps=["Check line statistics for the SNR margin in the evening window.",
                             "Apply the stable line profile to the connection."]),
        RetrievedItem(source_type="article", source_id="KB-1", title="Evening disconnections", score=0.8, rank=1,
                      retrieval_method="dense", metadata={}, steps=["Run a 72-hour monitored line test."],
                      text="Drops at the same time each evening point to peak-hour congestion on the cabinet."),
    ]


def _validate(steps, embedder, test_settings):
    res = Resolution(issue_summary="s", steps=steps, escalate=False)
    return asyncio.run(validate_resolution(res, _evidence(), embedder, test_settings))


def test_valid_grounded_resolution_passes(embedder, test_settings):
    res, cites, rep = _validate([Step(text="Check the line statistics and SNR margin during the evening.", citations=["TKT-1"]),
                                 Step(text="Run a 72 hour monitored line test.", citations=["KB-1"])], embedder, test_settings)
    assert rep.valid and not rep.invalid_citations and rep.citation_coverage == 1.0 and rep.grounded_ratio == 1.0
    assert {c.source_id for c in cites} == {"TKT-1", "KB-1"}


def test_invented_source_ids_are_stripped_and_flagged(embedder, test_settings):
    res, cites, rep = _validate([Step(text="Apply the stable line profile.", citations=["TKT-1", "TKT-9999"]),
                                 Step(text="Swap the router.", citations=["KB-404"])], embedder, test_settings)
    assert rep.invalid_citations == ["KB-404", "TKT-9999"] and not rep.valid
    assert all(c.source_id in {"TKT-1", "KB-1"} for c in cites)       # no fabricated id survives
    assert rep.uncited_steps == [2]                                   # step 2 lost its only (fake) citation
    assert all(cid in {"TKT-1", "KB-1"} for s in res.steps for cid in s.citations)


def test_unsupported_step_is_flagged_even_with_a_real_citation(embedder, test_settings):
    res, _, rep = _validate([Step(text="Apply the stable line profile.", citations=["TKT-1"]),
                             Step(text="Purchase a premium satellite dish and reinstall the operating system of the television.",
                                  citations=["TKT-1"])], embedder, test_settings)
    assert rep.unsupported_steps == [2] and res.steps[1].grounded is False and res.steps[0].grounded is True
    assert rep.grounded_ratio == 0.5 and not rep.valid               # below the minimum grounded ratio


def test_citation_ids_are_normalised_but_never_repaired_to_other_sources(embedder, test_settings):
    res, _, rep = _validate([Step(text="Apply the stable line profile.", citations=["[tkt-1]"])], embedder, test_settings)
    assert res.steps[0].citations == ["TKT-1"] and rep.valid


# ---------------------------------------------------------------- embedding of evidence text: de-duplication and the LRU cache
def _old_way(embedder, step_texts, texts):
    """What validate_resolution computed before: one encode of every step and every (repeated) evidence unit."""
    return embedder.encode_sync(list(step_texts) + list(texts))


def test_cached_embeddings_equal_the_direct_computation(embedder):
    texts = ["Run a 72-hour monitored line test.", "Apply the stable line profile.", "Run a 72-hour monitored line test.", "Check the SNR margin."]
    direct = _old_way(embedder, [], texts)
    got = asyncio.run(embedder.embed_cached(texts))
    assert got.shape == direct.shape
    assert abs(got - direct).max() < 1e-4, "batching and de-duplication must not change the embedding"
    assert (got[0] == got[2]).all(), "a repeated text gets exactly the same vector"


def test_repeated_evidence_is_embedded_once_per_call_and_then_served_from_the_cache(embedder, monkeypatch):
    embedder._units.clear()
    calls = []
    real = embedder.encode_sync
    monkeypatch.setattr(embedder, "encode_sync", lambda texts: calls.append(list(texts)) or real(texts))
    texts = ["alpha unit text one", "beta unit text two", "alpha unit text one", "alpha unit text one"]

    asyncio.run(embedder.embed_cached(texts))
    assert calls == [["alpha unit text one", "beta unit text two"]], "duplicates inside one call are embedded once"
    asyncio.run(embedder.embed_cached(texts + ["gamma unit text three"]))
    assert calls[1] == ["gamma unit text three"], "texts seen before are not embedded again"
    assert asyncio.run(embedder.embed_cached([])).shape[0] == 0


def test_unit_cache_is_a_bounded_lru(embedder, monkeypatch):
    embedder._units.clear()
    monkeypatch.setattr(embedder.s, "embedding_unit_cache_size", 3)
    asyncio.run(embedder.embed_cached(["t1", "t2", "t3"]))
    asyncio.run(embedder.embed_cached(["t1"]))          # t1 becomes the most recently used
    asyncio.run(embedder.embed_cached(["t4"]))          # evicts the least recently used: t2
    assert list(embedder._units) == ["t3", "t1", "t4"] and len(embedder._units) == 3
    monkeypatch.setattr(embedder.s, "embedding_unit_cache_size", 0)
    embedder._units.clear()
    asyncio.run(embedder.embed_cached(["t1", "t1"]))
    assert len(embedder._units) == 0, "size 0 turns the cache off (de-duplication still applies)"


def test_validation_result_is_unchanged_by_the_cache(embedder, test_settings):
    steps = [Step(text="Apply the stable line profile.", citations=["TKT-1", "KB-1"]), Step(text="Run a monitored line test.", citations=["KB-1"]),
             Step(text="Reboot the cabinet.", citations=["TKT-1"])]
    embedder._units.clear()
    cold = _validate([s.model_copy() for s in steps], embedder, test_settings)
    warm = _validate([s.model_copy() for s in steps], embedder, test_settings)  # second call is served from the cache
    for (_, _, a), (_, _, b) in [(cold, warm)]:
        assert a.grounded_ratio == b.grounded_ratio and a.unsupported_steps == b.unsupported_steps
    assert [s.grounding_score for s in cold[0].steps] == [s.grounding_score for s in warm[0].steps]
