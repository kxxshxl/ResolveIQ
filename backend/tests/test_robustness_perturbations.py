"""Perturbations must be deterministic (regression gate stability) and actually change the text in the intended way."""
from app.evaluation.robustness import PERTURBATIONS, lowercase_nopunct, noisy_wrapper, shouting, truncated, typos

TEXT = "My broadband keeps dropping around 8 PM each day, and I've restarted the router twice already. This is costing me money."


def test_deterministic_per_query_id_and_different_across_ids():
    for name, fn in PERTURBATIONS.items():
        assert fn(TEXT, "q1") == fn(TEXT, "q1"), name
    assert typos(TEXT, "q1", rate=0.5) != typos(TEXT, "q2", rate=0.5)


def test_typos_change_some_words_but_keep_length_close():
    out = typos(TEXT, "q1", rate=0.5)
    assert out != TEXT and abs(len(out) - len(TEXT)) <= len(TEXT.split())


def test_other_perturbations():
    assert lowercase_nopunct(TEXT, "q") == "my broadband keeps dropping around 8 pm each day and ive restarted the router twice already this is costing me money"
    assert shouting(TEXT, "q") == TEXT.upper()
    t = truncated(TEXT, "q")
    assert TEXT.startswith(t) and 0.5 < len(t.split()) / len(TEXT.split()) < 0.75
    n = noisy_wrapper(TEXT, "q")
    assert TEXT in n and len(n) > len(TEXT) + 40
