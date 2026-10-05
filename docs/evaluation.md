# Evaluation: method, results summary, experiments, decisions

Full tables (auto-generated from the recorded run, nothing hand-typed): [`evaluation_results.md`](evaluation_results.md);
raw JSON: `data/eval/results/latest.json`; the run recorded **before** this round of work: `data/eval/results/baseline_v1.json`;
tuning log: `data/eval/results/tuning.txt`; embedding comparison: `data/eval/results/embedding_experiment.txt`.
Data and splits: [`dataset.md`](dataset.md).

Reproduce (RTX 4070 laptop, Ollama with `qwen3:4b-instruct`, seeded database):

```
python -m app.evaluation.run --suites classification retrieval robustness evolving discovery   # ~4 min, no LLM
python -m app.evaluation.run --suites rag e2e --judge 20 --alt-openai-model qwen3:4b-instruct  # ~10 min, needs Ollama
python -m app.evaluation.gate                                                                   # regression gate (CI)
```

## Method

| Subsystem | Metrics | Queries | Notes |
|---|---|---|---|
| Classification | accuracy, macro P/R/F1 per dimension, confidence calibration | `test` (100, templated), `gold` (101 hand-written), `blind` (50 hand-written, written after the affect model was frozen) | strategies compared: rules, kNN+prototypes, `ensemble_legacy` (rules+kNN for every dimension = before), `ensemble` (shipped: + NLI affect model) |
| Retrieval | P@K, Recall@K, Hit@K, MRR, nDCG@K (K=1,3,5,10), latency p50/p95 | `test`, `gold`, `blind`; tickets and articles | lexical (Postgres FTS), BM25 (in-memory, eval-only, keeps the keyword baseline fair), dense, hybrid, hybrid+rerank |
| Robustness | the above under deterministic perturbations | `gold` + `blind` (151) | typos, no punctuation / lower-case, signature + unrelated asides, truncation to 60%, ALL CAPS |
| Generation / RAG | step faithfulness and hallucination (lexical containment in *all* evidence), citation validity, citation precision vs gold scenario, citation coverage, gold-step recall, answer relevance, optional LLM judge | 40 test + 101 gold, real LLM | variants: Qwen3-4B via the native Ollama provider, via the OpenAI-compatible provider, evidence-only (no LLM) |
| End-to-end | successful / correct resolution rate, abstention (false and correct), latency, failures | 141 in-domain + 12 out-of-domain | "correct" = resolved **and** cites a source from the right scenario |
| Evolving data | Hit@K / MRR / intent accuracy before vs after runtime ingestion | 12 queries, 2 new intents | no restart; cleaned up afterwards |
| Class discovery | candidate recall, cluster purity per new class, proposal precision, routing after acceptance | 21 complaints from 3 never-seen classes mixed with 251 known | parameters tuned on a different stream (eSIM + roaming + validation queries); accept step temporarily adds labels and removes them |

## Headline results

**Semantic vs keyword retrieval (tickets, held-out templated paraphrases `test`, n=100):**

| strategy | Hit@1 | Hit@5 | MRR | nDCG@10 | P@5 |
|---|---|---|---|---|---|
| lexical (Postgres FTS) | 0.150 | 0.450 | 0.280 | 0.133 | 0.146 |
| BM25 (in-memory) | 0.280 | 0.540 | 0.403 | 0.195 | 0.212 |
| **dense (pgvector)** | **0.660** | **0.930** | **0.774** | **0.492** | **0.558** |
| hybrid (weighted RRF) | 0.580 | 0.910 | 0.721 | 0.421 | 0.418 |
| hybrid + cross-encoder | 0.570 | 0.900 | 0.715 | 0.421 | 0.422 |

**On hand-written complaints** (Hit@1 / Hit@5 / MRR, tickets): the gap between semantic and keyword search is smaller than on the
adversarial `test` set but still large, and it holds on text written after everything was frozen.

| split | lexical | BM25 | dense | hybrid | hybrid + rerank |
|---|---|---|---|---|---|
| gold (101) | .66 / .90 / .76 | .75 / .93 / .82 | **.91 / .98 / .94** | .91 / .98 / .93 | .91 / .98 / .93 |
| blind (50) | .58 / .88 / .70 | .66 / .92 / .76 | **.90 / .98 / .94** | .80 / .96 / .87 | .82 / .96 / .88 |

KB articles (Hit@1): dense .74 / .88 / .90 on test / gold / blind versus lexical .37 / .65 / .70; on **blind articles hybrid wins** (.94 vs .90).

**Classification** (accuracy). `ensemble_legacy` = before; `ensemble` = shipped. The legacy rules reach 1.00 / 0.98 on the templated `test`
set only because their lexicon and the generator share vocabulary (circular); every hand-written split shows the real picture.

| split | strategy | intent | product | severity | sentiment |
|---|---|---|---|---|---|
| test (100) | legacy | 0.85 | 0.89 | 1.00 (circular) | 0.98 (circular) |
| test (100) | shipped | 0.85 | 0.89 | 0.72 | 0.85 |
| gold (101) | legacy | 0.95 | 0.98 | 0.48 | 0.39 |
| gold (101) | **shipped** | 0.95 | 0.98 | **0.65** | **0.66** |
| blind (50) | legacy | 0.96 | 0.92 | 0.48 | 0.48 |
| blind (50) | **shipped** | 0.96 | 0.92 | **0.64** | **0.78** |

Macro-F1 on blind: severity 0.35 → 0.58, sentiment 0.48 → 0.76. Severity is within one ordinal level for ~98% of gold queries.
Confidence is informative (on `test`): mean confidence when right vs wrong is 0.75 vs 0.62 (severity) and 0.81 vs 0.63 (sentiment).

**Robustness** (gold + blind, n=151; ensemble classifier, dense retrieval):

| perturbation | intent | severity | sentiment | ticket Hit@1 | ticket Hit@5 | article Hit@3 |
|---|---|---|---|---|---|---|
| clean | 0.940 | 0.649 | 0.702 | 0.907 | 0.980 | 0.967 |
| typos | 0.934 | 0.649 | 0.702 | 0.861 | 0.967 | 0.960 |
| no punctuation, lower-case | 0.921 | 0.556 | 0.589 | 0.874 | 0.980 | 0.967 |
| signature + unrelated aside | 0.947 | 0.715 | 0.649 | 0.795 | 0.980 | 0.960 |
| truncated to 60% | 0.907 | 0.464 | 0.391 | 0.848 | 0.974 | 0.934 |
| ALL CAPS | 0.940 | 0.642 | 0.768 | 0.907 | 0.980 | 0.967 |

Retrieval and intent are robust (Hit@5 never drops below 0.967). Severity and sentiment depend on cue words and punctuation: they lose 9-31 points when
the text is lower-cased without punctuation or truncated, which is the main remaining weakness for pasted-from-chat input.

**Emerging-class discovery** (3 never-seen classes, 7 complaints each in the stream, mixed with 251 known complaints):

| stage | result |
|---|---|
| novel complaints flagged as candidates (evidence confidence < 0.80) | 18 / 21 = **0.86**; the abstention gate alone catches **0.24** |
| known complaints wrongly flagged | 89 / 251 = 0.36 (a deliberately noisy filter; clustering removes most of it) |
| proposals produced | 5 (all `new_class`) |
| classes recovered (cluster purity >= 0.8) | **3 / 3**, each at purity 1.0; coverage of the class 0.86 / 0.86 / 0.57 |
| proposal precision | 0.60 (the 2 spurious proposals were mixed *known* topics: billing/plan wording, and genuine outage complaints that scored low on evidence) |
| keywords of recovered clusters | `number, transfer, old, network, code` · `app, says, update, subscription isn't, sync picture` · `voicemail, work missing, visual, ...` |
| held-out complaints routed to the right class after accepting the proposals | 0.00 → **0.56**, with **zero** resolved tickets for those classes |

Live check (separate API and worker processes, real LLM): 30 complaints from the three classes were sent to `/resolve`; **22 were
answered** from adjacent classes and only 8 abstained; the queued discovery job ran on the worker (attempt 1, 0.16 s) and produced three
proposals (`number_*`, `app_*`, `voicemail_*`). This is why discovery exists: the abstention gate is not a novelty detector.

**RAG (Qwen3-4B-instruct, evidence = top 3 tickets + 2 articles, 137 answered requests):** step faithfulness 0.9985, citation validity 1.000,
citation coverage 1.000, citation precision vs gold 0.903, gold-step recall 0.923, answer relevance 0.926; LLM self-judge 4.9 / 4.95 of 5.
Evidence-only baseline (no LLM, n=38): faithfulness 1.000 but citation precision 0.591 and gold-step recall 0.763. OpenAI-compatible
provider pointed at Ollama: 0.995 / 0.797 / 0.858 (n=38), i.e. the provider abstraction behaves the same.

**End-to-end (153 requests):** 97.2% in-domain resolved, **92.9% correct** (resolved *and* cites a source from the right scenario),
2.8% false abstention, 0 unreliable, **12 / 12 out-of-domain abstained** (including the prompt-injection attempt), 0 pipeline errors;
latency p50 6.3 s / p95 7.8 s, dominated by generation (p50 6.3 s). *Before this round (66 requests): 89.4% correct.*

**Evolving data:** two new intents (12 tickets, 2 articles) become searchable with no restart: Hit@5 0.00 → 0.92, MRR 0 → 0.61;
intent accuracy only reaches 0.50 at 6 tickets per new class because existing neighbours outvote them (unchanged limitation).

## Additional exploration (what was tried, what won, why)

1. **Keyword vs semantic** - see the tables above. Paraphrase queries share few words with history, so keyword scoring is
   dominated by incidental words from the impact/tone sentences. Dense Hit@1 is 4.4x lexical on `test` and 1.4-1.6x lexical (1.2x BM25 on gold) on hand-written text.
2. **Dense vs hybrid vs hybrid+rerank - the expected ordering did *not* hold on tickets.** Tuning on the validation split
   (`tuning.txt`) selected lexical RRF weight 0 and rerank blend 0. Reasons: the held-out paraphrases are deliberately lexically distant,
   and `ms-marco-MiniLM` is a query-to-passage relevance model, not a complaint-to-complaint similarity model. Hybrid helps on KB articles
   (blind Hit@1 .94 vs .90) and on gold P@5 with the reranker, so the machinery stays selectable per request, but **`dense` is the default**.
3. **Embedding model** (`embedding_experiment.txt`, 384-dim): MiniLM-L6 best on the selection split; bge-small / gte-small within noise on test.
4. **Metadata filtering**: a hard filter from the *predicted* product raised P@5 0.422 → 0.496 but not Hit@1/MRR; opt-in because a wrong prediction hides the right answer.
5. **Top-K / rerank pool size**: dense Hit@K 0.66 / 0.87 / 0.93 / 0.97 for K = 1/3/5/10; pool sizes above 10 change nothing.
6. **LLM / provider**: native Ollama and the OpenAI-compatible provider behave equivalently; evidence-only generation costs 0.2 s but cites the wrong scenario far more often.
7. **Severity and sentiment model** (the largest improvement this round). Diagnosis: the old method voted among *topic*-similar past tickets,
   which carries no information about how the customer describes the impact. Candidates, scored on all 101 hand-written queries:

   | candidate | sentiment acc | severity acc | note |
   |---|---|---|---|
   | majority class | 0.32 | 0.44 | |
   | logistic regression on MiniLM embeddings (trained on synthetic tickets) | 0.44 | 0.40 | learns template topics, not tone |
   | emotion model (`distilroberta`, mapped to 4 labels) | 0.51 | - | has no "frustrated" class |
   | zero-shot NLI, base | 0.48 | 0.52 | |
   | zero-shot NLI, large | 0.70 | 0.47 | good sentiment, poor severity as a single choice |
   | **14 NLI cue scores + logistic combiner (trained on synthetic tickets only), large** | **0.66** | **0.65** | shipped (C = 2); a 6-cue sentiment-only variant reached 0.71 but severity needs the other cues |
   | same with the base NLI model | 0.61-0.62 | 0.45-0.48 | 3.7x faster, clearly worse |

   Honest note on selection: the NLI model size and the C regularisation (2.0) were chosen looking at aggregate accuracy on `gold`
   (101); that is why the `blind` set (50, written afterwards with a different style mix) exists and why headline claims are
   quoted on `blind`: 0.64 / 0.78, consistent with gold. The combiner never trains on hand-written text.
8. **Serving the NLI model (found by benchmarking, not assumed).** On CPU the 435M-parameter model costs ~1.7 s per complaint in
   float32; int8 dynamic quantisation made it 1.1 s but **collapsed blind accuracy to 0.48 / 0.34** (vs 0.64 / 0.78), so it was rejected and
   the option removed. float32 on CPU reproduces the GPU results exactly. A separate finding: recent `transformers` keeps the checkpoint's
   fp16 dtype even on CPU, which made the first benchmark ~10x slower than necessary; the loader now forces float32 on CPU. In the request
   path classification now runs concurrently with retrieval, so the model's cost is mostly hidden.
9. **Discovery parameters** (tuned on eSIM + roaming + validation queries, evaluated on three other classes): average-linkage cosine
   distance 0.55 beat 0.45 (fragmented clusters) and 0.65 (merged distinct topics with known ones); a candidate threshold of 0.80 balances
   recall (0.86) against noise. A guard was added after the tuning stream produced a cluster of *known* complaints grouped only by
   their angry closing sentences: clusters whose members are already close to existing tickets (mean top-1 cosine >= 0.60) are
   either "extend the owner class" (if one intent owns them) or dropped as a tone/format artefact.
10. **LLM zero-shot classification fallback**: zero-shot Qwen3-4B alone is worse than the ensemble on intent/product (previous run:
    0.85 / 0.80). It no longer overrides severity/sentiment (the affect model is stronger) and only covers low-confidence intent/product. Not re-run this round.

## Calibration provenance

Retrieval weights, ensemble weights per dimension (intent rules 0.35 / kNN 0.65; product kNN only), and the abstention threshold **0.55**
(balanced accuracy 0.95) come from the validation split; the affect combiner is trained on synthetic tickets only; discovery parameters come
from the tuning stream described above. `test` and the discovery evaluation stream were not used for any selection; `blind` was written after the
affect model was frozen. `data/eval/thresholds.json` floors are set from this recorded run and enforced by `python -m app.evaluation.gate`.

## Limitations of these results (honest reading)

* **Synthetic corpus** (250 tickets, 25 scenarios, template-composed): use absolute numbers for relative comparison only. The hand-written
  sets cover the same 25 root causes in new wording, so they do not test unseen root causes inside existing intents.
* **Single labeller for the hand-written sets**: adjacent severity/sentiment classes (medium/high, frustrated/concerned) are genuinely
  ambiguous, so expect about +/-5 points of label noise; n=50 for `blind` means a 95% interval of roughly +/-13 points on any one cell.
* **Severity / sentiment remain the weakest outputs** (0.64-0.66 / 0.66-0.78) and degrade without punctuation or on truncated text.
* **Discovery precision is 0.6** on this stream and proposals need human review; it needs >= 4 similar complaints, so very rare new issues stay invisible.
  Only three new classes were available for evaluation. Routing after acceptance (0.56) is zero-shot; it improves as tickets are ingested but was not measured past 6 per class.
* **Hallucination metrics are proxies** (lexical containment; gold-step recall uses embedding similarity >= 0.6); the LLM judge is a 4B model grading itself.
* **Adjacent-domain gap**: unrelated requests are abstained (12/12) but near-neighbour classes are answered from the closest existing class (22 of 30 in the live check).
* RAG and end-to-end numbers were recorded on one laptop with one LLM (Qwen3-4B); CI re-runs only the LLM-free suites, so the RAG/e2e
  gate checks read the last locally recorded values.
