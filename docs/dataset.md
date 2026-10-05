# Dataset: source, preprocessing, schema mapping, splits, limitations

## Source: synthetic, scenario-driven (no public dataset used)

No public dataset was used. Public support-ticket corpora (e.g. Customer Support Tickets, Bitext) are generic, have no
resolution steps or KB articles, and carry no ground truth of *which tickets are the same problem*, which is exactly what a
retrieval benchmark needs. We therefore generate a deterministic, **seeded** synthetic telecom corpus
(`backend/scripts/generate_dataset.py`, seed 1337) from 27 hand-authored root-cause **scenarios**
(`backend/scripts/datagen/scenarios.py`): broadband disconnections (evening congestion vs random line fault vs Wi-Fi-only),
slow speed (peak hours / far rooms / mobile data), router and TV-box faults, four billing disputes, plan upgrade/downgrade,
SIM / coverage / calls-and-SMS, portal login / missing OTP / account takeover, area outages (broadband and mobile),
installation (missed appointment / not activated) - plus 2 scenarios that are **held out as brand-new classes**
(eSIM activation, international roaming) for the evolving-data demonstration.

Each scenario has: a KB article (title, explanation, procedure), a canonical resolution (4-5 steps + summary), allowed
severity levels, and **8 differently worded symptom statements**. Scenarios deliberately include hard negatives that share
vocabulary but differ in root cause (e.g. *evening drops* vs *evening slow speed* vs *random line fault* vs *Wi-Fi-only drops*;
*OTP not received* vs *portal login*; *double charge* vs *charged after cancel*).

## Pipeline

```
generate_dataset.py ──► data/raw/*.jsonl           ticketing-system-style export: HTML, mixed case, injected fake PII,
                                                    P1-P4 priorities, free-text moods, "retired" KB status
prepare_data.py     ──► data/processed/*.jsonl      schema mapping + HTML strip + whitespace/Unicode normalisation + PII redaction
seed.py             ──► PostgreSQL                  same ingestion service the API uses (re-redacts, validates labels, embeds, upserts)
```

### Schema mapping (raw → internal)

| Raw | Internal | Notes |
|---|---|---|
| `description` | `complaint_text` | HTML stripped, normalised, PII redacted |
| `issue_type` ("Broadband Disconnection") | `intent` (`broadband_disconnection`) | slugified |
| `product_line` ("WiFi Router") | `product` (`wifi_router`) | slugified |
| `priority` P1/P2/P3/P4 | `severity` critical/high/medium/low | |
| `customer_mood` ANGRY/FRUSTRATED/WORRIED/NEUTRAL | `sentiment` angry/frustrated/concerned/neutral | |
| `agent_notes[]` | `resolution_steps[]` | |
| `resolution_summary` | `resolution_summary` | |
| `closed_at` | `resolved_at` | |
| `scenario_id` | `metadata.scenario_id` | **ground truth for evaluation only**; never read by retrieval or classification |
| KB `body` (HTML), `procedure`, `topic`, `labels`, `status` retired | `content`, `steps`, `category`, `tags`, `status=deprecated` | |

### Synthetic augmentation

A complaint is composed from `symptom + impact sentence (→ severity label) + tone sentence (→ sentiment label)`
in random order, with light noise (lower-casing, "pls", HTML wrappers, injected fake emails / phones / account numbers / test
card numbers in ~20% of tickets to exercise the PII layer).

## Corpus and splits (leakage control)

| Set | Size | Built from | Used for |
|---|---|---|---|
| Historical tickets | 250 (10 per scenario) | symptoms 0-4, impact/tone sentences 0-3 | the retrieval corpus |
| KB articles | 26 (25 active + 1 retired) | one per scenario + a superseded near-duplicate (`KB-900`) | the retrieval corpus |
| `val` queries | 50 | **held-out** symptom #5, impact/tone sentences 4-5 | all tuning (retrieval weights, ensemble weights, abstention threshold) |
| `test` queries | 100 | **held-out** symptoms #6-#7, impact/tone sentences 4-5 | headline numbers |
| `gold` queries | 101 (`gold.jsonl` 26 + `gold_v2.jsonl` 75) | hand-written, naturally phrased (not from templates), manually labelled; 3 per scenario in different voices | honest check on non-templated language; dev signal while building the affect model |
| `blind` queries | 50 (`gold_blind.jsonl`) | hand-written **after** the affect model and its hyper-parameters were frozen (2 per scenario: voice-transcript, formal letter, non-native English, SMS shorthand) | never used for any selection: the generalisation estimate for severity/sentiment |
| `ood` queries | 12 | unrelated requests incl. a prompt-injection attempt | abstention |
| evolving | 2 intents · 12 tickets · 2 articles · 12 queries | scenarios absent from the launch corpus | evolving-data suite; also the *tuning* stream for discovery parameters |
| `novel_stream` | 3 never-seen classes (number porting, voicemail, TV streaming app) · 30 complaints | hand-written | discovery evaluation only (not used to tune it) |

### Labelling rubric for hand-written sets (severity / sentiment)

Severity: **low** = no or minor impact, workaround exists, customer says no rush; **medium** = regular disruption of normal use
without stated consequences; **high** = stated impact on work, income, studies, or an urgent deadline, or repeated failure with
explicit urgency; **critical** = health/safety/emergency risk, a business-down outage, or a prolonged complete outage affecting many
people. Sentiment: **angry** = hostility, insults or threats to cancel/complain; **frustrated** = exasperation after repeated failed
attempts; **concerned** = worry or a request for reassurance; **neutral** = calm and factual. One author labelled everything, so
expect roughly +/-5 points of label noise (see limitations).

Queries never reuse a corpus symptom sentence or impact/tone sentence, and `test` never participates in tuning, so
retrieval metrics measure **paraphrase generalisation**, not memorisation. Relevance is "same scenario" (about 10 relevant
tickets and 1 relevant article per query).

## Limitations (please read)

* **Synthetic, template-composed text.** Paraphrases are hand-written but there are only 8 per scenario and the
  impact/tone sentences are shared building blocks. Absolute numbers will be higher than on real tickets; the *relative*
  ordering of methods is the useful signal.
* **Severity / sentiment are partly circular on `test`.** Their labels come from the same sentence banks that the
  keyword lexicons in `data/taxonomy.json` were written to recognise (the banks are split train/held-out, the lexicons are
  not). The hand-written `gold` set is the honest check - and it shows much lower severity/sentiment accuracy.
* **The hand-written sets (101 + 50) have one labeller.** Severity and sentiment are subjective (adjacent classes such as
  medium/high or frustrated/concerned are genuinely ambiguous), so treat differences of a few points as noise. The sets are
  still 6x larger than the original 26 and, unlike `test`, are not circular.
* **Same 25 scenarios everywhere.** The hand-written complaints cover the same root causes as the corpus (in new wording); a
  brand-new root cause inside an existing intent is only exercised through the discovery stream.
* Single language (English), no typos beyond light noise, no multi-issue complaints, no conversation context.
* Lexical vs semantic gaps are partly *by design*: held-out symptoms were written to share few words with the corpus.
  Real corpora contain more exact-term cases (error codes, plan names) where lexical search is stronger.
