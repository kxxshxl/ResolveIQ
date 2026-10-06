# Evaluation results (auto-generated)

Generated 2026-10-06T16:17:04+00:00 by `python -m app.evaluation.run`. Embedding `sentence-transformers/all-MiniLM-L6-v2`, reranker `cross-encoder/ms-marco-MiniLM-L-6-v2`, LLM `['ollama:qwen3:4b-instruct']`. Corpus: {'tickets': 250, 'articles': 25, 'deprecated_articles': 1, 'ticket_embeddings': 250, 'kb_embeddings': 26}. Eval sets: {'val': 50, 'test': 100, 'gold': 101, 'blind': 50, 'ood': 12, 'evolving': 12}. Thresholds: {'abstain': 0.55, 'grounding': 0.45, 'min_grounded_ratio': 0.6}.

## 1. Classification

Held-out paraphrase queries (`test`, templated: severity/sentiment rules share vocabulary with the generator, so those numbers are circular), the hand-written `gold` set (101) and the `blind` set (50, written after the affect model was frozen; never used for any selection). `ensemble_legacy` = rules + kNN for every dimension (before the affect model); `ensemble` = shipped.

**split = test**

| strategy | dimension | accuracy | macro-P | macro-R | macro-F1 | n |
|---|---|---|---|---|---|---|
| rules | intent | 0.660 | 0.856 | 0.676 | 0.737 | 100 |
| rules | product | 0.580 | 0.768 | 0.643 | 0.687 | 100 |
| rules | severity | 1.000 | 1.000 | 1.000 | 1.000 | 100 |
| rules | sentiment | 0.990 | 0.991 | 0.988 | 0.989 | 100 |
| embedding | intent | 0.810 | 0.834 | 0.792 | 0.777 | 100 |
| embedding | product | 0.890 | 0.926 | 0.901 | 0.907 | 100 |
| embedding | severity | 0.580 | 0.530 | 0.487 | 0.491 | 100 |
| embedding | sentiment | 0.400 | 0.420 | 0.404 | 0.361 | 100 |
| ensemble_legacy | intent | 0.850 | 0.866 | 0.840 | 0.834 | 100 |
| ensemble_legacy | product | 0.890 | 0.926 | 0.901 | 0.907 | 100 |
| ensemble_legacy | severity | 1.000 | 1.000 | 1.000 | 1.000 | 100 |
| ensemble_legacy | sentiment | 0.980 | 0.980 | 0.975 | 0.977 | 100 |
| ensemble | intent | 0.850 | 0.866 | 0.840 | 0.834 | 100 |
| ensemble | product | 0.890 | 0.926 | 0.901 | 0.907 | 100 |
| ensemble | severity | 0.720 | 0.755 | 0.663 | 0.667 | 100 |
| ensemble | sentiment | 0.850 | 0.882 | 0.856 | 0.847 | 100 |

**split = gold**

| strategy | dimension | accuracy | macro-P | macro-R | macro-F1 | n |
|---|---|---|---|---|---|---|
| rules | intent | 0.743 | 0.822 | 0.728 | 0.761 | 101 |
| rules | product | 0.772 | 0.886 | 0.807 | 0.844 | 101 |
| rules | severity | 0.515 | 0.680 | 0.446 | 0.420 | 101 |
| rules | sentiment | 0.455 | 0.805 | 0.421 | 0.408 | 101 |
| embedding | intent | 0.960 | 0.960 | 0.965 | 0.960 | 101 |
| embedding | product | 0.980 | 0.981 | 0.981 | 0.981 | 101 |
| embedding | severity | 0.426 | 0.450 | 0.442 | 0.442 | 101 |
| embedding | sentiment | 0.317 | 0.313 | 0.361 | 0.304 | 101 |
| ensemble_legacy | intent | 0.951 | 0.954 | 0.956 | 0.952 | 101 |
| ensemble_legacy | product | 0.980 | 0.981 | 0.981 | 0.981 | 101 |
| ensemble_legacy | severity | 0.475 | 0.470 | 0.518 | 0.483 | 101 |
| ensemble_legacy | sentiment | 0.386 | 0.387 | 0.441 | 0.379 | 101 |
| ensemble | intent | 0.951 | 0.954 | 0.956 | 0.952 | 101 |
| ensemble | product | 0.980 | 0.981 | 0.981 | 0.981 | 101 |
| ensemble | severity | 0.653 | 0.730 | 0.612 | 0.609 | 101 |
| ensemble | sentiment | 0.663 | 0.771 | 0.637 | 0.645 | 101 |

**split = blind**

| strategy | dimension | accuracy | macro-P | macro-R | macro-F1 | n |
|---|---|---|---|---|---|---|
| rules | intent | 0.720 | 0.839 | 0.713 | 0.751 | 50 |
| rules | product | 0.780 | 0.865 | 0.795 | 0.819 | 50 |
| rules | severity | 0.500 | 0.495 | 0.298 | 0.260 | 50 |
| rules | sentiment | 0.520 | 0.734 | 0.455 | 0.449 | 50 |
| embedding | intent | 0.940 | 0.940 | 0.940 | 0.933 | 50 |
| embedding | product | 0.920 | 0.943 | 0.928 | 0.933 | 50 |
| embedding | severity | 0.460 | 0.330 | 0.316 | 0.318 | 50 |
| embedding | sentiment | 0.400 | 0.409 | 0.433 | 0.395 | 50 |
| ensemble_legacy | intent | 0.960 | 0.962 | 0.958 | 0.956 | 50 |
| ensemble_legacy | product | 0.920 | 0.943 | 0.928 | 0.933 | 50 |
| ensemble_legacy | severity | 0.480 | 0.358 | 0.345 | 0.350 | 50 |
| ensemble_legacy | sentiment | 0.480 | 0.510 | 0.524 | 0.478 | 50 |
| ensemble | intent | 0.960 | 0.962 | 0.958 | 0.956 | 50 |
| ensemble | product | 0.920 | 0.943 | 0.928 | 0.933 | 50 |
| ensemble | severity | 0.640 | 0.677 | 0.675 | 0.576 | 50 |
| ensemble | sentiment | 0.780 | 0.851 | 0.751 | 0.763 | 50 |

Ensemble confidence calibration (test): mean confidence when correct vs wrong

| dimension | mean conf (correct) | mean conf (wrong) |
|---|---|---|
| intent | 0.615 | 0.395 |
| product | 0.783 | 0.589 |
| severity | 0.753 | 0.624 |
| sentiment | 0.812 | 0.625 |

Ensemble classification latency (embedding excluded): p50 39.3 ms, p95 45.8 ms.

## 2. Retrieval

Query-embedding latency: {'avg_ms': 0.7, 'p50_ms': 0.6, 'p95_ms': 0.8, 'n': 251}. A document is relevant iff it belongs to the same root-cause scenario as the query (about 10 relevant tickets and 1 relevant KB article per query). Retrieval latency excludes query embedding.

**tickets, split = test** (n = 100)

| strategy | P@5 | Recall@10 | Hit@1 | Hit@5 | MRR | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| lexical | 0.146 | 0.127 | 0.150 | 0.450 | 0.280 | 0.133 | 3.200 | 4.300 |
| bm25 | 0.212 | 0.172 | 0.280 | 0.540 | 0.403 | 0.195 | 1.000 | 1.200 |
| dense | 0.558 | 0.441 | 0.660 | 0.930 | 0.774 | 0.492 | 1.100 | 1.300 |
| hybrid | 0.418 | 0.385 | 0.580 | 0.910 | 0.721 | 0.421 | 5.200 | 8.800 |
| hybrid_reranked | 0.422 | 0.385 | 0.570 | 0.900 | 0.715 | 0.421 | 15.300 | 17.500 |
| adaptive | 0.558 | 0.441 | 0.660 | 0.930 | 0.774 | 0.492 | 1.300 | 1.600 |

**tickets, split = gold** (n = 101)

| strategy | P@5 | Recall@10 | Hit@1 | Hit@5 | MRR | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| lexical | 0.535 | 0.394 | 0.663 | 0.901 | 0.761 | 0.460 | 3.000 | 4.100 |
| bm25 | 0.586 | 0.418 | 0.752 | 0.931 | 0.817 | 0.497 | 1.000 | 1.200 |
| dense | 0.711 | 0.578 | 0.911 | 0.980 | 0.941 | 0.652 | 1.200 | 1.400 |
| hybrid | 0.709 | 0.545 | 0.911 | 0.980 | 0.934 | 0.627 | 4.800 | 5.900 |
| hybrid_reranked | 0.737 | 0.552 | 0.911 | 0.980 | 0.934 | 0.638 | 15.900 | 18.600 |
| adaptive | 0.711 | 0.578 | 0.911 | 0.980 | 0.941 | 0.652 | 1.300 | 1.500 |

**tickets, split = blind** (n = 50)

| strategy | P@5 | Recall@10 | Hit@1 | Hit@5 | MRR | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| lexical | 0.504 | 0.350 | 0.580 | 0.880 | 0.703 | 0.411 | 3.100 | 4.000 |
| bm25 | 0.564 | 0.438 | 0.660 | 0.920 | 0.755 | 0.495 | 1.000 | 1.300 |
| dense | 0.692 | 0.588 | 0.900 | 0.980 | 0.935 | 0.649 | 1.200 | 1.500 |
| hybrid | 0.664 | 0.536 | 0.800 | 0.960 | 0.871 | 0.598 | 4.800 | 5.700 |
| hybrid_reranked | 0.672 | 0.552 | 0.820 | 0.960 | 0.884 | 0.614 | 15.100 | 17.100 |
| adaptive | 0.692 | 0.588 | 0.900 | 0.980 | 0.935 | 0.649 | 1.300 | 1.500 |

**articles, split = test** (n = 100)

| strategy | P@5 | Recall@10 | Hit@1 | Hit@5 | MRR | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| lexical | 0.134 | 0.790 | 0.370 | 0.670 | 0.492 | 0.563 | 1.900 | 2.200 |
| bm25 | 0.128 | 0.810 | 0.370 | 0.640 | 0.498 | 0.572 | 0.700 | 0.900 |
| dense | 0.194 | 0.990 | 0.740 | 0.970 | 0.829 | 0.868 | 1.000 | 1.100 |
| hybrid | 0.168 | 0.910 | 0.690 | 0.840 | 0.757 | 0.793 | 3.400 | 4.500 |
| hybrid_reranked | 0.168 | 0.910 | 0.720 | 0.840 | 0.772 | 0.804 | 25.500 | 27.900 |
| adaptive | 0.186 | 0.970 | 0.750 | 0.930 | 0.824 | 0.859 | 1.200 | 3.300 |

**articles, split = gold** (n = 101)

| strategy | P@5 | Recall@10 | Hit@1 | Hit@5 | MRR | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| lexical | 0.182 | 0.960 | 0.653 | 0.911 | 0.753 | 0.803 | 1.900 | 2.200 |
| bm25 | 0.176 | 0.941 | 0.663 | 0.881 | 0.753 | 0.798 | 0.700 | 0.900 |
| dense | 0.200 | 1.000 | 0.881 | 1.000 | 0.926 | 0.944 | 1.000 | 1.200 |
| hybrid | 0.200 | 1.000 | 0.881 | 1.000 | 0.927 | 0.946 | 3.100 | 3.500 |
| hybrid_reranked | 0.200 | 1.000 | 0.861 | 1.000 | 0.920 | 0.940 | 24.200 | 26.700 |
| adaptive | 0.200 | 1.000 | 0.861 | 1.000 | 0.912 | 0.933 | 1.200 | 3.300 |

**articles, split = blind** (n = 50)

| strategy | P@5 | Recall@10 | Hit@1 | Hit@5 | MRR | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| lexical | 0.200 | 1.000 | 0.700 | 1.000 | 0.813 | 0.859 | 2.100 | 2.800 |
| bm25 | 0.188 | 0.960 | 0.760 | 0.940 | 0.828 | 0.860 | 0.700 | 0.800 |
| dense | 0.200 | 1.000 | 0.900 | 1.000 | 0.942 | 0.957 | 1.100 | 1.300 |
| hybrid | 0.200 | 1.000 | 0.940 | 1.000 | 0.967 | 0.975 | 3.400 | 4.100 |
| hybrid_reranked | 0.200 | 1.000 | 0.940 | 1.000 | 0.967 | 0.975 | 24.200 | 25.800 |
| adaptive | 0.200 | 1.000 | 0.920 | 1.000 | 0.953 | 0.965 | 1.200 | 3.300 |

### Top-K sweep (tickets, test)

| strategy | Hit@1 | Hit@3 | Hit@5 | Hit@10 | nDCG@1 | nDCG@3 | nDCG@5 | nDCG@10 | P@1 | P@3 | P@5 | P@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lexical | 0.150 | 0.320 | 0.450 | 0.670 | 0.150 | 0.148 | 0.147 | 0.133 | 0.150 | 0.147 | 0.146 | 0.127 |
| bm25 | 0.280 | 0.460 | 0.540 | 0.710 | 0.280 | 0.248 | 0.228 | 0.195 | 0.280 | 0.237 | 0.212 | 0.172 |
| dense | 0.660 | 0.870 | 0.930 | 0.970 | 0.660 | 0.618 | 0.582 | 0.492 | 0.660 | 0.607 | 0.558 | 0.441 |
| hybrid | 0.580 | 0.820 | 0.910 | 0.970 | 0.580 | 0.513 | 0.457 | 0.421 | 0.580 | 0.490 | 0.418 | 0.385 |
| hybrid_reranked | 0.570 | 0.810 | 0.900 | 0.970 | 0.570 | 0.512 | 0.460 | 0.421 | 0.570 | 0.490 | 0.422 | 0.385 |

### Metadata filtering (hybrid_reranked, tickets, test)

| variant | P@5 | Hit@1 | MRR | nDCG@10 | p50 ms |
|---|---|---|---|---|---|
| no_filter | 0.422 | 0.570 | 0.715 | 0.421 | 15.300 |
| predicted_product_filter | 0.496 | 0.560 | 0.705 | 0.459 | 14.800 |
| oracle_product_filter | 0.564 | 0.680 | 0.796 | 0.523 | 14.100 |

Predicted-product accuracy on these queries: 0.88.

### Reranker candidate pool size (tickets, test)

| rerank_top_n | P@5 | Hit@1 | MRR | nDCG@10 | p50 ms |
|---|---|---|---|---|---|
| 5 | 0.418 | 0.570 | 0.706 | 0.296 | 10.400 |
| 10 | 0.420 | 0.570 | 0.715 | 0.421 | 11.700 |
| 20 | 0.422 | 0.570 | 0.715 | 0.421 | 16.000 |
| 30 | 0.422 | 0.570 | 0.715 | 0.421 | 19.200 |

Stale-document hygiene: deprecated articles ['KB-900'] returned 0 times across all strategies and queries.

## 2b. Robustness to messy input

Hand-written gold + blind queries (n = 151) under deterministic perturbations; ensemble classifier and dense retrieval.

| perturbation | intent | product | severity | sentiment | ticket_hit@1 | ticket_hit@5 | article_hit@3 |
|---|---|---|---|---|---|---|---|
| clean | 0.940 | 0.954 | 0.649 | 0.702 | 0.907 | 0.980 | 0.967 |
| typos | 0.947 | 0.947 | 0.649 | 0.702 | 0.861 | 0.967 | 0.960 |
| no_punctuation_lowercase | 0.920 | 0.954 | 0.556 | 0.589 | 0.874 | 0.980 | 0.967 |
| noise_and_signature | 0.947 | 0.947 | 0.715 | 0.649 | 0.795 | 0.980 | 0.960 |
| truncated_60pct | 0.907 | 0.960 | 0.464 | 0.391 | 0.848 | 0.974 | 0.934 |
| all_caps | 0.940 | 0.954 | 0.642 | 0.768 | 0.907 | 0.980 | 0.967 |

Largest drop versus clean per metric: {'intent': 0.0331, 'product': 0.0066, 'severity': 0.1854, 'sentiment': 0.3113, 'ticket_hit@1': 0.1126, 'ticket_hit@5': 0.0132, 'article_hit@3': 0.0331}

## 3. Generation / RAG

Sample sizes: {'in_domain_test': 40, 'gold': 101, 'ood': 12}. Primary generator: `ollama:qwen3:4b-instruct`.

| run | generator | answered | step faithfulness | step halluc. | response halluc. | citation validity | citation precision (gold) | citation coverage | gold-step recall | answer relevance (cos) |
|---|---|---|---|---|---|---|---|---|---|---|
| primary | ollama:qwen3:4b-instruct | 137 | 1.000 | 0.000 | 0.000 | 1.000 | 0.917 | 1.000 | 0.918 | 0.924 |
| extractive_no_llm | extractive_no_llm | 38 | 1.000 | 0.000 | 0.000 | 1.000 | 0.591 | 1.000 | 0.763 | 0.795 |

- `extractive_no_llm`: status {'degraded': 38, 'abstained': 2}, latency {'avg_ms': 168.7, 'p50_ms': 43.5, 'p95_ms': 588.0, 'n': 40}

Optional LLM judge: {'n': 20, 'judge_model': 'qwen3:4b-instruct', 'faithfulness_1to5': 4.9, 'relevance_1to5': 5.0}

## 4. End-to-end

| metric | value |
|---|---|
| in-domain n | 141 |
| successful resolution rate (status=resolved) | 0.965 |
| correct resolution rate (resolved and cites gold-relevant source) | 0.929 |
| false abstention rate (in-domain) | 0.028 |
| unreliable rate | 0.007 |
| out-of-domain n | 12 |
| correct abstention rate (out-of-domain) | 1.000 |
| overall abstention rate | 0.105 |
| latency total (ms) | {'avg_ms': 4390.8, 'p50_ms': 4901.7, 'p95_ms': 6148.8, 'n': 153} |
| latency generation stage (ms) | {'avg_ms': 4855.3, 'p50_ms': 4901.0, 'p95_ms': 6117.5, 'n': 137} |
| failures | {'pipeline_errors': 0, 'retrieval_failures': 0, 'generation_degraded': 0, 'citation_validation_failures': 1} |

In-domain status counts: {'resolved': 136, 'abstained': 4, 'unreliable': 1}; OOD status counts: {'abstained': 12}.

## 5. Evolving data

Two new intents (['esim_activation', 'international_roaming']) with 2 KB articles and 12 tickets were added at runtime through the normal services - no restart, no index rebuild.

| stage | Hit@1 | Hit@5 | MRR | intent accuracy | resolve status |
|---|---|---|---|---|---|
| before ingestion | 0.000 | 0.000 | 0.000 | 0.000 | {'degraded': 11, 'abstained': 1} |
| after ingestion | 0.500 | 0.917 | 0.607 | 0.500 | {'degraded': 12} |

Timings: {'add_taxonomy_labels_s': 0.04, 'ingest_2_articles_and_12_tickets_s': 0.09, 'single_ticket_ingest_ms': 27.6}; corpus version {'before': 40, 'after': 44}; restart required: False.
