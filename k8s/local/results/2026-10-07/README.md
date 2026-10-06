# Kubernetes re-verification on the current code (kind, 2026-10-07)

The 2026-10-06 runs (`../2026-10-06/`) were recorded on commit `b57fe0c` (recorded there as `2d32e72`, before the history was rewritten). This folder
re-runs the same harness on the code that followed: the console, case replay, adaptive retrieval, prompt and database work (migrations 003 and 004)
plus the post-submission changes (weighted new-topic correction, pinned dependencies). Raw console output; nothing edited.

Environment: kind v0.30.0, Kubernetes v1.34.0, one node, Docker Desktop (15.5 GB VM) on Windows 11 / WSL2. The LLM was `qwen3:4b-instruct` on
Ollama 0.35.1 in a GPU container on the host (port 11434), reached from the pods through `host.docker.internal`.

| File | What it is |
|---|---|
| `e2e_full_via_ingress.txt` | `bash k8s/local/deploy.sh --ingress` then `python k8s/local/e2e_test.py --ingress`: **22 checks, 22 passed** |
| `e2e_llm_down_via_ingress.txt` | the Ollama container stopped (connection refused), `e2e_test.py --degraded --ingress`: **16 checks, 16 passed**; readiness stayed 200 and the answer was `degraded` / `extractive` with valid citations |
| `cluster_state.txt` | pods, services, HPA, PDB, Ingress, policies, PVC, `kubectl top`, image ids, applied migrations, and the package versions inside the backend pod |
| `migrate_first_attempt_connection_timeout.txt` | the migrate Job's first pod timed out connecting to Postgres; the Job's retry succeeded 2 minutes later. This is the kindnet policy-sync lag described in `docs/kubernetes.md` section 6 |

What this run adds to the 2026-10-06 evidence:

* **An in-place upgrade.** The cluster's database was created by the earlier code (migrations 001 and 002). The migrate Job applied 003 and 004 to it
  and skipped the seed because the corpus was present; the new API and worker pods then served it. The earlier test data (extra tickets and classes)
  was still there, which is why the corpus shows more than 250 tickets.
* **Pinned dependencies.** The image was built with `backend/constraints.txt`. Before this change the image built 14 hours earlier already ran
  `fastapi` 0.142.2, `torch` 2.14.1 and `transformers` 5.18.0, while the evaluation ran 0.141.1, 2.8.0 and 5.17.0. The pod now reports the same
  versions as the evaluation environment (`cluster_state.txt`).

Not repeated here (still only in `../2026-10-06/`): the rolling-restart availability probe, the taxonomy propagation check, the packet-dropping LLM
variant and the NetworkPolicy connection matrix. The manifests those runs exercised did not change.
