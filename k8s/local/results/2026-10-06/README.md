# Kubernetes test results (kind, 2026-10-06)

Raw console output of the runs quoted in [`docs/kubernetes.md`](../../../../docs/kubernetes.md). Environment: kind v0.30.0, Kubernetes v1.34.0, one node,
Docker Desktop (15.5 GB VM) on Windows 11 / WSL2, Ollama on the host. Nothing here was edited except where a header comment says so.

| File | What it is | Valid? |
|---|---|---|
| `e2e_full_via_ingress.txt` | `e2e_test.py --ingress`: 22 checks, all PASS | yes (final run) |
| `e2e_full_port_forward_earlier.txt` | the same through `kubectl port-forward`, earlier code | yes, superseded |
| `e2e_llm_down_via_ingress.txt` | `e2e_test.py --degraded --ingress` with the LLM unreachable | yes |
| `llm_down_latency_final.txt` | 8 consecutive requests with the LLM refusing connections | yes (small sample) |
| `llm_unreachable_dropped_packets_before_connect_timeout.txt` / `..._after_...` | the packet-dropping variant, before and after `LLM_CONNECT_TIMEOUT_SECONDS` | yes (6 and 8 requests) |
| `taxonomy_propagation_after_fix.txt` | class added via replica A seen by replica B in 8.2 s | yes. The "before" run (B still at version 1 after 40 s) is quoted in `kubernetes.md` but its output was not saved |
| `rollout_before_prestop_port_forward.txt` | rolling restart at 4 req/s: 2 x HTTP 502 in 127 requests | yes |
| `rollout_after_prestop_port_forward.txt`, `rollout_after_prestop_via_ingress.txt` | after the `preStop` delay: 0 of 130 and 0 of 48 failed | yes (small samples) |
| `rollout_invalid_rate_limited.txt` | 429s: my probe exceeded the 120/min per-key rate limit (the limiter is shared across replicas via Redis) | **invalid as an availability measurement** |
| `rollout_invalid_port_forward_and_kindnet_lag.txt` | 181 s rollout, hung requests: new pods were cut off from Postgres until kindnet's next policy sync, and the port-forward path stalled | **invalid** (kept as evidence of kind's limits) |
| `networkpolicy_enforcement.txt` | connection attempts proving which pods can reach which; includes the note about kindnet failing open | yes, after restarting kindnet |
| `cluster_state.txt` | pods, services, HPA, PDB, Ingress, policies, PVC, `kubectl top` at the end | yes |
