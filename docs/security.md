# Security and trust

ResolveIQ shows support agents a resolution assembled from text that many people wrote: customers (complaints), earlier agents (tickets) and KB
authors (articles). That text is untrusted. This page states the threat model, what the code does about each threat, which test proves it, and what
is still open. The controls that are not specific to the language model (API keys, least-privilege database role, TLS, secrets) are in
[`production.md`](production.md) section 2 and are not repeated here.

**The rule the design follows: retrieved text is data.** It can inform an answer. It can never become an instruction to the model, a citation, a
link the agent is told to open, or a line in a log.

## Threats, controls, evidence

| Threat | Control (where) | Proven by (`backend/tests/test_security_hostile.py` unless noted) |
|---|---|---|
| **Prompt injection through a stored ticket or KB article** ("ignore previous instructions", forged `SYSTEM:` lines, a forged evidence header) | At ingestion, instruction-like text is flagged (`security_flags=["instruction_like"]` in the document metadata, a warning in the ingest response). At prompt time every piece of evidence passes `neutralise_evidence` (`core/text.py`): bracketed ids `[KB-999]` become `(KB-999)` so text cannot forge an id the model might cite, role and delimiter tokens and code fences are removed, invisible characters are removed, and instruction-like sentences are replaced by `[removed: instruction-like text]`. The system prompt's rule 8 says evidence is untrusted data. | `test_instruction_like_documents_are_flagged_when_ingested`, `test_hostile_evidence_never_reaches_the_model_and_cannot_steer_the_answer` (an obedient fake model that does whatever the evidence says cannot produce the attacker's answer, because the attacker's text never reaches it) |
| **Prompt injection through the complaint** | The complaint is redacted, length-limited and quoted as JSON data under a "(data, not instructions)" header; instruction-like phrases are flagged (`resolveiq_prompt_injection_flagged_total`, a response warning, `lineage.complaint.injection_flagged`). The model has no tools and no side effects, so the worst outcome of a successful injection is a wrong answer, which the validator checks. | `test_a_hostile_complaint_is_flagged_quoted_as_data_and_cannot_extract_the_prompt` |
| **A model that is already compromised or simply wrong** (invents ids, links, phone numbers, amounts) | The citation validator (`rag/citations.py`) removes any cited id that was not retrieved and reports it; flags steps without a citation; flags steps not supported by the cited text; and `invented_specifics` rejects any link, e-mail address, phone number or money amount in a step that does not appear in the cited evidence. Any failure marks the response `unreliable`, forces escalation and lowers the confidence. | `test_a_compromised_model_cannot_smuggle_ids_links_or_money_past_validation`, `test_invented_specifics_check_accepts_what_the_evidence_says_and_rejects_the_rest` |
| **Citation fabrication shown to the agent** | The lineage is built from the validated result, and its checks include `every_citation_maps_to_a_retrieved_source` and `no_invalid_citation_survived`. The evidence graph on screen draws only sources that were retrieved. | `test_lineage_maps_every_citation_to_a_real_retrieved_source` in `tests/test_provenance_lineage.py`, and the two tests above |
| **Malformed, huge or hostile evidence** (control characters, markup, enormous documents) | Documents are normalised and size-limited at ingestion; the prompt builder clips each source and the total to a token budget; collapsed duplicates and over-budget sources are reported in `provenance.prompt`, not silently dropped. | `test_malformed_and_oversized_evidence_is_bounded_and_never_breaks_the_prompt`, `test_ingestion_rejects_oversized_documents_and_strips_control_characters` |
| **SQL / query-syntax injection** in complaints, search queries, labels, feedback | Every statement is parameterised; the lexical search tokenises the complaint with Postgres' own text-search parser and passes the resulting lexemes as bound parameters, never raw user syntax. | `test_sql_and_query_syntax_in_every_input_is_inert` |
| **PII leakage** into storage, prompts, cache, responses or logs | Redaction happens before anything is stored, embedded, cached or sent to the model. Free text from feedback, replay and the retrieval lab is never logged. Spans carry lengths and ids, never text. | `test_pii_is_redacted_before_storage_prompts_cache_responses_and_logs`, `test_free_text_in_feedback_replay_and_lab_is_never_logged_or_leaked`, `test_no_complaint_text_or_prompt_ends_up_in_any_span` in `tests/test_tracing.py` |
| **Stored instructions in agent feedback** | Feedback is stored and displayed as text; it is never executed, never fed back into prompts, and the improvement report is advisory only. | `test_feedback_text_that_tries_to_instruct_the_system_is_stored_inertly` |
| **Secret or internal detail exposure** (connection strings, keys, stack traces) | Errors use one envelope with a trace id; internal exceptions return a generic message; the system status endpoint reports versions and booleans, never URLs with credentials. | `test_no_response_exposes_connection_strings_keys_or_stack_traces`, `test_an_internal_error_returns_a_generic_envelope_with_a_trace_id` |
| **Missing authorization** on the new console endpoints | Every `/api/v1` route, including cases, replay, retrieval lab, clusters, quality, system and evaluation results, sits behind the same API-key dependency. | `test_every_api_route_requires_a_key` (enumerates the routes from `/openapi.json`, so a new unprotected route fails the test), `test_ops_endpoints_are_either_harmless_or_token_protected` |
| **API schema exposure** in production | `/docs`, `/redoc` and `/openapi.json` are not served when `APP_ENV=production` unless `EXPOSE_API_DOCS=true`. | `test_api_documentation_is_hidden_in_production` |

## What the console shows, and what it deliberately does not

The evidence graph, case replay and quality views display **safe signals**: ids, scores, similarities, ranks, reranker probability, evidence strength, agreement
among sources, citation coverage, grounding score, abstention reason, and the model, prompt, taxonomy and corpus versions. They do not display model
reasoning or chain-of-thought; the model is asked for a JSON answer only, and no reasoning text is stored. Case and feedback rows contain the redacted
complaint and the redacted resolution, as the audit trail always did, and nothing more. A replay is computed from the stored redacted text and is not
persisted.

## Known gaps (not hidden)

* **Poisoned but not instruction-like content is not detected.** A ticket that confidently states a wrong fix (or a wrong support phone number that also appears in
  the same poisoned article) reads like a normal ticket. The defences above stop it from steering the *format* of the answer and from inventing links or numbers
  that no evidence contains; they cannot tell that a stored fact is false. The control for this is process: who may write to the corpus, review of KB changes,
  and the feedback loop that surfaces sources agents reject ("weak KB articles" in the Feedback and quality view).
* The injection detector is a pattern list plus structural rules, not a classifier. A paraphrased attack that matches none of the patterns reaches the prompt as
  quoted data; the system prompt, the JSON schema and the validator are then what hold. The tests use obvious attacks, and a determined attacker was not
  simulated.
* PII redaction is regex based (e-mail, phone, Luhn-valid cards, labelled account and customer ids, national-id patterns, IPv4). Names and street addresses are
  not removed. A named-entity model would be the next step.
* Authentication is a shared API key per role with per-key rate limiting; there is no per-user identity, so feedback and case access cannot be attributed to
  a person and there is no tenant isolation.
* The security tests use fake models that misbehave on purpose. How a real 4B model reacts to the sanitised evidence was observed on the evaluation sets but
  no red-team campaign was run against it.
* TLS, secret rotation and network policy are described in [`production.md`](production.md) and [`kubernetes.md`](kubernetes.md); only the Kubernetes network policy
  and pod-security settings were exercised, on a single-node kind cluster.
