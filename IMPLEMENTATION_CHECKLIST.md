# Biomedical RAG implementation checklist

Working basis: Clinical_RAG_Proposal_Revised_v2.docx, revised 15 November 2026, at `/Users/sai/Downloads/Clinical_RAG_Proposal_Revised_v2.docx`.

Purpose: complete the proposed study with traceable evidence, prioritising the data audit and pilot before the main experiment. This checklist implements the proposal; it does not replace the protocol or preregistration. The protocol clarifications below remain open until documented. Recommendations from the review are not automatically adopted protocol changes.

## Status and working rules

- [x] Review proposal v2 and compare it with the previous proposal.
- [x] Create this implementation checklist.
- [ ] Inventory any existing thesis code, data access, hardware and credentials. None has been verified for this project yet.
- [ ] Establish a run ledger with task, priority, status, configuration or commit, output path, validation evidence, and any blocker.
- [ ] Use statuses `not started`, `in progress`, `blocked`, and `verified`; check an item only when its stated evidence exists.
- [ ] Record exact counts, measured costs and observed runtime. Label estimates as estimates and unavailable results as unavailable.
- [ ] Trace every confirmatory result to a frozen configuration, input manifest, raw output and analysis command.
- [ ] Keep raw data immutable; store transformations and exclusions with reasons and source identifiers.
- [ ] Maintain a decision log separating proposal requirements, resolved clarifications, exploratory work and deviations.
- [ ] Do not inspect main-run outcomes to tune thresholds, prompts, chunking or sample selection.
- [ ] Do not describe evidence misses as proof of insufficiency, high faithfulness as confidence, or benchmark results as clinical safety.

Priority order: P0 protocol and data blockers; P1 pilot and preregistration; P2 main execution and analysis; P3 manuscript and release. Work within each phase can proceed independently where dependencies allow. Do not expand the study to additional models, datasets or retrieval methods without a recorded scope decision.

## P0 — Resolve protocol details before freezing implementation

Evidence for completion: a versioned decision log, configuration schema and amended protocol that agree. Clarifications may be developed during pilot preparation; all must be fixed before preregistration.

- [ ] Resolve the tokenizer/context-reuse conflict: either use identical delivered context across generators under one designated budgeting tokenizer, or recompute evidence coverage and context recall for model-specific contexts. Record which secondary-generator interpretation follows.
- [ ] Name the tokenizer and revision defining all chunk-size limits, independently of the context-budget tokenizer.
- [ ] Define the semantic chunker's percentile calculation, pilot calibration grid, selection criterion, tie-breaks and no-qualifying-setting fallback.
- [ ] Define the short-document semantic cutoff and its relationship to the per-document percentile.
- [ ] Specify precedence for minimum length, maximum length, merging, forced boundaries and documents shorter than the minimum.
- [ ] Define canonical source text and provenance as document ID, section and character offsets; BEC must use source-span intersections, not lexical similarity.
- [ ] Specify BEC's denominator after alignment exclusions, whitespace/Unicode normalisation, ambiguous matches, repeated headers and duplicate spans.
- [ ] Separate formatting validity, parsed final answer, substantive answering and abstention; define the registered handling of unparseable substantive answers.
- [ ] Define missing-score handling for every M1–M4 numerator and denominator, including bounds/sensitivity results.
- [ ] Specify BioASQ yes/no versus factoid sampling proportions and the dataset–task standardisation weights, including after sample expansion.
- [ ] Resolve dataset/type coding using observed strata or an equivalent identifiable model; avoid nonexistent dataset–task combinations.
- [ ] Define H1/H3 component tests, intersection–union family p-values, family-level Holm correction and the gate for H2 pairwise tests.
- [ ] Specify robust GEE covariance, standardisation, omnibus contrast computation and the complete Firth fallback inference procedure.
- [ ] Define power alternatives across all three arms and simulate the actual confirmatory testing procedure; assess equivalence precision/power separately.
- [ ] Define factoid normalisation and accepted gold synonyms without an ambiguous 'first-ranked answer' rule.
- [ ] Define oracle snippet ordering, deduplication and token/truncation rules; define R0 truncation against remaining context capacity.
- [ ] Clarify that H1 evaluates the prespecified score direction; report ties and any reversed discrimination without post hoc redefinition of H1.
- [ ] Confirm the 1,900-question cost model is a baseline; define cost scaling for an expanded BioASQ sample and the no-context GPU workload.
- [ ] Verify model/data licences, access conditions and primary judge snapshot availability; record a contingency rather than silently substituting unavailable components.

## P0 — Repository and reproducible environment

Proposal links: Sections 5.1, 5.5–5.10 and 8–9.

- [ ] Inventory the workspace and applicable project instructions before adding implementation code.
- [ ] Establish version control and a simple structure for configuration, source manifests, processing, retrieval, generation, judging, analysis and reports.
- [ ] Keep secrets and restricted/downloaded data out of version control; verify ignore rules.
- [ ] Pin Python packages, model/tokenizer revisions, RAGAS version, prompts, inference backend and analysis libraries.
- [ ] Record hardware, precision, quantisation, seeds and relevant deterministic settings.
- [ ] Build resumable stages with stable item IDs, configuration hashes and explicit retry policies.
- [ ] Distinguish failed calls, missing inputs and excluded items from valid zero-valued measurements.
- [ ] Establish a small end-to-end fixture run before spending on large generation or judging jobs.

Completion evidence: environment lock, configuration manifest and a reproducible fixture command with saved outputs.

## P0 — Phase 0 data access and evidence audit

Proposal link: Section 5.3. This phase determines the eligible pool; do not assume the proposed counts are achieved.

- [ ] Obtain BioASQ 13b training data under its access terms; record release, source, download date and checksum.
- [ ] Verify the actual training count and exact-answer fields; inventory eligible yes/no and factoid questions.
- [ ] Extract and deduplicate all relevant PMIDs across the training set for corpus construction.
- [ ] Fetch titles and abstracts with resumable requests, caching and a manifest of unavailable or empty records.
- [ ] Audit every candidate gold snippet against its source document and section.
- [ ] Re-anchor offsets only under the documented exact-match normalisation policy; flag ambiguous matches.
- [ ] Exclude snippets outside indexed title/abstract text and log reasons; merge overlapping gold spans by document and section.
- [ ] Retain questions only when at least one gold snippet aligns; report retained evidence and question counts by type.
- [ ] Preserve the distinction between unavailable corpus evidence and retriever misses.
- [ ] Obtain PQA-L and PQA-U; record releases/checksums and verify actual counts.
- [ ] Build PubMedQA retrieval text from contexts only; exclude titles, long answers, decisions and other answer-bearing metadata from indexed/delivered fields.
- [ ] Audit duplicate PMIDs, duplicate text and overlap between PQA-L and PQA-U; document deduplication policy.
- [ ] Partition pilot and main question IDs reproducibly; exclude pilot questions from main outcome analysis. Pilot source documents may remain in the retrieval corpus according to the fixed corpus policy.
- [ ] Reserve 100 PQA-L questions for the pilot and 900 for the baseline main sample; reserve 100 eligible BioASQ questions for the pilot and 100 PQA-U questions for pilot checks.
- [ ] Produce an eligibility flow report covering questions, unique articles, aligned/re-anchored/dropped snippets and exclusions.
- [ ] If BioASQ access fails, document the blocker. Treat rag-mini-bioasq as a protocol-changing fallback, not an equivalent replacement.

Gate: validated canonical corpora, gold-span maps, exclusions and pilot/main manifests exist. Sample size is still provisional until the pilot.

## P1 — Chunking and retrieval

Proposal links: Sections 5.4–5.5.

- [ ] Implement C1: consecutive 128-token windows, no overlap, with source-span provenance.
- [ ] Implement C2: greedy sentence packing to 128 tokens, no overlap; record long-sentence fragmentation.
- [ ] Implement C3: MedCPT article-encoder sentence embeddings, calibrated semantic breakpoints and resolved 64–192-token rules.
- [ ] Pin a common sentence splitter and define title/body segmentation for all arms.
- [ ] Implement R0 whole-document retrieval units with the resolved truncation policy.
- [ ] Implement R1 direct gold snippets/source context and R2 question-only generation.
- [ ] Implement MedCPT CLS embeddings in fp32; pin 512-token article and 64-token query truncation, raw inner product and exact FAISS IndexFlatIP search.
- [ ] Encode/deliver BioASQ titles as specified and exclude PubMedQA titles in every applicable arm.
- [ ] Assemble contexts in rank order, skipping non-fitting units through rank 50 under the 1,024-token budget including headers/separators.
- [ ] Save ranked candidates separately from the exact context delivered to the generator.
- [ ] Save all delivered source spans, chunk IDs, ranks, budget decisions and actual lengths in both tokenizers.
- [ ] Implement fixed-top-k = 5 sensitivity for C1–C3.
- [ ] Implement C1 16-token-overlap sensitivity with the specified span-merging/context-assembly policy.
- [ ] Report per-arm chunk lengths, chunks per article, fragmentation, title contribution and C3 forced-boundary rates.

Validation evidence: fixtures establish correct offsets, no unintended overlap, expected raw-inner-product ranking, budget compliance and exact agreement between recorded context and generator input.

## P1 — Generation and measurement

Proposal links: Sections 5.2 and 5.6–5.8.

- [ ] Configure Mistral-7B-Instruct-v0.3, official chat template, greedy decoding and 256-token generation limit.
- [ ] Configure Llama-3.1-8B-Instruct for the registered stratified 500-question subsample.
- [ ] Freeze the prompt requesting a short explanation and final `ANSWER:` line with task-appropriate labels or `INSUFFICIENT EVIDENCE`.
- [ ] Save full prompts, raw responses, finish reasons, timing, model revision and generation metadata.
- [ ] Implement last-ANSWER-line parsing, abstention, malformed-output and contradiction flags under the resolved policy.
- [ ] Implement exact label correctness and factoid synonym matching; keep the contains-synonym rule as sensitivity only.
- [ ] Calculate gold-document hits and BEC from the actual delivered spans; record zero, partial and higher coverage.
- [ ] Implement M1–M4 with explicit counts and denominators; distinguish M4 from masked incorrect answers because M4 does not require an evidence miss.
- [ ] Report abstention under evidence misses and gold-document hits, without labelling all such abstentions correct or incorrect.
- [ ] Implement full-response RAGAS faithfulness and reference-based context recall with pinned judge prompts/version/snapshot.
- [ ] Implement the separate decisive-answer support check; freeze how the question and label become a declarative claim.
- [ ] Exclude R2 from faithfulness scoring; retain correctness and abstention results.
- [ ] Cache context recall only when its complete inputs are identical.
- [ ] Save raw judge calls, extracted claims, judgements, parse failures, retries, token usage and cost.
- [ ] Implement the 10% twice-rescored reliability sample and 20% stratified Qwen2.5-14B second-judge sample.
- [ ] Implement ICC(2,1), threshold agreement/kappa and mean absolute differences with fixed sampling rules.
- [ ] Apply the Joren sufficient-context prompt to evidence misses only as the registered sensitivity analysis; save raw and adjusted results separately.

Validation evidence: meaningful fixtures cover missing/partial/complete gold coverage, duplicate spans, headers, invalid labels, abstentions, synonyms, contradictory responses and judge failures. No benchmark score is fabricated or substituted for a failed measurement.

## P1 — Pilot and preregistration

Proposal link: Section 5.10.

- [ ] Run the held-out 300-question pilot with all checks needed to validate the planned pipeline.
- [ ] Calibrate C3 using only the designated pilot documents and record the selected configuration.
- [ ] Measure miss rates, answered/scored misses, missingness, parsing errors and within-question overlap/correlation across arms.
- [ ] Inspect failures through logged examples; distinguish pipeline defects from legitimate benchmark events.
- [ ] Measure generation throughput, embedding time, memory, judge calls/tokens per item and actual expenditure.
- [ ] Simulate registered analysis and power for sample sizes from 1,000 BioASQ questions to the eligible non-pilot pool, keeping the PQA-L main sample fixed.
- [ ] Select the smallest permitted sample meeting the registered power rule; if none does, use the eligible maximum and report prospective achieved power under the stated assumptions.
- [ ] Assess whether the ±10-point equivalence conclusion is realistically supported by expected precision.
- [ ] Recalculate full workload and budget, including sensitivity runs, rescoring, secondary models, retries and sample expansion.
- [ ] Check feasibility against the proposed USD 150 judge budget and USD 50 paid-GPU fallback; record a scope decision if measured needs exceed them.
- [ ] Finalise all exclusions, prompts, thresholds, model versions, contrasts, multiplicity, missingness, fallback and stopping rules.
- [ ] Freeze main question IDs, secondary-model IDs, configurations, code and analysis scripts.
- [ ] Register the protocol and analysis plan on OSF before any main outcome run; save the registration link, timestamp and code commit.

Gate: pilot validation passes, scope is feasible, protocol ambiguities are resolved, and preregistration exists. Only then start the main experiment.

## P2 — Main experiment

- [ ] Generate primary-model outputs for every registered question across C1–C3 and R0–R2.
- [ ] Run the second generator on the registered 500-question subsample across registered arms.
- [ ] Run fixed-top-k and overlap sensitivities under frozen settings.
- [ ] Reconcile expected versus completed item counts for every dataset, task, arm and generator.
- [ ] Score the five context-bearing arms and planned sensitivities; perform registered reliability, second-judge and sufficiency checks.
- [ ] Monitor only completion, technical failures, token use and costs; do not tune the study from emerging effects.
- [ ] Retry only under the frozen policy; retain original failures and every attempt in the ledger.
- [ ] Record any unavoidable deviation with date, reason and affected items; separate amended/exploratory results from confirmatory results.
- [ ] Freeze the final analysis dataset with hashes and an exclusion/missingness reconciliation.

Gate: complete and reconciled run ledger, immutable raw outputs and a validated analysis dataset.

## P2 — Analysis and interpretation

Proposal link: Section 5.9.

- [ ] Report evidence coverage, source hits, parsing, missingness, abstention, correctness and raw M1–M4 counts before modelling.
- [ ] Fit the registered H2 GEE and compute standardised rates, omnibus test and pairwise risk differences under the multiplicity plan.
- [ ] Compute H1/H3 within-dataset AUROCs on the same eligible answered/scored cases, with explicit eligibility counts and paired uncertainty estimates.
- [ ] Run the registered stratified question bootstrap with 2,000 replicates and report unsuccessful fits/replicates.
- [ ] Estimate dataset/task contrasts and the secondary Failure × Arm and Failure × Arm × Dataset relationships.
- [ ] Run shared-failure, document-component clustering, GLMM, missing-score, thresholds, partial-coverage, top-k, overlap, second-generator/judge and sufficient-context sensitivities.
- [ ] Distinguish joint masked-and-incorrect events from M4, which includes high-faithfulness incorrect answers with evidence hits too.
- [ ] Apply the registered equivalence criterion; call imprecise null findings inconclusive.
- [ ] Keep length-adjusted models and unregistered analyses explicitly exploratory.
- [ ] Check whether conclusions change across datasets, judges, generators or eligibility definitions; report disagreement rather than choosing the favourable result.

Completion evidence: reproducible tables/figures, analysis logs and a claim-to-result map with intervals, denominators and limitations.

## P3 — Manuscript and reproducibility release

Proposal links: Sections 6–11.

- [ ] Update the targeted literature search and preserve a dated search log; verify citations against primary sources.
- [ ] Write methods from the executed registered protocol and clearly disclose deviations.
- [ ] Report audit attrition, final sample sizes, actual resource use, all confirmatory outcomes and material missingness.
- [ ] Describe the study as biomedical literature QA evaluation, not clinical validation or a universal chunker ranking.
- [ ] Release permitted code, configurations, manifests, audit outputs, generations, judge outputs and reproduction commands under applicable terms.
- [ ] Check release files for secrets and redistribution restrictions; do not assume public abstracts have unrestricted redistribution rights.
- [ ] Obtain the planned supervisor/peer read and address substantive feedback.
- [ ] Select the venue and verify its current submission requirements, deadlines and any publication fees.
- [ ] Prepare the submission and preprint; complete any required author/institution decisions before external submission.

## Immediate priority queue

1. Inventory existing code, BioASQ access and compute resources.
2. Resolve canonical text/provenance and context-budget/tokenizer decisions.
3. Acquire data and produce the Phase 0 eligibility report.
4. Build one traceable end-to-end fixture, then the held-out pilot.
5. Freeze the protocol only after pilot validation and registered-analysis power simulation.

The study is complete when the registered experiment and required analyses are reconciled, conclusions are supported by saved evidence, the reproducibility package is ready, and the proposed manuscript/submission milestones have been addressed. Passing a smoke test or obtaining a favourable result is not completion.
