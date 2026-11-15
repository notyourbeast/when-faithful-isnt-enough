# When Faithful Isn't Enough

Research project examining high RAGAS faithfulness despite missing benchmark evidence across chunking configurations in biomedical question answering.

## Current status

Planning and repository setup. No dataset audit, pilot, main experiment or research results have been completed in this repository.

The working research specification is `Clinical_RAG_Proposal_Revised_v2.docx`, revised 15 November 2026. The source document currently lives outside this repository. Protocol clarifications remain open and must be resolved before preregistration.

## Work plan

See [IMPLEMENTATION_CHECKLIST.md](IMPLEMENTATION_CHECKLIST.md) for priorities, validation evidence and phase gates.

1. Inventory existing code, dataset access and compute resources.
2. Resolve evidence provenance and context-budget/tokenizer rules.
3. Complete the data and evidence-alignment audit.
4. Implement and validate the held-out pilot.
5. Freeze and preregister the protocol before the main experiment.

## Research discipline

- Distinguish benchmark evidence coverage, answer sufficiency, faithfulness and correctness.
- Mark tasks complete only when supporting evidence exists.
- Preserve raw inputs and outputs with versioned configurations and source identifiers.
- Record deviations; do not silently change the registered experiment.
- Keep credentials, downloaded datasets, model weights and generated run outputs out of Git.
- Do not interpret these experiments as clinical validation.

Dependencies and execution commands will be documented after the environment and existing code have been audited.
