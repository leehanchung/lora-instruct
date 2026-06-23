"""eval_harness — two-phase deep-research evaluation.

Phase 1 (generate): run the shared dr_agent loop over a benchmark's tasks and
write rollout JSON artifacts. Phase 2 (score): score those artifacts offline with
the shared dr_agent reward registry. Decoupling them makes scoring cheap to
re-run and audit (pattern from DeepResearcher / StepFun runner).
"""
