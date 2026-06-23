# datagen

Synthetic deep-research **task generation**. Produces tasks with ground truth
baked in, so the output doubles as **training and eval data** (it conforms to the
`dr_agent` scoring schema, so a generated row scores unchanged in both).

## Layout (core + domains)

```
src/datagen/
  core/             stage base classes (BaseExplorer/Verifier/Distractor/Extender) + Task
  domains/<name>/   one folder per domain, identical stage-named layout:
                    explore.py · verify.py · distract.py · extend.py
                    prompts.py · seeds.txt · __main__.py (orchestrator)
outputs/            gitignored generated data, split raw/ -> verified/ -> final/
```

Add a domain = copy `domains/web/`'s layout, subclass `core/` stages. Heavy
per-domain deps go in `pyproject.toml` extras (`[sec]`, `[indexing]`, `[all]`).

## Pattern notes (from chroma context-1-data-gen, improved)

- Stages are **independently runnable** modules + a `__main__.py` orchestrator.
- Unlike the reference repo, stages write to **separate** `raw/verified/final`
  dirs (not mutate-in-place) for clean partial reruns and provenance.
- Prompts are versioned with the code in `prompts.py`.
