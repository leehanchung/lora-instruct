# web domain

Synthetic multi-hop web-search tasks. Stage layout (identical across domains):

| File | Stage | Output |
|---|---|---|
| `explore.py` | 1. draft question + truth + supporting items | `outputs/web/raw/` |
| `verify.py` | 2/4. validate quotes entail the truth | `outputs/web/verified/` |
| `distract.py` | 3. mine hard distractors | — |
| `extend.py` | 5. chain into multi-hop (level += 1) | `outputs/web/final/` |
| `prompts.py` | LLM prompt templates | — |
| `seeds.txt` | input seed topics | — |

Run the whole pipeline:

```bash
uv run python -m datagen.domains.web --seeds src/datagen/domains/web/seeds.txt --out outputs/web
```

To add a new domain (e.g. `sec`), copy this folder's file layout and subclass the
`core/` stage base classes.
