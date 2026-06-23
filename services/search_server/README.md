# search-server

The tool/search HTTP service for the deep-research agent. It exists so that:

- **heavy retrieval deps** (BM25, faiss, sentence-transformers) are isolated from
  `libs/dr_agent` and from the trainer — no faiss-vs-vllm dependency hell;
- the agent reaches tools over **one stable contract**, so backends (BM25 ↔ dense
  ↔ live web) are swappable without touching the agent or the RL loop;
- RL rollouts can hit a **self-hosted, reproducible** index instead of a
  rate-limited live API (the QUEST/Search-R1 pattern).

## Contract

```
POST /search   {"query","top_k"}  -> {"results":[{"id","title","snippet","url"}]}
POST /retrieve {"query","top_k"}  -> {"passages":[{"id","contents"}]}
POST /visit    {"url"}            -> {"content"}
GET  /health                     -> {"status":"ok"}
```

## Run

```bash
make build-index CORPUS=corpus.jsonl OUT=./index   # one-time, for bm25/dense
make serve                                         # serves on :8000
```

Consumed by `dr_agent.tools` (set `AgentConfig.tool_server_url`).

> Why `services/` and not `infra/`: `infra/` is for deployment/infrastructure;
> this is a runtime application service.
