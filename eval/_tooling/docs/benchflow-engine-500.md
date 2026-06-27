# BenchFlow agentic run → `provider api error / HTTP 500` with a self-hosted engine

**Status:** ✅ **RESOLVED.** Root cause found and fixed in the `make eval` driver
(`_tooling/Makefile`). Kept here as the investigation record because the symptom
(a "transient HTTP 500" on a self-hosted engine) is misleading and worth knowing.

## TL;DR

The full agentic BenchFlow path against a **self-hosted** engine (SGLang/vLLM)
failed: **every** model call came back to the agent as `provider api error
[provider_error/transient] HTTP 500`, every task errored, reward `0`.

It is **not** a network/proxy/streaming bug. The real cause:

> **benchflow's pi-acp registers the model with `maxTokens = 16384` by default**
> (`agents/pi_acp_launcher.py::_DEFAULT_MAX_TOKENS`) whenever `BENCHFLOW_PROVIDER_MODELS`
> carries no explicit `maxTokens`. pi then asks for a **completion of 16384 tokens** —
> the model's *entire* context. With any prompt, `prompt + 16384 > 16384`, so SGLang
> rejects **every** call with a context-length error. litellm **mislabels** that
> deterministic rejection as a *transient* `APIConnectionError` → the proxy returns
> HTTP 500 → pi retries 12× → litellm Router cools the (single) deployment down →
> all tasks fast-fail. The "transient 500" is a context-overflow 400 in disguise.

**The proof** — the proxy's own `stderr.log`, captured live before benchflow deletes it:

```
APIConnectionError: OpenAIException - Requested token count exceeds the model's
maximum context length of 16384 tokens. You requested a total of 17964 tokens:
1580 tokens from the input messages and 16384 tokens for the completion.
```

## The fix

Inject a model entry with a sane `maxTokens` so pi caps its completion request well
under the context window. The `make eval` driver now does this for every `vllm/`
(self-hosted) model:

```make
ENGINE_CONTEXT ?= 16384
ENGINE_MAXTOK  ?= 2048
ENGINE_MODELS  := [{"id":"$(MODEL_ID)","name":"$(MODEL_ID)","contextWindow":$(ENGINE_CONTEXT),"maxTokens":$(ENGINE_MAXTOK),"reasoning":false,"input":["text"]}]
# --agent-env 'BENCHFLOW_PROVIDER_MODELS=$(ENGINE_MODELS)'
```

benchflow's `_provider_models_for_proxy_alias` clones that entry onto the LiteLLM
alias pi sees in proxy mode, so the cap survives routing. With `maxTokens=2048`:
`1580 + 2048 = 3628 < 16384` → SGLang returns 200, pi tool-calls, the verifier scores.

**Verified:** a 2-task `make eval` run went from `errored=2` (all HTTP 500) to
`errored=0, n_tool_calls=1, reward=0.0` (real answers, model just gets them wrong —
that's the 2B *base* model's capability, not a plumbing fault).

> Note: raising `ENGINE_CONTEXT` does **not** help on its own — pi would just request
> the larger number as its completion and overflow again. The cap on `maxTokens` is
> the fix. Keep `ENGINE_MAXTOK` comfortably below `ENGINE_CONTEXT − max_prompt_tokens`.

## Why the earlier investigation pointed the wrong way

Every isolation test (SGLang direct, litellm lib, the proxy with benchflow's exact
config + callback, even from *inside a container* over `host.docker.internal`)
returned **200** — so suspicion fell on the container↔host streaming path under
Docker-Desktop-WSL. That was a **red herring**: those replays used pi's *logged*
request body, where `max_tokens` was `None` (so SGLang applied a small default). The
**live** pi-acp instead sends `max_tokens: 16384` from its model-registry default,
which the logged trajectory didn't surface. The container topology works fine; the
request itself was over-budget.

The misclassification compounds it: SGLang's context-length error *should* be a
non-retryable `BadRequestError`, but litellm tags it `APIConnectionError /
provider_error/transient`, so benchflow retries 12× and cools the deployment down —
making one bad request look like a flaky network.

## How to confirm on a fresh run

benchflow writes the proxy's logs to an ephemeral `/tmp/benchflow-litellm-*/` and
**deletes them on teardown**. To read them, grab them *while a run is in progress*:

```bash
# during a live `make eval`:
tail -n +1 /tmp/benchflow-litellm-*/stderr.log | grep -i "context length\|token count"
```

A context-overflow line there = this bug. (No line + a real network error = a
different problem.)

## Symptom signature (for future triage)

- Agent side: `api_error_info.status_counts = {"500": N}` for all N calls, `reward 0`,
  `error_category: api_error`.
- `…/trajectory/llm_trajectory.jsonl`: each `response.status_code = 500`,
  `response.body = {"raw": None, "error": {"type": "NoneType", "message": "None"}}`
  (benchflow's failure callback fired with `response_obj=None`).
- Timing tell: first request ~400ms (a real attempt), the rest ~30ms (litellm Router
  **cooldown** fast-fails) — a uniform, *instant* 500 storm, not random flakiness.
- Engine returns 200 to a hand-replayed request → the difference is the **request**
  (its `max_tokens`), not the route or the network.

## Upstream-worthy notes (benchflow-ai)

1. `pi_acp_launcher._DEFAULT_MAX_TOKENS = 16384` is a poor default for a 16k-context
   model: it equals the whole window, guaranteeing overflow on the first token of
   any prompt. A default like `min(4096, contextWindow // 4)` would be safer.
2. litellm classifying a context-length error as `provider_error/transient` causes a
   12× retry storm + deployment cooldown on what is a permanent 400 — worth a
   non-retryable mapping.
