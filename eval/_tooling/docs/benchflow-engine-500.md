# BenchFlow agentic run → `provider api error / HTTP 500` with a self-hosted engine

**Status:** open, root-cause narrowed (not fully pinned). **Workaround:**
`make eval-direct` (a Docker-free debug fallback) while this is open.

## TL;DR

Running the full agentic BenchFlow path against a **self-hosted** engine (SGLang,
or vLLM-with-tool-calls) fails: **every** model call comes back to the agent as
`provider api error [provider_error/transient] HTTP 500`, the task errors, reward
is `0`. The **engine returns 200** on every request — the 500 is manufactured by
BenchFlow's host-side **LiteLLM proxy** relaying to the agent.

Crucially, the failure is **not** in any single component: the engine, the
`litellm` library, the proxy *route*, BenchFlow's logging *callback*, BenchFlow's
*exact proxy config*, and *pi's exact request* all return **200 in isolation on
the host**. The 500 only appears in the live run, where the proxy is bound to
`host.docker.internal` and the agent streams from **inside the Docker sandbox** —
so the remaining suspect is the **container↔host streaming path under Docker
Desktop on WSL**, the same networking class as the proxy-bind patch (see
`README.md` → "Docker-Desktop-on-WSL networking patch").

## First, correct a misconception: there is no `hosted_vllm` route

The run-config says `model: vllm/qwen35-2b-base`. It's tempting to think BenchFlow
routes that through litellm's `hosted_vllm/` provider. **It does not.** BenchFlow's
`vllm/` provider is its generic "self-hosted OpenAI-compatible server" prefix, and
it resolves to litellm **`openai/<model>` + a custom `api_base`**:

```text
# benchflow.providers.litellm_config.resolve_litellm_route("vllm/qwen35-2b-base", …)
model_alias:    benchflow-vllm-qwen35-2b-base
litellm_params: {model: openai/qwen35-2b-base,
                 api_base: http://localhost:30000/v1,
                 api_key: os.environ/OPENAI_API_KEY}
```

The proxy registers four aliases (`benchflow-vllm-…`, `openai/benchflow-vllm-…`,
`qwen35-2b-base`, `openai/qwen35-2b-base`), all → `openai/qwen35-2b-base` @ api_base.
pi is configured to call it as `litellm/benchflow-vllm-qwen35-2b-base`, where
`litellm/` is **pi's provider name** (stripped before the wire); the wire model is
`benchflow-vllm-qwen35-2b-base`.

## Symptom signature

- Agent side: `[ERR] <task> (provider api error [provider_error/transient])`,
  `api_error_info.status_counts = {"500": N}` for all N calls, `reward 0`.
- `…/trajectory/llm_trajectory.jsonl`: each entry `response.status_code = 500`,
  `response.body = {"raw": None, "error": {"type": "NoneType", "message": "None"}}`
  — i.e. the proxy's failure callback recorded an empty response.
- Engine (SGLang) access log: `POST /v1/chat/completions … 200 OK` for the
  requests it received (from `127.0.0.1`, the host-side proxy).
- vLLM "worked" earlier **only because that model never emitted tool calls** — the
  agentic tool path (and its streaming responses) was never exercised.

## Isolation matrix (all run on the host)

| # | Setup | tools | stream | Result |
|---|-------|:----:|:-----:|--------|
| 1 | SGLang direct (`curl :30000`) | ✓ | ✓ | **200**, valid tool_calls (`qwen3_coder` parser) |
| 2 | `litellm` **library**, `openai/…`+api_base | ✓ | ✓ | **200** |
| 3 | `litellm` **library**, `hosted_vllm/…`+api_base | ✓ | ✓ | **200** |
| 4 | plain `litellm` **proxy**, `openai/…` route | ✓ | ✓ | **200** |
| 5 | plain `litellm` **proxy**, `hosted_vllm/…` route | ✓ | ✓ | **200** |
| 6 | proxy + BenchFlow's **callback** registered | ✓ | ✓ | **200** |
| 7 | proxy with BenchFlow's **exact generated config** (4 aliases + callback + master_key + drop_params), wire model `benchflow-vllm-qwen35-2b-base` | ✓ | ✓ | **200** |
| 8 | #7 + **pi's exact recorded request** (system+user msgs, 4 tools, stream) | ✓ | ✓ | **200** |
| 9 | **BenchFlow live run** (proxy bound `host.docker.internal`, pi inside Docker sandbox) | ✓ | ✓ | **500** ✗ |

Everything up to and including pi's exact request, replayed against BenchFlow's
exact proxy config, returns 200. The single thing #9 has that #8 doesn't: the
**proxy is bound to `host.docker.internal` and the client (pi) is inside the
container**, streaming back across the Docker-Desktop-WSL boundary.

(One red herring: sending the literal wire model `litellm/benchflow-vllm-qwen35-2b-base`
to the proxy yields a clean **400 "Invalid model name"** — but pi strips its
`litellm/` provider prefix before the wire, so this isn't what happens in the run.)

## Most likely root cause

A streaming chat-completion relayed **proxy(host) → pi(container) over
`host.docker.internal`** drops / resets mid-stream under Docker Desktop on WSL2
(engine in a separate VM). The proxy sees the client go away mid-stream, aborts
the upstream call, and surfaces it to the agent as a transient provider 500 with
an empty body — which matches the `raw: None` trajectory record. This is the same
class of issue as the proxy *bind-address* problem already patched; reachability
was fixed, but **streaming** across that boundary appears not to survive.

Not yet done: capture the proxy's own `stderr.log` from a live run (BenchFlow
writes it to an ephemeral `/tmp/benchflow-litellm-*/` and **deletes it on
teardown** — instrument `benchflow/providers/litellm_runtime.py` to keep it, or
`tail -F` it during a run) to see the exact proxy-side exception at the moment of
the 500. That is the one piece that would turn "most likely" into "confirmed."

## Reproduction recipe

```bash
PYBF=~/.local/share/uv/tools/benchflow/bin/python
# (a) the engine itself is fine:
curl -s :30000/v1/chat/completions -d '{"model":"qwen35-2b-base","stream":true,
  "messages":[{"role":"user","content":"hi"}],
  "tools":[{"type":"function","function":{"name":"write","parameters":{}}}],
  "tool_choice":"auto"}' -H 'content-type: application/json' | tail -1   # data: [DONE]

# (b) BenchFlow's EXACT proxy config also serves 200 on the host:
$PYBF - <<'PY'
import yaml
from benchflow.providers.litellm_config import resolve_litellm_route, litellm_proxy_config
r = resolve_litellm_route("vllm/qwen35-2b-base",
      {"BENCHFLOW_PROVIDER_BASE_URL":"http://127.0.0.1:30000/v1","OPENAI_API_KEY":"dummy"})
yaml.safe_dump(litellm_proxy_config(r, master_key="sk-1234"), open("/tmp/bf.yaml","w"))
PY
$PYBF -c "from benchflow.providers.litellm_logging import callback_module_source as s;open('/tmp/benchflow_litellm_callback.py','w').write(s())"
PYTHONPATH=/tmp OPENAI_API_KEY=dummy BENCHFLOW_LITELLM_LOG_PATH=/tmp/cb.jsonl \
  ~/.local/share/uv/tools/benchflow/bin/litellm --config /tmp/bf.yaml --port 41890 &
# POST with wire model "benchflow-vllm-qwen35-2b-base" + stream + tools  ->  200

# (c) only the live run fails:  cd eval/_tooling && make eval BENCHMARK=simpleqa
```

## Fix directions (in rough order of payoff)

1. **Bypass the proxy** — have pi talk directly to the engine inside the sandbox
   (`host.docker.internal:30000`) instead of the host proxy. Costs BenchFlow's
   proxy-side usage/cost/trajectory capture, but removes the cross-boundary stream.
   (No clean BenchFlow flag for this today — proxy is "always used for routable
   agents"; would need a small BenchFlow change.)
2. **Confirm the streaming theory** — keep the proxy `stderr.log` (above) and read
   the real exception; if it's a mid-stream client reset, try forcing non-streaming
   in the proxy/agent path.
3. **Use vLLM for the agentic path** — the proxy works with it (`make serve-vllm`,
   set the run-config base URL to `:8000`); note the *base* model still rarely
   tool-calls there, so scores stay low.
4. **Report upstream** to benchflow-ai with this isolation matrix.

## Impact

BenchFlow is the sole runner (`make eval`), so this bug currently blocks real runs
against the **self-hosted** engine. Two ways forward until it's fixed:
- `make eval-direct` — Docker-free debug signal (pi → engine, same tasks + scorer,
  **not** BenchFlow). Good for sanity-checking the model, not a substitute.
- A first-party agent/model (`make eval AGENT=claude MODEL=…`) doesn't go through
  the self-hosted proxy path, so it's unaffected.

Fixing **Fix direction #1** (pi → engine direct inside the sandbox) is the clean
unblock for self-hosted models under BenchFlow.
