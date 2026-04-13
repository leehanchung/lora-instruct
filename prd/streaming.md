# Streaming plan

Implementation plan for showing Claude Code's live reasoning / tool use
in Discord, similar to the terminal UI in Claude Code or Claude Desktop.

This is a *plan*, not a spec of implemented behavior. Nothing in this
document is in the code yet.

## Goal

When a user `@mentions` the bot, Discord should show a live "status"
message that updates as Claude Code progresses through its run:
thinking, tool calls, tool results, and final response. The user should
get meaningful "I'm working on it" feedback within a second of sending
the message, and should be able to watch progress without guessing
whether the bot is hung.

## Non-goals

- **Character-by-character typing animation.** Every Discord update is
  an HTTP edit — we cannot stream at Claude Code's native token rate.
- **Pixel-perfect parity with Claude Code's terminal UI.** Discord can
  only render markdown, so no box-drawing, no fancy borders.
- **Rendering full tool result bodies inline.** Tool outputs (file
  contents, command output) can be megabytes. We'll show a one-liner
  summary and truncate anything long.
- **Preserving the stream across bot restarts.** If the bot container
  dies mid-run, the status message gets stranded. Acceptable.

## The constraint that shapes everything

Discord rate-limits message edits to ~5 edits per 5 seconds per
channel. A single Claude run can emit hundreds of events in a few
seconds. The renderer has to **coalesce** events into at most one edit
per second, otherwise we'll hit 429s from Discord and the bot will get
throttled.

Every design decision below is downstream of this constraint.

## UX — what it looks like in Discord

### During the run

One "status" message per dispatch, edited in place. Content is a
compact transcript with emoji prefixes for block types. Example
progression (each block is the entire message contents after an edit):

```
💭 Thinking about your request...
```

```
💭 Thought
🔧 Read `src/app.py` ✓
🔧 Read `src/handlers.py`
```

```
💭 Thought
🔧 Read `src/app.py` ✓
🔧 Read `src/handlers.py` ✓
🔧 Grep `handle_channel_message`
✍️  Writing response...
```

### After the run

Two final states:

1. **Status message** is rewritten to a collapsed one-line summary:
   `✅ Done • 5 tools • 8.4s`
2. **Final response** is posted as a *separate* message so it's
   findable in Discord search and doesn't get buried inside a long
   transcript.

### On error

Status message freezes on the last rendered state, then a second
message is posted with the error (same pattern as current
`_dispatch_and_respond` error path).

## Event model

Each Claude Code `stream-json` line maps to one of these internal
event types, which is what the dispatcher yields across the
bot ↔ Modal boundary:

```python
# apps/delulu_sandbox_modal/src/delulu_sandbox_modal/events.py (new)
from typing import Literal, TypedDict

class ToolUseEvent(TypedDict):
    type: Literal["tool_use"]
    tool: str                 # "Read", "Grep", "Bash", ...
    summary: str              # one-liner for display, e.g. "`src/app.py`"

class ToolResultEvent(TypedDict):
    type: Literal["tool_result"]
    tool: str                 # matches the preceding tool_use
    ok: bool                  # False if Claude Code reported an error
    summary: str              # short preview of the result

class ThinkingEvent(TypedDict):
    type: Literal["thinking"]
    text: str                 # hidden behind spoiler in the rendered output

class AssistantTextEvent(TypedDict):
    type: Literal["text"]
    text: str                 # accumulated assistant response so far

class DoneEvent(TypedDict):
    type: Literal["done"]
    final_text: str           # full assistant response
    num_turns: int
    duration_ms: int

class ErrorEvent(TypedDict):
    type: Literal["error"]
    message: str
```

The mapping from Claude Code's `stream-json` schema to these events
lives in one translator function (`_flatten_stream_event`) so when
Claude Code's schema drifts we have one place to fix.

## Architecture changes

### `apps/delulu_sandbox_modal/src/delulu_sandbox_modal/app.py`

`run_claude_code` becomes a Modal **generator function**:

- Change `--output-format text` → `--output-format stream-json`.
- Use `subprocess.Popen` instead of `subprocess.run` so we can read
  stdout line-by-line as Claude Code emits events.
- For each line: `json.loads` it, run it through `_flatten_stream_event`,
  `yield` the resulting event dict (or skip unknown event types).
- On process exit, yield a final `DoneEvent` with the aggregated
  `final_text` and run stats.
- If the subprocess exits with a non-zero code, yield an `ErrorEvent`
  before returning.
- Credential persistence (the volume commit after the run) still
  happens exactly once at the end, same as today.

### `apps/delulu_discord/src/delulu_discord/dispatcher.py`

`SandboxDispatcher.run_task` becomes an **async generator**. Uses
Modal's `.remote_gen.aio(...)` instead of `.remote(...)`:

```python
async def run_task(self, ...) -> AsyncIterator[Event]:
    async for event in self._fn.remote_gen.aio(
        session_id=..., workspace_path=..., prompt=..., resume=...
    ):
        yield event
```

The synchronous-call-in-executor dance we have today goes away —
`remote_gen.aio` is async-native.

### `apps/delulu_discord/src/delulu_discord/handlers.py`

`_dispatch_and_respond` becomes a driver for the event stream:

1. Post an initial `"💭 Thinking about your request..."` message.
2. Iterate `self.dispatcher.run_task(...)` (now an async iterator).
3. Append each event to an in-memory transcript.
4. A background coroutine (`_flush_loop`) edits the status message at
   most once per second with `_render(transcript)`.
5. When the stream ends: cancel the flush loop, write the final
   status message (`✅ Done • ...`), and post the final text as a
   separate message.
6. On error: log, freeze the status message, post an error message.

The renderer `_render(transcript, *, done=False)` is pure and
deterministic — given a list of events, produce a markdown string
fitting in Discord's 2000-char limit. If the transcript would exceed
the limit, truncate older tool calls (keep the most recent N).

### New file: `apps/delulu_discord/src/delulu_discord/streaming.py`

Houses the renderer, the debouncer / flush loop, and the event-to-
markdown formatters. Keeping this out of `handlers.py` so the handler
stays small and testable.

## Implementation order

Ship in three commits so each one is independently useful and
revertable:

### Commit 1 — emit events (no UX change yet)

- Add `apps/delulu_sandbox_modal/src/delulu_sandbox_modal/events.py`
  with the typed event dicts. **Note:** the two apps are independent
  packages — `delulu_discord` cannot import from `delulu_sandbox_modal`.
  For v1 the bot will consume events as untyped dicts. If the duck-typing
  gets painful, promote `events.py` to a third top-level package under
  `apps/` (e.g. `apps/delulu_shared`) that both sides depend on.
- Rewrite `run_claude_code` as a generator that parses stream-json and
  yields events.
- Temporarily keep `SandboxDispatcher` synchronous: have it collect
  the stream, extract the final text from the `DoneEvent`, and return
  a string (same public surface as today). This lets us ship and test
  the sandbox changes without touching the bot yet.
- Verify on the droplet: bot behavior unchanged, `docker logs disco`
  shows no regressions.

### Commit 2 — async generator dispatcher

- Change `SandboxDispatcher.run_task` to `AsyncIterator[Event]` using
  `remote_gen.aio`.
- Update `handlers._dispatch_and_respond` to consume the iterator but
  still post a single final message (just the `final_text` from the
  `DoneEvent`). No live editing yet.
- Verify: same behavior as today, just plumbed differently.

### Commit 3 — live status message

- Add `apps/delulu_discord/src/delulu_discord/streaming.py` with
  `_render` and the flush loop.
- Replace the stub consumer in `_dispatch_and_respond` with the real
  live-rendering driver.
- Rate-limit the flush loop to 1 edit/sec.
- Test with prompts that trigger multiple tool calls.

Split this way, the risky parts (Modal generator plumbing, Discord
rate limits) ship separately so we can bisect if something breaks.

## Decisions to lock in before starting

1. **Show thinking blocks?** Yes, but collapsed behind a Discord
   spoiler: `||🧠 Reasoning: claude was considering...||`. Keeps the
   transcript scannable while still letting curious users peek.
2. **One live message or many?** One. Status message edited in place,
   plus one final message for the response text.
3. **Final text: streamed or posted at end?** Posted at end, as a
   separate message. Doubles as search-friendly archive of responses.
4. **Thread naming still uses the user prompt.** No change.
5. **Status message collapses on done.** `✅ Done • N tools • Ts` —
   scannable in scroll-back without reopening.
6. **Flush cadence.** 1 edit/sec while the run is active. If an edit
   takes longer than the interval, coalesce and only flush once the
   previous edit completes.
7. **Error path.** Freeze status at last state, post error as a new
   message with `⚠️` prefix.

## Risks and unknowns

- **Claude Code's `stream-json` schema is not a stable contract.** It
  may change between versions, especially for thinking blocks and
  tool results. Mitigation: one translator function, defensive parsing
  (unknown events → skip, malformed JSON → log and skip), pin a known-
  good version of Claude Code in the sandbox image.
- **Modal generator latency.** Each `yield` is an RPC round trip (tens
  of ms). For short runs with few events this is imperceptible; for
  runs with hundreds of events it can add a second or two of total
  latency. Probably fine in practice, worth measuring once.
- **Discord edit rate limit backoff.** If we hit a 429, we need to
  stop editing and wait. The flush loop should catch
  `discord.HTTPException` with status 429, sleep for the `Retry-After`
  duration, and resume.
- **Long tool outputs.** A `Read` on a 100KB file produces a huge
  tool_result. We must truncate aggressively (e.g., show "✓ (34 KB)")
  rather than trying to embed any of the content.
- **Concurrent dispatches.** Today we run tasks serially via asyncio;
  with streaming, two concurrent threads would each have their own
  flush loop. Should still work, but means our 1/sec cadence is
  per-thread, not global. Worth double-checking before shipping.

## Testing approach

- **Unit test `_render`** with canned event lists: empty, one tool,
  many tools, thinking blocks, mid-run, done state, error state,
  transcript that overflows 2000 chars (exercises truncation).
- **Unit test `_flatten_stream_event`** against a handful of real
  `stream-json` lines captured from Claude Code. Stash a small JSONL
  fixture file under `tests/fixtures/stream-json/`.
- **Integration test on the droplet** with prompts of varying shapes:
  - Quick Q&A (no tools) — verify status message collapses immediately
  - Multi-tool run ("read these three files and summarize") — verify
    live updates
  - Long-running task — verify no rate limit errors
  - Tool error ("read a file that doesn't exist") — verify error path
- **Visual acceptance**: send the prompts above from a human and
  eyeball the Discord thread. The live message should feel responsive,
  not jittery, and not lag the final text by more than a few seconds.

## Estimated effort

- Commit 1 (event emission in sandbox): ~2 hours
- Commit 2 (async generator plumbing): ~1 hour
- Commit 3 (live renderer + debouncer): ~3-4 hours
- Testing / polishing: ~1-2 hours

Call it a full day of focused work, give or take.

## Pointers for future implementation

- Event schema reference — look at the actual output of
  `claude -p "..." --output-format stream-json` in a test workspace
  before starting commit 1. The `--help` text describes the general
  shape but field names matter.
- Modal generator docs —
  `https://modal.com/docs/guide/generators` is the canonical reference
  for the `remote_gen` / `.aio` interface.
- Discord edit rate limit —
  `discord.py` exposes rate limits via `HTTPException.retry_after`;
  don't roll your own backoff.
