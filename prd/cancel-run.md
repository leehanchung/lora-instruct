# Cancel a running dispatch

Implementation plan for letting a user interrupt a Claude Code run
that's in flight from the Discord side.

This is a *plan*, not a spec of implemented behavior. Nothing in this
document is in the code yet.

## Goal

When a user has `@mentioned` the bot and Claude Code is actively
running — tool calls streaming through the status message, no final
response yet — let the user press a single affordance in Discord and
have the bot terminate the Modal sandbox invocation within a second
or so. The status message freezes on whatever it showed last, the
bot posts a short `🛑 Cancelled` note, and no final assistant text
is posted.

## Non-goals

- **Pause / resume.** Different semantics, different plumbing, no
  obvious use today. If you want to "pause" just cancel and re-dispatch.
- **Cancelling runs from outside Discord.** No CLI, no HTTP endpoint.
  If you need to kill a stuck sandbox that the bot can't reach, use
  the Modal dashboard.
- **Undoing side effects.** Cancellation stops the run, but any files
  Claude Code already wrote to the workspace volume stay. Git state
  inside the workspace is whatever Claude Code left it at.
- **Batch cancel** ("cancel every running dispatch"). Per-thread only.
- **Cancelling a run that has already posted its final message.**
  By the time the response is out, there's nothing left to stop.

## The constraint that shapes everything

Cancellation latency matters. If a user clicks Cancel and the bot
takes 20 seconds to acknowledge, they will click again, then again,
then post "wtf", then assume it's broken. The backend propagation
itself is fast — Modal cancels a generator invocation in under a
second, and `subprocess.kill` on the Popen is immediate — but the
bot side has to wire the affordance to an asyncio task it can
actually cancel, which means holding a reference to the right task
from the right place at the right time.

Every design decision below is downstream of "click → ack in under
one second".

## UX — what it looks like in Discord

### The affordance

Three options considered:

1. **Reaction** — post a `🛑` reaction on the status message,
   listen for `on_reaction_add`, cancel on click.
   - **Pros:** simplest, one-tap, works on mobile, no new UI primitives.
   - **Cons:** reactions require the `reactions` gateway intent
     (non-privileged but must be enabled in both the dev portal and
     `discord.Intents`), and any user in the channel can tap the
     reaction unless we scope it explicitly in the handler.
2. **Button** (a `discord.ui.View` attached to the status message).
   - **Pros:** explicit "Cancel" label; cleanest Discord-native UX;
     per-user scoping is natural via `interaction.user.id`.
   - **Cons:** more code; views go stale after bot restart (the
     button becomes a no-op), which is fine for a personal bot.
3. **Reply keyword** — treat any thread message whose content is
   `!cancel` (or similar) as a cancel signal for the in-flight
   dispatch in that thread.
   - **Pros:** zero UI plumbing, zero new intents.
   - **Cons:** ugly; breaks flow of normal thread replies; still
     needs the same tracking dict everything else would.

**Decision (to lock in before starting):** pick one. Leaning button
for the polish — the per-user scoping falls out cleanly from the
interaction object, and a labelled cancel button is friendlier than
a lone 🛑 emoji that users have to guess the meaning of. Reactions
are a reasonable fallback if we don't want to deal with `View`
lifecycle concerns.

### During the run

Status message posts as it does today, with an extra UI element
attached (button / reaction / nothing, depending on choice above).
The rest of the live-rendering pipeline from `prd/streaming.md` is
unchanged — tool calls still stream in at 1 edit/sec, thinking
blocks still collapse to a spoiler.

### On cancel

1. Status message is edited one last time to append a
   `🛑 Cancelled by @han • 3 tools so far • 4.2s` footer *in place
   of* the `✅ Done` footer it would have gotten on a clean run.
   Everything above stays visible, so the user can still see what
   Claude was doing when they hit the button.
2. The button (if present) is disabled on the view so a double-click
   doesn't fire a second cancel.
3. A separate `🛑 Cancelled` message is posted below. No partial
   final text is posted — assembling a partial answer from an
   interrupted run is more likely to confuse than to help.
4. The tracking dict entry for that thread is removed.

### On natural completion (no cancel)

Same as today's plan from `prd/streaming.md`: status footer becomes
`✅ Done • N tools • Ts`, final text posts as a separate message,
button (if present) is removed or disabled by hitting `view.stop()`.

### On error (sandbox raises)

Same as today: status freezes, separate `⚠️ <error>` message posts,
button disabled, tracking dict entry removed.

## Architecture

### Where cancellation lives on the client side

The async generator in `apps/delulu_discord/src/delulu_discord/dispatcher.py`
is the right place to cancel. The handler's
`async for event in self.dispatcher.run_task(...)` loop becomes a
cancellable task — we wrap the existing `_dispatch_and_respond` body
(or its iterator portion) in `asyncio.create_task(...)` so we have a
handle to cancel, and store that handle keyed by `thread.id`.

```python
# apps/delulu_discord/src/delulu_discord/handlers.py

class MessageHandler:
    def __init__(self, ...):
        ...
        self._in_flight: dict[int, asyncio.Task] = {}

    async def _dispatch_and_respond(self, thread, session, ...):
        task = asyncio.create_task(self._run_dispatch(thread, session, ...))
        self._in_flight[thread.id] = task
        try:
            await task
        except asyncio.CancelledError:
            await self._handle_cancelled(thread, session)
        finally:
            self._in_flight.pop(thread.id, None)
```

The UI affordance callback looks up the task by `thread.id` (not by
status message id — threads are the stable unit) and calls
`task.cancel()`. That's it on the bot side.

### Where cancellation lives on the sandbox side

`run_claude_code` is a generator function. When the bot cancels the
async iterator, Modal propagates the cancellation to the sandbox
container, which raises a cancellation exception inside the running
generator. The existing `try: ... finally:` block in
`run_claude_code`:

```python
try:
    for raw_line in proc.stdout:
        ...
        yield event
    returncode = proc.wait()
    stderr_text = proc.stderr.read() if proc.stderr is not None else ""
finally:
    killer.cancel()
    if proc.poll() is None:
        proc.kill()
        proc.wait()
```

already handles the process kill and watchdog teardown. Good — no
new teardown code needed.

The interesting question is `volume.commit()`, which today runs
*after* the try/finally:

```python
volume.commit()

if returncode != 0:
    yield ErrorEvent(...)
    return

yield DoneEvent(...)
```

On a clean run this persists rotated credentials and any files
Claude Code wrote. On a cancelled run, the generator exits via
exception before reaching `volume.commit()` — so the half-run
state is **not** committed and the next dispatch sees the pre-run
volume state. That's probably right: cancellation means "pretend
this didn't happen" semantics, and the one loss — a refreshed OAuth
token sitting in the sandbox's ephemeral disk — is cheap because
the next dispatch triggers another refresh.

**Decision (to lock in):** cancelled runs do NOT commit the volume.
Document this in a comment so the next reader doesn't "fix" it.

### Authorization — who can cancel

Options:
- **Self-only.** The user who dispatched the run is the only one
  who can cancel. Requires `user_id` on the `Session` dataclass
  (not tracked today — see "Prerequisites").
- **Anyone in the thread.** Simpler. Fine for a small private bot.
  Not safe if the bot is ever added to a public server.
- **Mods + self.** Some ACL on top of self-only. Overkill today.

**Decision:** self-only. Easier to relax later than to tighten.

### Button callback logic

```python
class CancelView(discord.ui.View):
    def __init__(self, *, task: asyncio.Task, user_id: int):
        super().__init__(timeout=None)
        self._task = task
        self._user_id = user_id

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.danger, emoji="🛑")
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self._user_id:
            await interaction.response.send_message(
                "Only the user who started this run can cancel it.",
                ephemeral=True,
            )
            return
        button.disabled = True
        await interaction.response.edit_message(view=self)
        self._task.cancel()
```

`timeout=None` matters — we don't want the view to auto-expire and
strand the run.

## Prerequisites

1. **`Session.user_id`.** `session_manager.Session` today tracks
   `session_id`, `thread_id`, `workspace_path`, and timestamps. It
   does **not** track the Discord user who created the session. Add
   `user_id: int | None = None` to the dataclass and populate it
   from `message.author.id` in `MessageHandler.handle_channel_message`
   (and carry it into `handle_thread_reply` by looking up the
   existing session). Self-only cancel needs this.
2. **Reactions intent** (only if we pick the reaction UX). `discord.Intents`
   already enables `message_content`; add `reactions = True` and flip
   the toggle in the Discord developer portal too. Not required for
   the button UX.

Neither prerequisite is blocking.

## Implementation order

Ship in two commits so each is independently useful and revertable:

### Commit 1 — plumbing

- Add `_in_flight: dict[int, asyncio.Task]` to `MessageHandler.__init__`.
- Split `_dispatch_and_respond` into a wrapping coroutine that
  stores/pops the task in `_in_flight`, and a body coroutine that
  actually runs the dispatch loop.
- Handle `asyncio.CancelledError` in the wrapper: freeze the status
  message with the cancel footer and post the `🛑 Cancelled` message.
- Make `run_claude_code` in the sandbox module-docstring its
  volume-commit-on-cancel semantics (no code change — just make the
  expected behavior explicit in a comment for the next reader).
- Verify: manually `asyncio.get_running_loop().get_task(...)` in a
  REPL / log statement to confirm the tracking dict is populated and
  cleaned up. Nothing user-visible yet.

### Commit 2 — UI affordance

- Implement whichever of {reaction, button, keyword} we chose.
- Attach the affordance to the status message in `_dispatch_and_respond`.
- Wire the callback to `task.cancel()` keyed by `thread.id`.
- Disable the affordance on success / cancel / error paths.
- End-to-end test on the droplet: dispatch a slow prompt
  (e.g. `@bot read every .py file under apps/ and summarize each`),
  click cancel, verify:
  - Status message freezes with `🛑 Cancelled` footer
  - No final message appears
  - `docker logs disco` shows `dispatch.cancelled session_id=...`
  - Modal dashboard shows the invocation cancelled (not timed out)

Split this way, the behavioral change (task tracking, cancel path)
lands separately from the UI primitive. If the view/button misbehaves
we can revert Commit 2 alone and still have the tracking dict sitting
there ready for a different UI.

## Decisions to lock in before starting

1. **UX affordance:** button / reaction / reply keyword. Recommend
   button for the labelled affordance and native per-user scoping.
2. **Authorization:** self-only / anyone / mods+self. Recommend
   self-only.
3. **Volume commit on cancel:** no (pretend the run didn't happen).
4. **Post-cancel status message:** freeze the transcript + append
   `🛑 Cancelled by @user • N tools so far • Ts` footer in place of
   the `✅ Done` footer.
5. **Post-cancel separate message:** a bare `🛑 Cancelled.` (no
   partial final text).
6. **Double-click protection:** disable the button/reaction on the
   first click; second click becomes a no-op.
7. **Natural-completion cleanup:** `view.stop()` after `finalize_done`
   so the button stops responding once the run is actually done
   (otherwise a user clicking Cancel on a message whose run already
   finished would get a confusing "not your run" error).

## Risks and unknowns

- **Modal cancellation latency.** Documented as "under a second" but
  we haven't measured on our deployment. Worth capturing a few samples
  on the droplet after Commit 1 so we know the real floor before
  promising sub-second ack in the UX.
- **Cancel clicked during `finalize_done`.** Race: the user hits
  Cancel while the main coroutine is already past the `async for`
  loop and running `finalize_done`. `task.cancel()` on a task that
  has nothing left to cancel is a no-op; the button callback should
  detect this case and respond with an ephemeral "already done" rather
  than freezing the status.
- **View lifecycle across bot restarts.** If the bot container
  restarts mid-run, the view object is gone but the status message
  still shows the button. Clicks on stale buttons fail with
  `discord.NotFound` or similar. Acceptable for a personal bot;
  mitigate with a 5-minute view timeout if it becomes annoying.
- **Reactions intent** (if we go that route). Need to flip it in the
  Discord developer portal AND in `discord.Intents(...)` in
  `main.py`. Missing the portal toggle is an easy-to-forget failure
  mode — flag it in the PR description so the reviewer catches it.
- **Orphaned Modal invocations.** If `task.cancel()` fires but the
  cancellation never propagates to Modal (network blip), we'd have
  a sandbox still running out there, billing CPU, until the 300s
  timeout. Cheap enough to ignore for a personal bot; note it.

## Testing approach

- **Unit tests** for `CancelView.cancel` (or the reaction handler):
  - Wrong user → ephemeral error, `task.cancel` NOT called
  - Right user → button disabled, `task.cancel` called, message edited
- **Unit tests** for the tracking dict:
  - Task added on dispatch start
  - Task removed on clean completion
  - Task removed on error
  - Task removed on cancel
- **Integration test on the droplet** with a prompt that takes 10+
  seconds (multi-tool, read several files). Click cancel, verify all
  four acceptance signals in the Commit 2 section above.

## Estimated effort

- `Session.user_id` prerequisite: 15 min
- Tracking dict + `_dispatch_and_respond` split: 45 min
- Cancel handler + freeze footer: 30 min
- `CancelView` + attachment: 1 hour
- Tests (unit + on-droplet): 1 hour
- Docs + PR: 30 min

Call it a half-day of focused work.

## Pointers for future implementation

- discord.py Views docs — `https://discordpy.readthedocs.io/en/stable/interactions/api.html#view`
  is the canonical reference for `discord.ui.View`, button callbacks,
  and per-interaction authorization.
- Modal cancellation docs — `https://modal.com/docs/guide/cancellation`
  covers how client-side cancellation propagates to running sandboxes.
  The `.aio` generator path should behave identically to the sync
  path for cancellation; verify once on the droplet.
