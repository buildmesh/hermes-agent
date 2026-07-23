# Bug: Projected MCP Finalizer Is Falsely Rejected as Not Final

Date: 2026-07-23

Status: Open

## Summary

A protocol-v3 persistent specialist turn successfully discovered and invoked the projected
`mcp__hermes_tools.finalize_telegram_presentation` tool. The presenter returned a valid render
artifact, and no tool invocation followed it. The model then emitted the brief acknowledgement
required by the bridge prompt.

Hermes nevertheless failed the turn with:

```text
TERMINAL_PRESENTER_NOT_FINAL
terminal presenter was not the final tool invocation
```

The correct presenter artifact was discarded and the user received an error response.

## Sanitized Live Evidence

The affected turn was a routine read-only current-stock lookup after a fresh specialist boundary.

- Protocol: `telegram.bridge.persistent_service.v3`
- Runtime: Codex app server
- Model: `gpt-5.3-codex-spark`
- Projected namespace: `mcp__hermes_tools`
- Projected function name: `finalize_telegram_presentation`
- Presenter ID: `preppingdb-telegram`
- Presenter duration: approximately 0.35 seconds
- Presenter result: successful, schema-valid grouped inventory render artifact
- Tool invocations after the finalizer: none
- Model output after the tool result: one brief acknowledgement
- Hermes result: `TERMINAL_PRESENTER_NOT_FINAL`
- Initial model-and-tools duration: 38.8 seconds
- Total worker duration: 40.4 seconds

The Codex transcript records the finalizer as a namespaced MCP `function_call`, followed by an
`mcp_tool_call_end` result and then the assistant acknowledgement. It is the final tool invocation
in the turn.

No private message content, chat/user identifiers, credentials, raw authorization material, or
provider secrets are retained in this report.

## Expected Behavior

Hermes should recognize a successful projected MCP finalizer call as the final tool invocation when:

1. the function name is `finalize_telegram_presentation`;
2. its namespace is the Hermes tools MCP namespace;
3. it is not batched with another tool call;
4. no later tool invocation occurs; and
5. only the permitted assistant acknowledgement follows.

The captured artifact should become the render candidate with
`candidate_source.kind == "terminal_presenter"`.

## Acceptance Criteria

- A Codex app-server integration test records the finalizer in the same namespaced MCP transcript
  shape used by a real deferred-tool invocation.
- A successful finalizer call followed only by an assistant acknowledgement passes finality
  validation.
- The exact captured artifact becomes the render candidate.
- Candidate provenance is `terminal_presenter` with the correct presenter ID and invocation ID.
- A later domain or presenter tool call still fails finality validation.
- A finalizer batched with another tool call still fails finality validation.
- A failed finalizer followed by a successful unbatched retry remains supported.

## Likely Investigation Area

Inspect the conversion from Codex app-server response items into the `messages` collection consumed
by `_terminal_presenter_is_final_and_unbatched`. The deployed transcript represents the discovered
tool as a namespaced MCP function call and its completion as `mcp_tool_call_end`; the existing
finality logic may expect only the older assistant `tool_calls` plus `role == "tool"` shape.

