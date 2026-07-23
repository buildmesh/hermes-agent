# Bug: Negotiated Terminal Presenter Tool Is Not Projected into Codex

Date: 2026-07-23

Status: Open

## Summary

A deployed protocol-v3 persistent specialist turn negotiated terminal-presenter finalization and
Hermes created an active presenter endpoint with a turn token. The bridge prompt consequently told
the model that `finalize_telegram_presentation` was available, but the Codex app-server tool search
returned no matching tool.

The model spent most of the turn trying to discover the promised tool, invoked the presenter script
directly as a fallback, and ended with `Done.`. Hermes recorded that acknowledgement as
`candidate_source.kind == "model_final"` instead of promoting the valid presenter artifact. TBA
then corrected `Done.` into a valid but incorrect user-facing response containing only that text.

## Sanitized Live Evidence

The affected request was a routine read-only current-stock lookup after a specialist reset.

- Protocol: `telegram.bridge.persistent_service.v3`
- Runtime: Codex app server
- Model: `gpt-5.3-codex-spark`
- Hermes prompt: stated that negotiated `finalize_telegram_presentation` was available
- Presenter endpoint: existed for the active runtime and contained a turn token
- Presenter declaration: loaded successfully at version `1.0.1`
- First tool search for the exact finalizer name: returned `tools: []`
- Second tool search for terminal presenter finalization: returned unrelated connector namespaces
  and no finalizer
- Domain CLI duration: approximately 0.43 seconds
- Initial model-and-tools duration: 56.0 seconds
- Correction duration: 4.2 seconds
- Approximate total before delivery: 61.7 seconds

The model made exploratory calls for profile listing, skill reads, CLI help, source searches,
presenter manifest inspection, presenter source inspection, endpoint inspection, and a nonexistent
shell command. It then invoked the presenter script directly. The script produced a valid inventory
table, but that output was not captured as a terminal artifact because the registered finalizer tool
was never called.

Durable results:

- original `candidate_source.kind == "model_final"`
- original render candidate was the acknowledgement `Done.`
- `correction_attempt_count == 1`
- delivered `candidate_source.kind == "model_correction"`
- delivered payload text was only `Done.`

No private message content, chat/user identifiers, credentials, raw authorization material, or
provider secrets are retained in this report.

## Expected Behavior

When Hermes enables the negotiated terminal-presenter path for a turn:

1. `finalize_telegram_presentation` is projected into that same Codex app-server turn.
2. Searching for the exact tool name returns the finalizer tool and its schema.
3. The model can call it with the presenter ID and structured input.
4. Hermes captures the exact artifact and records terminal-presenter provenance.
5. The post-tool acknowledgement cannot become the render candidate.

Hermes must not tell the model the finalizer is available unless the transport has successfully
projected it.

## Acceptance Criteria

- An integration test uses the real Codex app-server projection path with an active negotiated
  presenter turn.
- Exact-name tool discovery returns `finalize_telegram_presentation`.
- Invoking the discovered tool reaches the active presenter endpoint and returns its artifact.
- A successful finalizer call followed by a brief acknowledgement records
  `candidate_source.kind == "terminal_presenter"`.
- The acknowledgement is never exposed as the render candidate.
- If projection fails, the worker fails closed or uses the ordinary model-render contract; it does
  not issue a prompt claiming that the unavailable finalizer exists.
- Capability downgrade continues to omit both the tool and finalizer-specific prompt instructions.

## Likely Investigation Area

Inspect the boundary between:

- the persistent worker's negotiated presenter-turn setup;
- service-gated registration of `finalize_telegram_presentation`;
- the Hermes tools MCP server's dynamic tool projection; and
- Codex app-server deferred tool search metadata.

The endpoint and turn token existed, so presenter loading and turn activation had succeeded. The
failure appears later, in registration visibility or projection into Codex.

