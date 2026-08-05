import copy
from pathlib import Path

import pytest

from hermes_cli.interaction_session import (
    InteractionContext,
    InteractionSessionDeclaration,
    InteractionSessionError,
    resolve_consumer_context,
    terminal_consumer_context,
    terminal_session_transition,
)


def _context() -> InteractionContext:
    return InteractionContext(
        context_type="report",
        version="1.0.0",
        description="Report parameters.",
        schema_path=Path("unused"),
        schema_sha256="sha256:" + "a" * 64,
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["period"],
            "properties": {"period": {"type": "string"}},
        },
        consumers=(("terminal_mutation", "save-report"),),
    )


def _control() -> dict:
    return {
        "profile": "reports",
        "profile_version": "1.0.0",
        "installed_profile_sha256": "sha256:" + "b" * 64,
        "conversation_generation": 4,
        "active": True,
        "snapshot": {
            "session_id": "sctx_test",
            "context_type": "report",
            "context_version": "1.0.0",
            "schema_sha256": "sha256:" + "a" * 64,
            "revision": 2,
            "state": {"period": "2026-Q3"},
            "state_sha256": "unused",
            "expires_at": "2099-01-01T00:00:00Z",
        },
    }


def test_resolves_exact_consumer_and_projects_an_immutable_runtime_value() -> None:
    context = _context()
    declaration = InteractionSessionDeclaration("reports", "1.0.0", "digest", {"report": context})
    assert resolve_consumer_context(declaration, "terminal_mutation", "save-report") is context
    control = _control()
    projected = terminal_consumer_context(control, context)
    control["snapshot"]["state"]["period"] = "changed"
    assert projected["state"] == {"period": "2026-Q3"}
    assert projected["conversation_generation"] == 4


def test_consumer_missing_or_wrong_context_fails_before_execution() -> None:
    context = _context()
    inactive = _control()
    inactive.update(active=False)
    inactive.pop("snapshot")
    with pytest.raises(InteractionSessionError, match="requires an active") as missing:
        terminal_consumer_context(inactive, context)
    assert missing.value.code == "SESSION_CONTEXT_REQUIRED"
    wrong = _control()
    wrong["snapshot"]["context_version"] = "2.0.0"
    with pytest.raises(InteractionSessionError) as mismatch:
        terminal_consumer_context(wrong, context)
    assert mismatch.value.code == "SESSION_CONTEXT_TYPE_MISMATCH"


def test_terminal_directive_is_closed_validated_and_digest_bound() -> None:
    transition = terminal_session_transition(
        _control(), _context(), "evt-1",
        {"operation": "replace", "context_type": "report", "context_version": "1.0.0", "state": {"period": "2026-Q4"}},
    )
    assert transition["prior_revision"] == 2
    assert transition["revision"] == 3
    assert transition["state_sha256"].startswith("sha256:")
    malformed = copy.deepcopy(transition)
    with pytest.raises(InteractionSessionError) as invalid:
        terminal_session_transition(_control(), _context(), "evt-1", malformed)
    assert invalid.value.code == "SESSION_TRANSITION_INVALID"
