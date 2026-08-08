import copy
import types
from pathlib import Path

import pytest

from hermes_cli.interaction_session import (
    InteractionContext,
    InteractionSessionDeclaration,
    InteractionSessionError,
    resolve_consumer_context,
    resolve_producer_context,
    terminal_consumer_context,
    terminal_session_transition,
)
from hermes_cli.persistent_specialist_worker import (
    PROTOCOL_V3,
    PersistentSpecialistWorker,
    _validate_v3_response,
)
from hermes_cli.terminal_workflow import TerminalWorkflowError


def _context(
    *,
    producers: tuple[tuple[str, str], ...] = (),
    consumers: tuple[tuple[str, str], ...] = (("terminal_mutation", "save-report"),),
) -> InteractionContext:
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
        consumers=consumers,
        producers=producers,
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


def test_resolves_producer_independently_from_consumer() -> None:
    context = _context(
        producers=(("terminal_workflow", "start-report"),),
        consumers=(),
    )
    declaration = InteractionSessionDeclaration("reports", "1.0.0", "digest", {"report": context})
    assert resolve_producer_context(declaration, "terminal_workflow", "start-report") is context
    assert resolve_consumer_context(declaration, "terminal_workflow", "start-report") is None


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


def test_producer_only_accepts_inactive_control_and_both_role_requires_context() -> None:
    producer = _context(
        producers=(("terminal_workflow", "start-report"),),
        consumers=(),
    )
    worker = object.__new__(PersistentSpecialistWorker)
    worker._interaction_session_declaration = InteractionSessionDeclaration(
        "reports", "1.0.0", "digest", {"report": producer}
    )
    inactive = _control()
    inactive["active"] = False
    inactive.pop("snapshot")
    worker._active_interaction_session_turn = {
        "event_id": "evt-1",
        "control": inactive,
        "transition": None,
    }
    session_context, consumer_context, producer_context = worker._terminal_session_invocation(
        "terminal_workflow", "start-report"
    )
    assert session_context is None
    assert consumer_context is None
    assert producer_context is producer

    both = _context(
        producers=(("terminal_workflow", "start-report"),),
        consumers=(("terminal_workflow", "start-report"),),
    )
    worker._interaction_session_declaration = InteractionSessionDeclaration(
        "reports", "1.0.0", "digest", {"report": both}
    )
    with pytest.raises(InteractionSessionError) as required:
        worker._terminal_session_invocation("terminal_workflow", "start-report")
    assert required.value.code == "SESSION_CONTEXT_REQUIRED"


def test_prepared_transition_rejects_producer_before_execution() -> None:
    producer = _context(
        producers=(("terminal_workflow", "start-report"),),
        consumers=(),
    )
    worker = object.__new__(PersistentSpecialistWorker)
    worker._interaction_session_declaration = InteractionSessionDeclaration(
        "reports", "1.0.0", "digest", {"report": producer}
    )
    worker._active_interaction_session_turn = {
        "event_id": "evt-1",
        "control": _control(),
        "transition": {"operation": "renew"},
    }
    with pytest.raises(InteractionSessionError) as prepared:
        worker._terminal_session_invocation("terminal_workflow", "start-report")
    assert prepared.value.code == "SESSION_TRANSITION_ALREADY_PREPARED"


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


def test_committed_failure_fields_are_closed_and_v3_valid() -> None:
    worker = object.__new__(PersistentSpecialistWorker)
    worker._active_presenter_turn = {
        "committed_mutation_evidence": {
            "outcome": "committed",
            "operation_id": "tmut_" + "1" * 64,
            "workflow_id": "save-report",
            "workflow_version": "1.0.0",
        }
    }
    worker._active_interaction_session_turn = {
        "event_id": "evt-1",
        "control": _control(),
    }
    fields = worker._committed_mutation_failure_fields()
    response = {
        "protocol_version": PROTOCOL_V3,
        "request_id": "req_1",
        "operation": "turn",
        "status": "failed",
        "execution_state": "completed",
        "event_id": "evt-1",
        "runtime_instance_id": "runtime-1",
        "error": {"code": "TERMINAL_PRESENTER_NOT_FINAL", "message": "failed", "retryable": False},
        **fields,
    }
    _validate_v3_response(response)
    assert fields["interaction_session_transition"]["operation"] == "quarantine"
    assert fields["interaction_session_transition"]["prior_revision"] == 2


def test_inactive_committed_failure_keeps_evidence_without_a_fake_transition() -> None:
    worker = object.__new__(PersistentSpecialistWorker)
    worker._active_presenter_turn = {
        "committed_mutation_evidence": {
            "outcome": "committed",
            "operation_id": "tmut_" + "1" * 64,
            "workflow_id": "start-report",
            "workflow_version": "1.0.0",
        }
    }
    control = _control()
    control["active"] = False
    control.pop("snapshot")
    worker._active_interaction_session_turn = {"event_id": "evt-1", "control": control}
    fields = worker._committed_mutation_failure_fields()
    assert "interaction_session_transition" not in fields
    _validate_v3_response({
        "protocol_version": PROTOCOL_V3,
        "request_id": "req_1",
        "operation": "turn",
        "status": "failed",
        "execution_state": "completed",
        "event_id": "evt-1",
        "runtime_instance_id": "runtime-1",
        "error": {"code": "SESSION_TRANSITION_INVALID", "message": "failed", "retryable": False},
        **fields,
    })


@pytest.mark.asyncio
async def test_session_workflow_failure_suppresses_renewal_until_valid_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hermes_cli.persistent_specialist_worker as module

    worker = object.__new__(PersistentSpecialistWorker)
    worker.profile_root = tmp_path
    worker._terminal_workflows = {"report": object()}
    worker._terminal_presenters = {"report-presenter": object()}
    worker._active_presenter_turn = {
        "artifact": None,
        "terminal_workflow": True,
        "turn_token": "present_turn_test",
    }
    worker._active_interaction_session_turn = {
        "event_id": "evt-1",
        "control": _control(),
        "transition": None,
    }
    worker._terminal_session_invocation = types.MethodType(
        lambda self, kind, workflow_id: (
            {"state": {"period": "Q3"}},
            _context(),
            _context(producers=(("terminal_workflow", "report"),), consumers=()),
        ),
        worker,
    )
    worker._terminal_directive_resolver = types.MethodType(
        lambda self, context: lambda directive: None,
        worker,
    )
    request = {
        "protocol_version": "hermes.terminal_presenter.v1",
        "operation": "workflow",
        "turn_token": "present_turn_test",
        "workflow_id": "report",
        "input": {},
    }

    monkeypatch.setattr(
        module,
        "execute_terminal_workflow",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            TerminalWorkflowError("WORKFLOW_FAILED", "mapper failed")
        ),
    )
    failed = await worker._handle_presenter_request(request, workflow_locked=True)
    assert failed["status"] == "failed"
    assert failed["error"]["code"] == "WORKFLOW_FAILED", failed
    assert worker._active_interaction_session_turn["suppress_session_renewal"] is True

    monkeypatch.setattr(
        module,
        "execute_terminal_workflow",
        lambda *args, **kwargs: {
            "outcome": "terminal_ready",
            "presenter_id": "report-presenter",
            "presenter_input": {},
            "workflow_id": "report",
            "workflow_version": "1.0.0",
            "producer_sha256": "a" * 64,
            "mapper_sha256": None,
            "checkpoint": {"workflow_id": "report"},
            "session_transition": None,
        },
    )
    monkeypatch.setattr(
        module,
        "execute_terminal_presenter",
        lambda *args, **kwargs: {
            "content": "ok",
            "sha256": "b" * 64,
            "presenter_id": "report-presenter",
            "presenter_version": "1.0.0",
            "presenter_sha256": "c" * 64,
        },
    )
    completed = await worker._handle_presenter_request(request, workflow_locked=True)
    assert completed["status"] == "completed"
    assert "suppress_session_renewal" not in worker._active_interaction_session_turn


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["producer", "consumer", "both", "undeclared"])
@pytest.mark.parametrize("outcome,renewed", [("committed", True), ("not_applied", False), ("conflict", False)])
async def test_mutation_implicit_renewal_depends_on_outcome_not_declared_role(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    outcome: str,
    renewed: bool,
) -> None:
    import hermes_cli.persistent_specialist_worker as module

    worker = object.__new__(PersistentSpecialistWorker)
    worker.profile_root = tmp_path
    worker.agent_id = "reports"
    worker.last_error_code = None
    worker._terminal_mutations = {"save-report": object()}
    worker._terminal_presenters = {"report-presenter": object()}
    worker._active_presenter_turn = {
        "artifact": None,
        "terminal_mutation": True,
        "turn_token": "present_turn_test",
        "conversation_id": "conv-test",
        "event_id": "evt-1",
    }
    worker._active_interaction_session_turn = {
        "event_id": "evt-1",
        "control": _control(),
        "transition": None,
    }
    context = _context(
        producers=(("terminal_mutation", "save-report"),) if role in {"producer", "both"} else (),
        consumers=(("terminal_mutation", "save-report"),) if role in {"consumer", "both"} else (),
    )
    worker._terminal_session_invocation = types.MethodType(
        lambda self, kind, workflow_id: (
            ({"state": {"period": "Q3"}} if role in {"consumer", "both"} else None),
            (context if role in {"consumer", "both"} else None),
            (context if role in {"producer", "both"} else None),
        ),
        worker,
    )
    worker._terminal_directive_resolver = types.MethodType(
        lambda self, context: lambda directive: None,
        worker,
    )
    monkeypatch.setattr(module, "execute_terminal_mutation", lambda *args, **kwargs: {
        "outcome": outcome,
        "operation_id": "tmut_" + "1" * 64,
        "workflow_id": "save-report",
        "workflow_version": "1.0.0",
        "presenter_id": "report-presenter",
        "presenter_input": {},
        "handler_sha256": "a" * 64,
        "mapper_sha256": "b" * 64,
        "journal_path": tmp_path / "journal.json",
        "session_transition": None,
        "session_transition_error": None,
    })
    monkeypatch.setattr(module, "execute_terminal_presenter", lambda *args, **kwargs: {
        "content": "ok",
        "sha256": "c" * 64,
        "presenter_id": "report-presenter",
        "presenter_version": "1.0.0",
        "presenter_sha256": "d" * 64,
    })
    monkeypatch.setattr(module, "record_terminal_mutation_presentation", lambda *args, **kwargs: None)
    response = await worker._handle_presenter_request({
        "protocol_version": "hermes.terminal_presenter.v1",
        "operation": "mutation",
        "turn_token": "present_turn_test",
        "workflow_id": "save-report",
        "input": {},
    }, workflow_locked=True)
    assert response["status"] == "completed"
    assert (
        "suppress_session_renewal" not in worker._active_interaction_session_turn
    ) is renewed
