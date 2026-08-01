import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from hermes_cli.terminal_mutation import (
    MUTATION_JOURNAL_RELATIVE_PATH,
    TerminalMutationError,
    execute_terminal_mutation,
    load_terminal_mutations,
    recover_terminal_mutation_journal,
    terminal_mutations_available,
)


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _schema(
    path: Path,
    properties: dict,
    required: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "type": "object",
                "additionalProperties": False,
                "required": required,
                "properties": properties,
            }
        ),
        encoding="utf-8",
    )


def install_mutation(root: Path, *, failing: bool = False) -> None:
    scripts = root / "skills/test"
    contract = root / "contract"
    scripts.mkdir(parents=True)
    contract.mkdir(parents=True)
    (root / "config.yaml").write_text(
        "model:\n  provider: openai-codex\n"
        "  openai_runtime: codex_app_server\n",
        encoding="utf-8",
    )
    handler = scripts / "mutate.py"
    handler.write_text(
        (
            "raise SystemExit(7)\n"
            if failing
            else (
                "import json,pathlib,sys\n"
                "v=json.load(sys.stdin)\n"
                "c=pathlib.Path('handler-count.txt')\n"
                "c.write_text(str(int(c.read_text())+1) if c.exists() else '1')\n"
                "value=v['input']['value']\n"
                "if value in {'deny','conflict'}:\n"
                " print(json.dumps({'outcome':'not_applied' if value=='deny' else 'conflict',"
                "'result':{'value':value,'record_id':'none'}},separators=(',',':')))\n"
                " raise SystemExit(0)\n"
                "p=pathlib.Path('backend-operations.json')\n"
                "db=json.loads(p.read_text()) if p.exists() else {}\n"
                "op=v['operation_id']\n"
                "if op not in db:\n"
                " db[op]={'value':value,'record_id':'rec-1'}\n"
                " p.write_text(json.dumps(db,sort_keys=True))\n"
                "print(json.dumps({'outcome':'committed','result':db[op]},separators=(',',':')))\n"
            )
        ),
        encoding="utf-8",
    )
    presenter = scripts / "present.py"
    presenter.write_text(
        "import json,sys\nv=json.load(sys.stdin)\n"
        "print(json.dumps([{'render':{'text':v['value']}}],separators=(',',':')),end='')\n",
        encoding="utf-8",
    )
    input_schema = contract / "mutation.input.json"
    output_schema = contract / "mutation.output.json"
    presenter_schema = contract / "presenter.input.json"
    _schema(
        input_schema,
        {"value": {"type": "string"}},
        ["value"],
    )
    _schema(
        output_schema,
        {
            "outcome": {
                "enum": ["committed", "not_applied", "conflict"]
            },
            "result": {
                "type": "object",
                "additionalProperties": False,
                "required": ["value", "record_id"],
                "properties": {
                    "value": {"type": "string"},
                    "record_id": {"type": "string"},
                },
            },
        },
        ["outcome", "result"],
    )
    _schema(
        presenter_schema,
        {
            "value": {"type": "string"},
            "record_id": {"type": "string"},
        },
        ["value", "record_id"],
    )
    (contract / "terminal-presenters.json").write_text(
        json.dumps(
            {
                "schema_version": "telegram.bridge.terminal_presenters.v1",
                "presenters": [
                    {
                        "presenter_id": "test-presenter",
                        "version": "1.0.0",
                        "description": "Present a mutation.",
                        "handler": "skills/test/present.py",
                        "sha256": _digest(presenter),
                        "input_schema": "contract/presenter.input.json",
                        "input_schema_sha256": _digest(presenter_schema),
                        "command": ["python3", "{handler}"],
                        "timeout_seconds": 2,
                        "max_output_bytes": 4096,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (contract / "terminal-mutations.json").write_text(
        json.dumps(
            {
                "schema_version": "telegram.bridge.terminal_mutations.v1",
                "workflows": [
                    {
                        "workflow_id": "test-create",
                        "version": "1.0.0",
                        "description": "Create one test record.",
                        "mutation": True,
                        "idempotency": {
                            "mode": "operation_id",
                            "guarantee": "transactional",
                        },
                        "handler": {
                            "handler": "skills/test/mutate.py",
                            "sha256": _digest(handler),
                            "command": ["python3", "{handler}"],
                            "input_schema": "contract/mutation.input.json",
                            "input_schema_sha256": _digest(input_schema),
                            "output_schema": "contract/mutation.output.json",
                            "output_schema_sha256": _digest(output_schema),
                            "timeout_seconds": 2,
                            "max_output_bytes": 4096,
                        },
                        "mapper": None,
                        "presenter_id": "test-presenter",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def execute(root: Path, value: str = "note") -> dict:
    return execute_terminal_mutation(
        root,
        specialist_id="test-agent",
        conversation_id="conv-test",
        event_id="evt-test",
        workflow_id="test-create",
        workflow_input={"value": value},
    )


def test_loads_transactional_operation_id_declaration(
    tmp_path: Path,
) -> None:
    install_mutation(tmp_path)
    mutation = load_terminal_mutations(tmp_path)["test-create"]
    assert mutation.presenter_id == "test-presenter"


def test_availability_requires_negotiated_active_turn(
    tmp_path: Path,
) -> None:
    install_mutation(tmp_path)
    assert not terminal_mutations_available(tmp_path)
    endpoint = (
        tmp_path / "state/persistent-runtime/presenter-endpoint.json"
    )
    endpoint.parent.mkdir(parents=True)
    endpoint.write_text(
        json.dumps(
            {
                "turn_token": "present_turn_test",
                "terminal_mutation": True,
            }
        ),
        encoding="utf-8",
    )
    assert terminal_mutations_available(tmp_path)


def test_completed_mutation_replays_without_handler_execution(
    tmp_path: Path,
) -> None:
    install_mutation(tmp_path)
    first = execute(tmp_path)
    second = execute(tmp_path)
    assert first["operation_id"] == second["operation_id"]
    assert first["presenter_input"] == second["presenter_input"]
    assert (tmp_path / "handler-count.txt").read_text() == "1"
    backend = json.loads(
        (tmp_path / "backend-operations.json").read_text()
    )
    assert list(backend) == [first["operation_id"]]


def test_concurrent_same_event_allocates_one_operation(
    tmp_path: Path,
) -> None:
    install_mutation(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _item: execute(tmp_path), range(2)))
    assert results[0]["operation_id"] == results[1]["operation_id"]
    assert (tmp_path / "handler-count.txt").read_text() == "1"


def test_same_event_rejects_changed_mutation_input(tmp_path: Path) -> None:
    install_mutation(tmp_path)
    execute(tmp_path)
    with pytest.raises(
        TerminalMutationError,
        match="already bound",
    ) as exc:
        execute(tmp_path, "changed")
    assert exc.value.sealed is True
    assert (tmp_path / "handler-count.txt").read_text() == "1"


@pytest.mark.parametrize("value", ["deny", "conflict"])
def test_noncommitted_outcome_is_terminal_and_replayable(
    tmp_path: Path,
    value: str,
) -> None:
    install_mutation(tmp_path)
    first = execute(tmp_path, value)
    second = execute(tmp_path, value)
    assert first["outcome"] == (
        "not_applied" if value == "deny" else "conflict"
    )
    assert second["outcome"] == first["outcome"]
    assert (tmp_path / "handler-count.txt").read_text() == "1"
    assert not (tmp_path / "backend-operations.json").exists()


def test_handler_failure_seals_unknown_outcome(tmp_path: Path) -> None:
    install_mutation(tmp_path, failing=True)
    with pytest.raises(TerminalMutationError) as exc:
        execute(tmp_path)
    assert exc.value.sealed is True
    record_path = next(
        (tmp_path / MUTATION_JOURNAL_RELATIVE_PATH).glob("*.json")
    )
    record = json.loads(record_path.read_text())
    assert record["execution_state"] == "outcome_unknown"
    with pytest.raises(TerminalMutationError, match="unknown"):
        execute(tmp_path)


def test_recovery_marks_executing_and_preparing_terminal(
    tmp_path: Path,
) -> None:
    directory = tmp_path / MUTATION_JOURNAL_RELATIVE_PATH
    directory.mkdir(parents=True)
    path = directory / "record.json"
    path.write_text(
        json.dumps(
            {
                "execution_state": "executing",
                "presentation_state": "preparing",
            }
        ),
        encoding="utf-8",
    )
    assert recover_terminal_mutation_journal(tmp_path)
    record = json.loads(path.read_text())
    assert record["execution_state"] == "outcome_unknown"
    assert record["presentation_state"] == "presentation_failed"


def test_corrupt_journal_withholds_runtime(tmp_path: Path) -> None:
    directory = tmp_path / MUTATION_JOURNAL_RELATIVE_PATH
    directory.mkdir(parents=True)
    (directory / "corrupt.json").write_text("{", encoding="utf-8")
    assert recover_terminal_mutation_journal(tmp_path) is False


def test_rejects_nontransactional_idempotency_claim(
    tmp_path: Path,
) -> None:
    install_mutation(tmp_path)
    manifest_path = tmp_path / "contract/terminal-mutations.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["workflows"][0]["idempotency"]["guarantee"] = "best_effort"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(TerminalMutationError, match="transactional"):
        load_terminal_mutations(tmp_path)
