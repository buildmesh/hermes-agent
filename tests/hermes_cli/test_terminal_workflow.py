import hashlib
import json
from pathlib import Path

import pytest

from hermes_cli.terminal_workflow import (
    TerminalWorkflowError,
    execute_terminal_workflow,
    load_terminal_workflows,
)


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_schema(path: Path, properties: dict, required: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }), encoding="utf-8")


def install_workflow(root: Path, *, mapper: bool = False) -> None:
    scripts = root / "skills/test"
    contract = root / "contract"
    scripts.mkdir(parents=True)
    contract.mkdir(parents=True)
    (root / "config.yaml").write_text(
        "model:\n  provider: openai-codex\n"
        "  openai_runtime: codex_app_server\n",
        encoding="utf-8",
    )
    producer = scripts / "producer.py"
    producer.write_text(
        "import json,sys,pathlib,time\n"
        "p=pathlib.Path('producer-count.txt')\n"
        "p.write_text(str(int(p.read_text())+1) if p.exists() else '1')\n"
        "time.sleep(0.05)\n"
        "v=json.load(sys.stdin)\n"
        "print(json.dumps({'outcome':v['outcome'],"
        "'result':{'payloads':v['payloads']} if v['outcome']=='terminal_ready' else None,"
        "'model_result':{'reason':'clarify'} if v['outcome']=='continue' else None},"
        "separators=(',',':')))\n",
        encoding="utf-8",
    )
    mapper_path = scripts / "mapper.py"
    mapper_path.write_text(
        "import json,sys\nv=json.load(sys.stdin)\n"
        "print(json.dumps({'payloads':v['payloads']},separators=(',',':')))\n",
        encoding="utf-8",
    )
    input_schema = contract / "workflow.input.json"
    producer_output = contract / "workflow.output.json"
    mapper_input = contract / "mapper.input.json"
    mapper_output = contract / "mapper.output.json"
    presenter_input = contract / "presenter.input.json"
    _write_schema(input_schema, {
        "outcome": {"enum": ["terminal_ready", "continue"]},
        "payloads": {"type": "array"},
    }, ["outcome", "payloads"])
    _write_schema(producer_output, {
        "outcome": {"enum": ["terminal_ready", "continue"]},
        "result": {"type": ["object", "null"]},
        "model_result": {"type": ["object", "null"]},
    }, ["outcome", "result", "model_result"])
    for path in (mapper_input, mapper_output, presenter_input):
        _write_schema(path, {"payloads": {"type": "array"}}, ["payloads"])
    presenter = scripts / "presenter.py"
    presenter.write_text(
        "import json,sys\nv=json.load(sys.stdin)\n"
        "print(json.dumps(v['payloads'],separators=(',',':')),end='')\n",
        encoding="utf-8",
    )
    (contract / "terminal-presenters.json").write_text(json.dumps({
        "schema_version": "telegram.bridge.terminal_presenters.v1",
        "presenters": [{
            "presenter_id": "test-presenter",
            "version": "1.0.0",
            "description": "Render test results.",
            "handler": "skills/test/presenter.py",
            "sha256": _digest(presenter),
            "input_schema": "contract/presenter.input.json",
            "input_schema_sha256": _digest(presenter_input),
            "command": ["python3", "{handler}"],
            "timeout_seconds": 2,
            "max_output_bytes": 4096,
        }],
    }), encoding="utf-8")

    def program(handler: Path, in_schema: Path, out_schema: Path) -> dict:
        return {
            "handler": str(handler.relative_to(root)),
            "sha256": _digest(handler),
            "command": ["python3", "{handler}"],
            "input_schema": str(in_schema.relative_to(root)),
            "input_schema_sha256": _digest(in_schema),
            "output_schema": str(out_schema.relative_to(root)),
            "output_schema_sha256": _digest(out_schema),
            "timeout_seconds": 2,
            "max_output_bytes": 4096,
        }

    (contract / "terminal-workflows.json").write_text(json.dumps({
        "schema_version": "telegram.bridge.terminal_workflows.v1",
        "workflows": [{
            "workflow_id": "test-workflow",
            "version": "1.0.0",
            "description": "Produce and present a test result.",
            "read_only": True,
            "producer": program(producer, input_schema, producer_output),
            "mapper": (
                program(mapper_path, mapper_input, mapper_output)
                if mapper
                else None
            ),
            "presenter_id": "test-presenter",
        }],
    }), encoding="utf-8")


def test_loads_closed_digest_bound_read_only_workflow(tmp_path: Path) -> None:
    install_workflow(tmp_path)
    workflow = load_terminal_workflows(tmp_path)["test-workflow"]
    assert workflow.presenter_id == "test-presenter"
    assert workflow.read_only is True


@pytest.mark.parametrize("mapper", [False, True])
def test_executes_terminal_ready_with_optional_profile_mapper(
    tmp_path: Path,
    mapper: bool,
) -> None:
    install_workflow(tmp_path, mapper=mapper)
    result = execute_terminal_workflow(
        tmp_path,
        "test-workflow",
        {"outcome": "terminal_ready", "payloads": [{"render": {"text": "ok"}}]},
    )
    assert result["outcome"] == "terminal_ready"
    assert result["presenter_id"] == "test-presenter"
    assert result["presenter_input"]["payloads"][0]["render"]["text"] == "ok"


def test_continue_returns_to_model_without_presenter_input(tmp_path: Path) -> None:
    install_workflow(tmp_path)
    result = execute_terminal_workflow(
        tmp_path,
        "test-workflow",
        {"outcome": "continue", "payloads": []},
    )
    assert result["outcome"] == "continue"
    assert result["model_result"] == {"reason": "clarify"}
    assert "presenter_input" not in result


def test_completed_producer_checkpoint_prevents_reexecution(tmp_path: Path) -> None:
    install_workflow(tmp_path, mapper=True)
    first = execute_terminal_workflow(
        tmp_path,
        "test-workflow",
        {"outcome": "terminal_ready", "payloads": []},
    )
    second = execute_terminal_workflow(
        tmp_path,
        "test-workflow",
        {"outcome": "terminal_ready", "payloads": []},
        checkpoint=first["checkpoint"],
    )
    assert second["presenter_input"] == first["presenter_input"]
    assert (tmp_path / "producer-count.txt").read_text() == "1"


def test_completed_producer_checkpoint_rejects_changed_retry_input(
    tmp_path: Path,
) -> None:
    install_workflow(tmp_path)
    first = execute_terminal_workflow(
        tmp_path,
        "test-workflow",
        {"outcome": "terminal_ready", "payloads": []},
    )
    with pytest.raises(TerminalWorkflowError, match="retry input differs"):
        execute_terminal_workflow(
            tmp_path,
            "test-workflow",
            {"outcome": "terminal_ready", "payloads": [{"changed": True}]},
            checkpoint=first["checkpoint"],
        )


def test_rejects_mutating_v1_declaration(tmp_path: Path) -> None:
    install_workflow(tmp_path)
    path = tmp_path / "contract/terminal-workflows.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["workflows"][0]["read_only"] = False
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(TerminalWorkflowError, match="only read_only"):
        load_terminal_workflows(tmp_path)


def test_rejects_changed_producer_bytes(tmp_path: Path) -> None:
    install_workflow(tmp_path)
    (tmp_path / "skills/test/producer.py").write_text("raise SystemExit(1)\n")
    with pytest.raises(TerminalWorkflowError, match="digest mismatch"):
        load_terminal_workflows(tmp_path)
