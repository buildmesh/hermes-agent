import hashlib
import json
import os
import time
from pathlib import Path

import pytest

from hermes_cli.terminal_presenter import (
    TerminalPresenterError,
    execute_terminal_presenter,
    load_terminal_presenters,
    terminal_presenters_available,
)


def _write_presenter(
    root: Path,
    *,
    handler_source: str,
    schema: dict | None = None,
    max_output_bytes: int = 4096,
) -> None:
    handler = root / "skills/test/present.py"
    input_schema = root / "contract/test.input.schema.json"
    handler.parent.mkdir(parents=True)
    input_schema.parent.mkdir(parents=True)
    handler.write_text(handler_source, encoding="utf-8")
    input_schema.write_text(
        json.dumps(schema or {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "required": ["value"],
            "properties": {"value": {"type": "string"}},
        }),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "telegram.bridge.terminal_presenters.v1",
        "presenters": [{
            "presenter_id": "test-presenter",
            "version": "1.0.0",
            "description": "Render a test result.",
            "handler": "skills/test/present.py",
            "sha256": "sha256:" + hashlib.sha256(handler.read_bytes()).hexdigest(),
            "input_schema": "contract/test.input.schema.json",
            "input_schema_sha256": "sha256:" + hashlib.sha256(input_schema.read_bytes()).hexdigest(),
            "command": ["python3", "{handler}"],
            "timeout_seconds": 2,
            "max_output_bytes": max_output_bytes,
        }],
    }
    (root / "contract/terminal-presenters.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def test_presenter_returns_exact_stdout_and_sanitizes_telegram_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_presenter(
        tmp_path,
        handler_source=(
            "import json, os, sys\n"
            "value=json.load(sys.stdin)['value']\n"
            "sys.stdout.write(value + '|' + str('TELEGRAM_BOT_TOKEN' in os.environ) + '\\n')\n"
        ),
    )
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "must-not-leak")

    result = execute_terminal_presenter(
        tmp_path,
        "test-presenter",
        {"value": "exact  \u2603"},
    )

    assert result["content"] == "exact  \u2603|False\n"
    assert result["sha256"] == hashlib.sha256(result["content"].encode()).hexdigest()


def test_presenter_rejects_invalid_input_before_execution(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    _write_presenter(
        tmp_path,
        handler_source=f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n",
    )

    with pytest.raises(TerminalPresenterError, match="input failed validation") as exc:
        execute_terminal_presenter(tmp_path, "test-presenter", {"wrong": "shape"})

    assert exc.value.code == "PRESENTER_INPUT_INVALID"
    assert not marker.exists()


def test_presenter_rejects_digest_mismatch(tmp_path: Path) -> None:
    _write_presenter(tmp_path, handler_source="print('ok')\n")
    (tmp_path / "skills/test/present.py").write_text("print('changed')\n", encoding="utf-8")

    with pytest.raises(TerminalPresenterError) as exc:
        load_terminal_presenters(tmp_path)

    assert exc.value.code == "PRESENTER_DIGEST_MISMATCH"


def test_presenter_rejects_oversize_output(tmp_path: Path) -> None:
    _write_presenter(
        tmp_path,
        handler_source="import sys\nsys.stdout.write('x' * 257)\n",
        max_output_bytes=256,
    )

    with pytest.raises(TerminalPresenterError) as exc:
        execute_terminal_presenter(tmp_path, "test-presenter", {"value": "unused"})

    assert exc.value.code == "PRESENTER_OUTPUT_TOO_LARGE"


def test_presenter_requires_closed_self_contained_input_schema(tmp_path: Path) -> None:
    _write_presenter(
        tmp_path,
        handler_source="print('unused')\n",
        schema={
            "type": "object",
            "properties": {"nested": {"$dynamicRef": "https://example.invalid/schema"}},
        },
    )

    with pytest.raises(TerminalPresenterError) as exc:
        load_terminal_presenters(tmp_path)

    assert exc.value.code == "PRESENTER_CONFIG_INVALID"


def test_presenter_manifest_is_bounded_to_32_declarations(tmp_path: Path) -> None:
    _write_presenter(tmp_path, handler_source="print('unused')\n")
    manifest_path = tmp_path / "contract/terminal-presenters.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["presenters"] = manifest["presenters"] * 33
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(TerminalPresenterError) as exc:
        load_terminal_presenters(tmp_path)

    assert exc.value.code == "PRESENTER_CONFIG_INVALID"


def test_presenter_availability_requires_active_turn_endpoint(tmp_path: Path) -> None:
    _write_presenter(tmp_path, handler_source="print('ok')\n")
    assert not terminal_presenters_available(tmp_path)
    endpoint = tmp_path / "state/persistent-runtime/presenter-endpoint.json"
    endpoint.parent.mkdir(parents=True)
    endpoint.write_text(json.dumps({"turn_token": "present_turn_test"}), encoding="utf-8")
    assert terminal_presenters_available(tmp_path)


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group containment")
def test_presenter_timeout_kills_descendant_holding_output_pipe(tmp_path: Path) -> None:
    child_pid = tmp_path / "child.pid"
    _write_presenter(
        tmp_path,
        handler_source=(
            "import subprocess\n"
            f"child=subprocess.Popen(['python3','-c','import time; time.sleep(30)'])\n"
            f"open({str(child_pid)!r}, 'w').write(str(child.pid))\n"
        ),
    )
    manifest_path = tmp_path / "contract/terminal-presenters.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["presenters"][0]["timeout_seconds"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(TerminalPresenterError) as exc:
        execute_terminal_presenter(tmp_path, "test-presenter", {"value": "unused"})

    assert exc.value.code == "PRESENTER_TIMEOUT"
    pid = int(child_pid.read_text())
    for _ in range(20):
        stat_path = Path(f"/proc/{pid}/stat")
        if not stat_path.exists():
            break
        fields = stat_path.read_text(encoding="utf-8").split()
        if len(fields) > 2 and fields[2] == "Z":
            break
        time.sleep(0.01)
    else:
        raise AssertionError("presenter descendant survived process-group termination")
