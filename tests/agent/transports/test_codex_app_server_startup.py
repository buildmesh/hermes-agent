"""Exercise initialization deadlines over real subprocess pipes."""

import subprocess
import sys

import pytest

from agent.transports.codex_app_server import CodexAppServerClient


@pytest.fixture
def delayed_client(tmp_path, monkeypatch):
    # Replace only process selection: the transport still uses real OS pipes,
    # reader threads, JSON messages, and deadline queues.
    server = """
import json, sys, time
for line in sys.stdin:
    message = json.loads(line)
    if message.get("method") == "initialize":
        time.sleep(11)
        print(json.dumps({"id": message["id"], "result": {"userAgent": "slow-test-server"}}), flush=True)
    elif message.get("method") == "initialized":
        print(json.dumps({"method": "test/ready", "params": {}}), flush=True)
"""
    popen = subprocess.Popen

    def spawn(_command, **kwargs):
        return popen([sys.executable, "-u", "-c", server], **kwargs)

    monkeypatch.setattr(subprocess, "Popen", spawn)
    client = CodexAppServerClient(codex_home=str(tmp_path / "codex"))
    try:
        yield client
    finally:
        client.close()


def test_cold_start_can_exceed_ten_seconds(delayed_client):
    result = delayed_client.initialize()
    assert result["userAgent"] == "slow-test-server"
    assert delayed_client.take_notification(timeout=5)["method"] == "test/ready"


def test_explicit_initialization_deadline_is_still_enforced(delayed_client):
    with pytest.raises(TimeoutError, match="initialize.*timed out"):
        delayed_client.initialize(timeout=0.1)
    assert not delayed_client._initialized
    assert delayed_client._pending == {}
