"""Audit C3/C4: run artifacts are private files and tool errors never echo
URL credentials."""

from __future__ import annotations

import os
import stat


def test_trace_and_run_files_are_private(tmp_path, monkeypatch):
    monkeypatch.setattr("os.umask", lambda m: 0o000)
    os.umask(0o000)
    from apodex.run_layout import secure_open
    from apodex.trace import TraceObserver
    t = TraceObserver(str(tmp_path / "trace.jsonl"), mode="react", cwd=str(tmp_path))
    t._write({"t": "x"})
    assert stat.S_IMODE(os.stat(tmp_path / "trace.jsonl").st_mode) == 0o600
    p = tmp_path / "loose.json"
    p.write_text("x")
    os.chmod(p, 0o664)
    with secure_open(p, "w") as f:
        f.write("y")
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o600 and p.read_text() == "y"


def test_error_strings_do_not_echo_url_credentials():
    from plugins.tools._bounded_fetch import redact_secrets
    msg = ("Client error '401 Unauthorized' for url "
           "'https://user:s3cret@proxy.example/v1/search?q=x&key=QUERYKEY&api_key=K2'")
    out = redact_secrets(msg)
    assert "s3cret" not in out and "QUERYKEY" not in out and "K2" not in out
    assert "https://***@proxy.example/v1/search?q=x&key=***&api_key=***" in out
    assert redact_secrets("plain text, no urls") == "plain text, no urls"
