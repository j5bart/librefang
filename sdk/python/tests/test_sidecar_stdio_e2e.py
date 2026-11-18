"""Real-pipe end-to-end tests for the sidecar stdio path.

Every other test in this suite drives `run()` in-process with an
injectable line source — which is exactly why the 2026-11-18 18:40Z
silent loss was invisible: a 314KB FileData frame written atomically
into a real OS pipe was consumed by the reader thread's buffered
`for line in sys.stdin` and never reached the command loop, and no
test had ever exercised the bytes-between-pipe-and-queue path.

These tests spawn a real sidecar process (the same
``python -m <adapter>`` entry point the daemon uses) and write frames
to its stdin pipe the way the Rust daemon does — one atomic write per
frame — then observe protocol frames on stdout and structured logs on
stderr.
"""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path

import pytest

SDK_ROOT = Path(__file__).resolve().parents[1]

#: Minimal test-only adapter: stashes `emit` from `produce`, and echoes
#: every `send` back as a `message` event whose text is JSON describing
#: what arrived, so the tests can assert on the dispatched command.
ECHO_ADAPTER = '''
import asyncio
import json
import os

from librefang.sidecar import SidecarAdapter, run_stdio_main
from librefang.sidecar.protocol import message


class EchoAdapter(SidecarAdapter):
    def __init__(self):
        self._emit = None

    async def produce(self, emit):
        self._emit = emit

    async def on_send(self, cmd):
        deadline = asyncio.get_event_loop().time() + 5.0
        while self._emit is None:
            if asyncio.get_event_loop().time() > deadline:
                raise RuntimeError("produce never ran; emit unavailable")
            await asyncio.sleep(0.01)
        info = {"channel_id": cmd.channel_id}
        if isinstance(cmd.content, dict):
            kind = next(iter(cmd.content), None)
            info["content_kind"] = kind
            payload = cmd.content.get(kind) or {}
            if isinstance(payload, dict):
                info["filename"] = payload.get("filename")
                info["mime_type"] = payload.get("mime_type")
                data = payload.get("data")
                if isinstance(data, list):
                    info["data_len"] = len(data)
                    info["data_sum"] = sum(data)
        else:
            info["text"] = cmd.text
        block = float(os.environ.get("ECHO_ON_SEND_BLOCK_SECS", "0") or 0)
        if block > 0:
            await asyncio.sleep(block)
        self._emit(message("echo", "echo", text=json.dumps(info)))


if __name__ == "__main__":
    run_stdio_main(EchoAdapter)
'''


def _spawn(tmp_path, env_extra):
    adapter_path = tmp_path / "echo_sidecar_adapter.py"
    adapter_path.write_text(ECHO_ADAPTER)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(SDK_ROOT), str(tmp_path)])
    env.update(env_extra)
    return subprocess.Popen(
        [sys.executable, "-m", "echo_sidecar_adapter"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )


def _read_chunk(stream, deadline):
    """select() + read1() so a silent pipe cannot hang the test."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None
    ready, _, _ = select.select([stream], [], [], remaining)
    if not ready:
        return None
    chunk = stream.read1(65536)
    return chunk if chunk else None


def _next_event(proc, deadline=10.0):
    """Read stdout until the first non-ready protocol event (or deadline).

    Returns the parsed event dict, or None on timeout/EOF.
    """
    end = time.monotonic() + deadline
    buf = b""
    while True:
        chunk = _read_chunk(proc.stdout, end)
        if chunk is None:
            return None
        buf += chunk
        while b"\n" in buf:
            line, _, buf = buf.partition(b"\n")
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("method") == "ready":
                continue
            return event


def _next_stderr(proc, deadline=10.0):
    """Read stderr until at least one complete log line is available."""
    end = time.monotonic() + deadline
    buf = b""
    while b"\n" not in buf:
        chunk = _read_chunk(proc.stderr, end)
        if chunk is None:
            return buf.decode(errors="replace")
        buf += chunk
    return buf.decode(errors="replace")


def _shutdown(proc):
    try:
        proc.stdin.write(b'{"method":"shutdown"}\n')
        proc.stdin.flush()
    except (BrokenPipeError, OSError):
        pass


def _terminate(proc):
    for fn in (proc.stdin, proc.stdout, proc.stderr):
        try:
            fn.close()
        except OSError:
            pass
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _send_frame(proc, params, *, method="send"):
    frame = json.dumps({"method": method, "params": params}).encode()
    proc.stdin.write(frame + b"\n")
    proc.stdin.flush()


def test_filedata_frame_larger_than_pipe_buffer_roundtrips(tmp_path):
    # 44,536 bytes of deterministic data — the exact size of the
    # incident frame — serialises to a ~314KB JSON frame, ~5x the
    # 64KB default pipe buffer, and must arrive byte-exact.
    data = bytes((i * 37 + 11) % 256 for i in range(44536))
    proc = _spawn(tmp_path, {})
    try:
        _send_frame(proc, {
            "channel_id": "c9j46i9v",
            "text": "media",
            "content": {"FileData": {
                "data": list(data),
                "filename": "pipeline-test.jpg",
                "mime_type": "image/jpeg",
            }},
            "user": {"platform_id": "c9j46i9v",
                     "display_name": "c9j46i9v",
                     "librefang_user": None},
        })
        event = _next_event(proc)
        assert event is not None, "echo never arrived — frame was lost"
        assert event["method"] == "message"
        info = json.loads(event["params"]["text"])
        assert info["content_kind"] == "FileData"
        assert info["filename"] == "pipeline-test.jpg"
        assert info["mime_type"] == "image/jpeg"
        assert info["data_len"] == 44536
        assert info["data_sum"] == sum(data)
        _shutdown(proc)
        assert proc.wait(timeout=10) == 0
    finally:
        _terminate(proc)


def test_overlong_line_is_dropped_loudly_and_service_resyncs(tmp_path):
    # A partial line that blows past max_line_bytes must be dropped
    # with a loud stderr log, the corrupt tail skipped at the next
    # newline, and the following good frame still served.
    proc = _spawn(tmp_path, {"LIBREFANG_SIDECAR_MAX_LINE_BYTES": "1024"})
    try:
        proc.stdin.write(b'{"method":"send","params":{"text":"'
                         + b"x" * 5000)  # no newline: over the 1024 cap
        proc.stdin.flush()
        time.sleep(0.3)
        proc.stdin.write(b'"}\n')  # completes the corrupt line — skipped
        _send_frame(proc, {"channel_id": "r", "text": "after-resync",
                           "user": {}})
        event = _next_event(proc)
        assert event is not None, "no event after resync — service died"
        info = json.loads(event["params"]["text"])
        assert info["text"] == "after-resync"
        _shutdown(proc)
        proc.wait(timeout=10)
        err = proc.stderr.read().decode(errors="replace")
        assert "exceeded max_line_bytes" in err
        assert "resyncing" in err
    finally:
        _terminate(proc)


def test_partial_line_stall_is_loud_on_stderr(tmp_path):
    # The exact 18:40Z shape: bytes arrive, no newline ever does. The
    # watchdog must report it loudly within the stall threshold, and
    # completing the frame afterwards must still dispatch it.
    proc = _spawn(tmp_path, {"LIBREFANG_SIDECAR_READER_STALL_SECS": "1"})
    try:
        proc.stdin.write(b'{"method":"send","params":{"channel_id":"r",'
                         b'"text":"stalled"')
        proc.stdin.flush()
        err = _next_stderr(proc, deadline=5.0)
        assert "no newline" in err, f"watchdog silent: {err!r}"
        assert "writer may have died mid-frame" in err
        # Complete the frame — the stall must not have poisoned it.
        proc.stdin.write(b',"user":{}}}\n')
        proc.stdin.flush()
        event = _next_event(proc)
        assert event is not None, "completed frame was lost after stall"
        info = json.loads(event["params"]["text"])
        assert info["text"] == "stalled"
        _shutdown(proc)
        assert proc.wait(timeout=10) == 0
    finally:
        _terminate(proc)


def test_queue_backlog_is_loud_on_stderr(tmp_path):
    # A wedged dispatch loop (on_send blocked) must not silently
    # swallow the queued frames behind it: the watchdog reports the
    # backlog, and once the block clears every frame is served.
    proc = _spawn(tmp_path, {
        "LIBREFANG_SIDECAR_QUEUE_STALL_SECS": "0.5",
        "ECHO_ON_SEND_BLOCK_SECS": "3",
    })
    try:
        _send_frame(proc, {"channel_id": "r", "text": "one", "user": {}})
        _send_frame(proc, {"channel_id": "r", "text": "two", "user": {}})
        err = _next_stderr(proc, deadline=5.0)
        assert "queued but not dispatched" in err, f"silent: {err!r}"
        # The block clears; both frames must be served, in order.
        first = _next_event(proc, deadline=10.0)
        second = _next_event(proc, deadline=10.0)
        assert first is not None and second is not None
        assert json.loads(first["params"]["text"])["text"] == "one"
        assert json.loads(second["params"]["text"])["text"] == "two"
        _shutdown(proc)
        assert proc.wait(timeout=10) == 0
    finally:
        _terminate(proc)
