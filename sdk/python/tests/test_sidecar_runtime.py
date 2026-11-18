"""Runtime behaviour tests for librefang.sidecar.runtime.

Driven through the injectable I/O of `run()` (no real subprocess):
a queue feeds stdin lines, a list captures emitted events.
"""

import asyncio
import io
import json
import threading
import time

import pytest

from librefang.sidecar import ProducerCrashed, ReaderCrashed, SidecarAdapter, run
from librefang.sidecar.protocol import Send


class RecordingAdapter(SidecarAdapter):
    capabilities = ["typing"]

    def __init__(self):
        self.sends = []
        self.commands = []
        self.shutdown_called = False

    async def on_send(self, cmd: Send) -> None:
        self.sends.append(cmd)

    async def on_command(self, cmd) -> None:
        self.commands.append(cmd)
        await super().on_command(cmd)

    async def on_shutdown(self) -> None:
        self.shutdown_called = True


async def _drive(adapter, lines, *, ready_interval=0.01, timeout=2.0):
    """Feed `lines` (then EOF) into run(); return emitted events."""
    q: asyncio.Queue = asyncio.Queue()
    for ln in lines:
        q.put_nowait(ln)
    q.put_nowait(None)  # EOF -> run() returns

    emitted = []

    async def line_source():
        return await q.get()

    await asyncio.wait_for(
        run(adapter, line_source=line_source, emit=emitted.append,
            ready_interval=ready_interval),
        timeout=timeout,
    )
    return emitted


async def test_ready_handshake_stops_after_ack():
    adapter = RecordingAdapter()
    # No ack -> several ready re-announces before EOF ends the run.
    emitted = await _drive(adapter, [], ready_interval=0.01)
    readies = [e for e in emitted if e["method"] == "ready"]
    assert len(readies) >= 1
    assert readies[0]["params"]["capabilities"] == ["typing"]

    # With an early ack, re-announcing stops; only the first ready or two.
    adapter2 = RecordingAdapter()
    emitted2 = await _drive(
        adapter2,
        ['{"method":"ready_ack"}'],
        ready_interval=0.5,  # long; ack must short-circuit the wait
    )
    assert sum(1 for e in emitted2 if e["method"] == "ready") <= 2


async def test_ready_reannounce_is_bounded_without_ack():
    # No ack ever arrives (pre-#5219 daemon). The loop must stop
    # re-announcing after ready_max_attempts instead of flooding
    # stdout forever, while the run keeps serving until shutdown.
    adapter = RecordingAdapter()
    emitted = []
    ready_cap_reached = asyncio.Event()
    delivered = False

    def emit(event):
        emitted.append(event)
        if event["method"] == "ready":
            readies = sum(item["method"] == "ready" for item in emitted)
            if readies == 3:
                ready_cap_reached.set()

    async def line_source():
        nonlocal delivered
        if not delivered:
            delivered = True
            await ready_cap_reached.wait()
            return '{"method":"shutdown"}'
        return None

    await asyncio.wait_for(
        run(adapter, line_source=line_source, emit=emit,
            ready_interval=0.01, ready_max_attempts=3),
        timeout=2.0,
    )
    readies = [e for e in emitted if e["method"] == "ready"]
    assert len(readies) == 3  # capped, not unbounded
    assert adapter.shutdown_called  # run lifecycle still intact

    # ready_max_attempts=0 keeps the legacy unbounded behaviour.
    adapter2 = RecordingAdapter()
    emitted2 = []
    fourth_ready_emitted = asyncio.Event()
    delivered2 = False

    def emit2(event):
        emitted2.append(event)
        if event["method"] == "ready":
            readies = sum(item["method"] == "ready" for item in emitted2)
            if readies == 4:
                fourth_ready_emitted.set()

    async def line_source2():
        nonlocal delivered2
        if not delivered2:
            delivered2 = True
            await fourth_ready_emitted.wait()
            return '{"method":"shutdown"}'
        return None

    await asyncio.wait_for(
        run(adapter2, line_source=line_source2, emit=emit2,
            ready_interval=0.01, ready_max_attempts=0),
        timeout=2.0,
    )
    assert sum(1 for e in emitted2 if e["method"] == "ready") >= 4


async def test_send_command_dispatched():
    adapter = RecordingAdapter()
    line = json.dumps({
        "method": "send",
        "params": {"channel_id": "c", "text": "hello",
                   "content": {"Text": "hello"},
                   "user": {"platform_id": "c", "display_name": "D",
                            "librefang_user": None}},
    })
    await _drive(adapter, ['{"method":"ready_ack"}', line])
    assert len(adapter.sends) == 1
    assert adapter.sends[0].text == "hello"
    assert adapter.sends[0].content == {"Text": "hello"}


async def test_unknown_command_does_not_crash():
    adapter = RecordingAdapter()
    await _drive(adapter, [
        '{"method":"ready_ack"}',
        '{"method":"some_future_cmd","params":{}}',
        '{"method":"send","params":{"channel_id":"c","text":"ok","user":{}}}',
    ])
    # Run survived the unknown command and still dispatched the send.
    assert any(s.text == "ok" for s in adapter.sends)


async def test_shutdown_command_ends_run_and_calls_hook():
    adapter = RecordingAdapter()
    # Shutdown before EOF; run() must return promptly and call on_shutdown.
    await _drive(adapter, ['{"method":"ready_ack"}', '{"method":"shutdown"}'])
    assert adapter.shutdown_called is True


async def test_invalid_json_emits_error_and_continues():
    adapter = RecordingAdapter()
    emitted = await _drive(adapter, [
        "not-json{",
        '{"method":"send","params":{"channel_id":"c","text":"after","user":{}}}',
    ])
    assert any(e["method"] == "error" for e in emitted)
    assert any(s.text == "after" for s in adapter.sends)


async def test_non_object_command_params_emit_error_and_continue():
    adapter = RecordingAdapter()
    emitted = await _drive(adapter, [
        '{"method":"send","params":["not","an","object"]}',
        '{"method":"send","params":{"channel_id":"c","text":"after","user":{}}}',
    ])

    assert any(e["method"] == "error" for e in emitted)
    assert [send.text for send in adapter.sends] == ["after"]


async def test_non_string_command_method_emits_error_and_continues():
    adapter = RecordingAdapter()
    emitted = await _drive(adapter, [
        '{"method":[],"params":{}}',
        '{"method":"send","params":{"channel_id":"c","text":"after","user":{}}}',
    ])

    assert any(e["method"] == "error" for e in emitted)
    assert [send.text for send in adapter.sends] == ["after"]


async def test_producer_emits_inbound_messages():
    class Producer(SidecarAdapter):
        async def on_send(self, cmd):  # unused here
            pass

        async def produce(self, emit):
            from librefang.sidecar import Content, protocol
            emit(protocol.message("u", "n", content=Content.text("tick")))
            # then idle until shutdown/EOF cancels us
            await asyncio.sleep(60)

    emitted = await _drive(Producer(), [], ready_interval=0.01)
    msgs = [e for e in emitted if e["method"] == "message"]
    assert msgs and msgs[0]["params"]["content"] == {"Text": "tick"}


@pytest.mark.parametrize("bad", ["", "   ", "\n"])
async def test_blank_lines_are_skipped(bad):
    adapter = RecordingAdapter()
    await _drive(adapter, [bad, '{"method":"shutdown"}'])
    assert adapter.shutdown_called


async def test_producer_crash_exits_nonzero_after_cleanup():
    # A fatal, unhandled producer error must surface as ProducerCrashed
    # (which run_stdio turns into a nonzero process exit, not a clean
    # shutdown), and on_shutdown cleanup must still run before it.
    class Crashing(SidecarAdapter):
        def __init__(self):
            self.shutdown_called = False

        async def on_send(self, cmd):
            pass

        async def produce(self, emit):
            raise RuntimeError("transport died unrecoverably")

        async def on_shutdown(self):
            self.shutdown_called = True

    adapter = Crashing()
    with pytest.raises(ProducerCrashed) as ei:
        await _drive(adapter, [])
    # __cause__ preserves the original transport error for diagnostics.
    assert isinstance(ei.value.__cause__, RuntimeError)
    assert "transport died unrecoverably" in str(ei.value.__cause__)
    assert adapter.shutdown_called, "cleanup must run before nonzero exit"


async def test_reader_crash_stops_run_after_cleanup():
    adapter = RecordingAdapter()

    async def broken_line_source():
        raise RuntimeError("stdin transport failed")

    with pytest.raises(ReaderCrashed) as error:
        await asyncio.wait_for(
            run(
                adapter,
                line_source=broken_line_source,
                emit=lambda _event: None,
                ready_interval=0.01,
            ),
            timeout=1.0,
        )

    assert isinstance(error.value.__cause__, RuntimeError)
    assert "stdin transport failed" in str(error.value.__cause__)
    assert adapter.shutdown_called, "cleanup must run before reader failure surfaces"


async def test_unexpected_parser_error_stops_reader_instead_of_hanging(monkeypatch):
    adapter = RecordingAdapter()
    delivered = False

    async def line_source():
        nonlocal delivered
        if not delivered:
            delivered = True
            return '{"method":"send","params":{}}'
        return None

    def broken_parser(_line):
        raise TypeError("unexpected parser failure")

    monkeypatch.setattr("librefang.sidecar.runtime.protocol.parse_command", broken_parser)

    with pytest.raises(ReaderCrashed) as error:
        await asyncio.wait_for(
            run(
                adapter,
                line_source=line_source,
                emit=lambda _event: None,
                ready_interval=0.01,
            ),
            timeout=1.0,
        )

    assert isinstance(error.value.__cause__, TypeError)
    assert adapter.shutdown_called


async def test_reader_emit_failure_stops_run_after_cleanup():
    adapter = RecordingAdapter()
    lines = iter(["not-json{", None])

    async def line_source():
        return next(lines)

    def emit(event):
        if event["method"] == "error":
            raise OSError("stdout write failed")

    with pytest.raises(ReaderCrashed) as error:
        await asyncio.wait_for(
            run(
                adapter,
                line_source=line_source,
                emit=emit,
                ready_interval=0.01,
            ),
            timeout=1.0,
        )

    assert isinstance(error.value.__cause__, OSError)
    assert adapter.shutdown_called


async def test_stdio_reader_thread_failure_reaches_async_reader(monkeypatch):
    from librefang.sidecar.runtime import _run_stdio

    class BrokenStdin:
        # The byte-level reader reaches the transport through fileno();
        # a transport that cannot even provide one fails right there.
        def fileno(self):
            raise OSError("stdin device failed")

    adapter = RecordingAdapter()
    monkeypatch.setattr("sys.stdin", BrokenStdin())
    monkeypatch.setattr("sys.stdout", io.StringIO())

    with pytest.raises(ReaderCrashed) as error:
        await asyncio.wait_for(
            _run_stdio(
                adapter,
                ready_interval=0.01,
                ready_max_attempts=1,
            ),
            timeout=1.0,
        )

    assert isinstance(error.value.__cause__, OSError)
    assert "stdin device failed" in str(error.value.__cause__)
    assert adapter.shutdown_called


def test_run_stdio_exits_nonzero_after_producer_crash(monkeypatch):
    # run_stdio is the process entry point: a ProducerCrashed from run()
    # must exit the process with status 1 so the daemon supervisor sees a
    # nonzero exit, distinguishable from a clean shutdown/EOF. os._exit is
    # patched out here because it would take the test process with it;
    # the subprocess-level proof lives in
    # test_telegram_adapter.py::test_wedged_poll_exits_the_process.
    from librefang.sidecar import run_stdio

    async def crash(*_args, **_kwargs):
        raise ProducerCrashed("producer failed")

    exits: list[int] = []
    monkeypatch.setattr("librefang.sidecar.runtime._run_stdio", crash)
    monkeypatch.setattr("librefang.sidecar.runtime.os._exit",
                        lambda code: exits.append(code))

    run_stdio(RecordingAdapter(), ready_interval=0.01, ready_max_attempts=1)

    assert exits == [1]


# ---------------------------------------------------------------------------
# Byte-level stdin pump + reader watchdog (the silent-loss hardening)
#
# The 2026-11-18 18:40Z incident: a 314KB FileData frame was written
# atomically into the sidecar's stdin pipe, the bytes were consumed by
# the `for line in sys.stdin` reader thread, and the line was never
# delivered to the command loop — no error, no log, no dispatch. The
# TextIOWrapper read-ahead owned those bytes and nothing observed them.
# The pump below reads at the byte level so every buffered byte is
# accounted for, guards against unbounded partial lines, and a
# watchdog reports stalls loudly instead of silently.
# ---------------------------------------------------------------------------


class _ChunkedStdin:
    """read_fn stub: yields the given byte chunks, then EOF (b"")."""

    def __init__(self, *chunks: bytes):
        self._chunks = list(chunks)

    def __call__(self) -> bytes:
        if self._chunks:
            return self._chunks.pop(0)
        return b""


def _pump(*chunks, max_line_bytes=1024 * 1024, progress=None):
    from librefang.sidecar.runtime import _pump_stdin_lines

    # A lone bytes argument is one chunk, not an iterable of ints.
    if len(chunks) == 1 and isinstance(chunks[0], (bytes, bytearray)):
        chunks = (bytes(chunks[0]),)
    lines = []
    _pump_stdin_lines(_ChunkedStdin(*chunks), lines.append,
                      max_line_bytes=max_line_bytes, progress=progress)
    return lines


async def test_pump_reassembles_line_split_across_reads():
    # A frame far larger than any pipe buffer arrives in many small
    # reads; the pump must reassemble it byte-exactly. In-process mirror
    # of the >64KB real-pipe e2e in test_sidecar_stdio_e2e.py.
    payload = '{"method":"send","params":{"text":"' + "x" * 200000 + '"}}'
    chunks = [payload[i:i + 7].encode() for i in range(0, len(payload), 7)]
    chunks.append(b"\n")
    lines = _pump(*chunks)
    assert lines == [payload]


async def test_pump_splits_multiple_lines_in_one_chunk():
    # Several frames arriving in one read (or one write) must all be
    # delivered; a blank line is passed through as "" for run() to skip.
    lines = _pump(b'{"a":1}\n\n{"b":2}\n')
    assert lines == ['{"a":1}', '', '{"b":2}']


async def test_pump_drops_undecodable_line_and_continues(capsys):
    # Old behaviour: a bad byte raised UnicodeDecodeError inside the
    # `for line in sys.stdin` iterator and took the whole sidecar down.
    # Now the corrupt line is dropped loudly and service continues.
    lines = _pump(b'{"a":"\xff\xfe"}\n{"ok":1}\n')
    assert lines == ['{"ok":1}']
    assert "dropping undecodable stdin line" in capsys.readouterr().err


async def test_pump_overlong_line_dropped_and_resyncs(capsys):
    # max_line_bytes=64: the 100-byte partial (no newline) trips the
    # guard; everything up to the NEXT newline is dropped as the tail
    # of the corrupt line; service resumes at the following line.
    lines = _pump(b"x" * 100, b"tail-of-corrupt-line\n", b'{"ok":1}\n',
                  max_line_bytes=64)
    assert lines == ['{"ok":1}']
    err = capsys.readouterr().err
    assert "exceeded max_line_bytes" in err
    assert "resyncing" in err


async def test_pump_reports_eof_with_partial_line(capsys):
    # A truncated final frame (writer died mid-write) must not vanish
    # silently at EOF the way it did under `for line in sys.stdin`.
    lines = _pump(b'{"method":"send","params":{"text"')
    assert lines == []
    err = capsys.readouterr().err
    assert "partial line" in err
    assert "truncated" in err


async def test_pump_progress_tracks_partial_bytes():
    from librefang.sidecar.runtime import _ReaderProgress

    progress = _ReaderProgress()
    lines = _pump(b'{"a":', b'1}\n', progress=progress)
    assert lines == ['{"a":1}']
    assert progress.partial_bytes == 0
    assert progress.partial_since is None

    progress2 = _ReaderProgress()
    _pump(b'{"partial":', progress=progress2)
    assert progress2.partial_bytes == len(b'{"partial":')
    assert progress2.partial_since is not None


async def test_watchdog_logs_partial_line_stall(capsys):
    from librefang.sidecar.runtime import _ReaderProgress, _start_watchdog

    progress = _ReaderProgress()
    progress.observe(314021)  # the incident frame size
    progress.partial_since = time.monotonic() - 2.0  # stalled 2s already

    stop = threading.Event()
    t = _start_watchdog(progress, reader_stall_secs=1.0, queue=None,
                        queue_stall_secs=30.0, check_interval=0.05,
                        stop=stop)
    try:
        deadline = time.monotonic() + 2.0
        err = ""
        while time.monotonic() < deadline:
            err += capsys.readouterr().err
            if "no newline" in err:
                break
            await asyncio.sleep(0.05)
        assert "no newline" in err, f"watchdog never reported: {err!r}"
        assert "314021" in err
    finally:
        stop.set()
        t.join(timeout=2.0)


async def test_watchdog_logs_queue_backlog(capsys):
    from librefang.sidecar.runtime import _ReaderProgress, _start_watchdog

    progress = _ReaderProgress()
    q: asyncio.Queue = asyncio.Queue()
    q.put_nowait('{"one":1}')
    q.put_nowait('{"two":2}')

    stop = threading.Event()
    t = _start_watchdog(progress, reader_stall_secs=30.0, queue=q,
                        queue_stall_secs=0.2, check_interval=0.05,
                        stop=stop)
    try:
        deadline = time.monotonic() + 2.0
        err = ""
        while time.monotonic() < deadline:
            err += capsys.readouterr().err
            if "queued but not dispatched" in err:
                break
            await asyncio.sleep(0.05)
        assert "queued but not dispatched" in err, f"never reported: {err!r}"
        assert '"queued": 2' in err
    finally:
        stop.set()
        t.join(timeout=2.0)


async def test_watchdog_quiet_when_progress_is_healthy(capsys):
    from librefang.sidecar.runtime import _ReaderProgress, _start_watchdog

    progress = _ReaderProgress()
    q: asyncio.Queue = asyncio.Queue()
    stop = threading.Event()
    t = _start_watchdog(progress, reader_stall_secs=0.1, queue=q,
                        queue_stall_secs=0.1, check_interval=0.05,
                        stop=stop)
    try:
        await asyncio.sleep(0.3)
        err = capsys.readouterr().err
        assert "no newline" not in err
        assert "queued but not dispatched" not in err
    finally:
        stop.set()
        t.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Send-ack sink (P0 delivery confirmation)
#
# `send` was fire-and-forget: the daemon's tool result only proved the
# pipe write, so a send that died inside the adapter (or never got
# dispatched) still reported success. When the daemon requests acks
# (LIBREFANG_SIDECAR_SEND_ACKS) and the adapter declares the
# `send_acks` capability, the runtime installs a sink the adapter
# reports delivery outcomes through, and emits send_failed itself when
# on_command raises for a stamped send.
# ---------------------------------------------------------------------------


class _AckingAdapter(SidecarAdapter):
    capabilities = ["send_acks"]

    def __init__(self):
        self.acked = []

    async def on_send(self, cmd):
        self.acked.append(cmd.id)
        self.ack_send(cmd.id)


async def test_ack_sink_installed_when_requested_and_supported(monkeypatch):
    monkeypatch.setenv("LIBREFANG_SIDECAR_SEND_ACKS", "1")
    adapter = _AckingAdapter()
    emitted = await _drive(adapter, [
        '{"method":"ready_ack"}',
        '{"method":"send","params":{"channel_id":"c","text":"hi",'
        '"id":"cmd-1"}}',
        '{"method":"shutdown"}',
    ])
    assert adapter.acked == ["cmd-1"]
    assert [e for e in emitted if e["method"] == "send_ack"] == [
        {"method": "send_ack", "params": {"id": "cmd-1"}},
    ]


async def test_ack_sink_absent_without_env_request(monkeypatch):
    monkeypatch.delenv("LIBREFANG_SIDECAR_SEND_ACKS", raising=False)
    adapter = _AckingAdapter()
    emitted = await _drive(adapter, [
        '{"method":"ready_ack"}',
        '{"method":"send","params":{"channel_id":"c","text":"hi",'
        '"id":"cmd-2"}}',
        '{"method":"shutdown"}',
    ])
    assert adapter.acked == ["cmd-2"]  # ack_send ran, but silently no-oped
    assert not [e for e in emitted if e["method"] == "send_ack"]


async def test_ack_sink_absent_without_capability(monkeypatch):
    monkeypatch.setenv("LIBREFANG_SIDECAR_SEND_ACKS", "1")

    class NoCapAdapter(_AckingAdapter):
        capabilities = []  # daemon asked, adapter never opted in

    adapter = NoCapAdapter()
    emitted = await _drive(adapter, [
        '{"method":"ready_ack"}',
        '{"method":"send","params":{"channel_id":"c","text":"hi",'
        '"id":"cmd-3"}}',
        '{"method":"shutdown"}',
    ])
    assert not [e for e in emitted if e["method"] == "send_ack"]


async def test_failed_send_command_emits_send_failed(monkeypatch):
    monkeypatch.setenv("LIBREFANG_SIDECAR_SEND_ACKS", "1")

    class FailingAdapter(SidecarAdapter):
        capabilities = ["send_acks"]

        async def on_send(self, cmd):
            raise RuntimeError("nextcloud 401 — token rejected")

    emitted = await _drive(FailingAdapter(), [
        '{"method":"ready_ack"}',
        '{"method":"send","params":{"channel_id":"c","text":"hi",'
        '"id":"cmd-9"}}',
        '{"method":"shutdown"}',
    ])
    fails = [e for e in emitted if e["method"] == "send_failed"]
    assert len(fails) == 1
    assert fails[0]["params"]["id"] == "cmd-9"
    assert "token rejected" in fails[0]["params"]["error"]


async def test_ack_send_with_error_emits_send_failed(monkeypatch):
    monkeypatch.setenv("LIBREFANG_SIDECAR_SEND_ACKS", "1")

    class MediaFailAdapter(SidecarAdapter):
        capabilities = ["send_acks"]

        async def on_send(self, cmd):
            # The adapter handled the failure internally (chat notice
            # posted) — it still owes the daemon a send_failed.
            self.ack_send(cmd.id, error="malformed byte array")

    emitted = await _drive(MediaFailAdapter(), [
        '{"method":"ready_ack"}',
        '{"method":"send","params":{"channel_id":"c","text":"hi",'
        '"id":"cmd-4"}}',
        '{"method":"shutdown"}',
    ])
    assert [e for e in emitted if e["method"] == "send_failed"] == [
        {"method": "send_failed",
         "params": {"id": "cmd-4", "error": "malformed byte array"}},
    ]
