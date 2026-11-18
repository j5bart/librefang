"""Runtime for LibreFang sidecar channel adapters.

Subclass :class:`SidecarAdapter`, implement :meth:`SidecarAdapter.on_send`
(and optionally :meth:`SidecarAdapter.produce` to push inbound platform
messages), then ``run_stdio(MyAdapter())``. The framework owns:

* the stdin command-reader loop and JSON parsing;
* the ``ready`` handshake — it re-announces ``ready`` (with your
  declared capabilities) every ``ready_interval`` until LibreFang sends
  ``ready_ack``, so a fresh process is idempotently re-discovered.
  Re-announcing is bounded by ``ready_max_attempts`` (``0`` = forever):
  after that many un-acked emits it stops re-announcing but keeps
  running, so a pre-#5219 daemon (no ``ready_ack``) gets the first
  ``ready`` and the adapter keeps working without spamming stdout;
* graceful ``shutdown`` (runs :meth:`SidecarAdapter.on_shutdown`);
* keeping **stdout** free of anything but protocol frames (log via
  :mod:`librefang.sidecar.logging`, which writes stderr).

The ``ready`` → ``ready_ack`` handshake targets the **post-#5219**
sidecar daemon (see :mod:`librefang.sidecar.protocol`). ``main``'s
daemon has no ``ready_ack`` command; against a pre-#5219 daemon the
loop simply stops re-announcing after ``ready_max_attempts`` (the
first ``ready`` is still delivered, so the adapter functions). This
SDK is wire-complete ahead of #5219 landing — merge #5219 first for
the full handshake; it degrades cleanly on the current ``main``.

Responsibility split — read this:

* **Process restart is the daemon's job.** LibreFang's supervisor
  respawns a crashed sidecar with backoff and a circuit-breaker. Your
  adapter must be *crash-safe*: hold no irreplaceable in-process state,
  and let the framework re-announce ``ready`` on each fresh start. Do
  **not** try to keep your own process alive across a fatal error.
* **Platform reconnect is the adapter's job.** Reconnecting a dropped
  Telegram long-poll / WebSocket is your transport's concern. Use
  :func:`with_backoff` for that loop; it is independent of the
  daemon-managed process lifecycle.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
import traceback
from typing import Any, Awaitable, Callable, Dict, List, Optional

from . import logging as log
from . import protocol
from .protocol import (
    Command,
    ReadyAck,
    Send,
    Shutdown,
)

EmitFn = Callable[[Dict[str, Any]], None]
LineSource = Callable[[], Awaitable[Optional[str]]]


class _ReaderThreadFailure:
    """Carries a synchronous stdin failure across the asyncio queue."""

    def __init__(self, error: Exception) -> None:
        self.error = error


class ProducerCrashed(Exception):
    """Raised from :func:`run` when :meth:`SidecarAdapter.produce` exits
    with an unhandled error.

    Cleanup (``on_shutdown``) has already run by the time this is raised.
    :func:`run_stdio` exits the process with status 1 so the daemon
    supervisor sees a nonzero exit (distinguishable from a clean
    ``shutdown``/EOF). Library callers using :func:`run` directly can
    catch this to inspect ``__cause__`` (the original transport error)
    before deciding how to surface the failure."""


class ReaderCrashed(Exception):
    """Raised from :func:`run` when the command reader exits with an
    unexpected error.

    Cleanup has already run. :func:`run_stdio` exits with status 1 so the
    daemon supervisor can restart the sidecar instead of leaving a process
    alive with no command consumer."""


class SidecarAdapter:
    """Base class for a sidecar channel adapter.

    Override :meth:`on_send` (required) and, for platforms you poll,
    :meth:`produce`. Declare optional capabilities so LibreFang routes
    rich features (typing/reaction/interactive/thread/streaming/
    typing_events) to you instead of degrading to plain text.
    """

    #: Capability strings, e.g. ``["typing", "interactive"]``.
    capabilities: List[str] = []
    #: Multi-bot account id, if this adapter is one of several.
    account_id: Optional[str] = None
    #: Post errors to the user privately (log-only) when True.
    suppress_error_responses: bool = False
    #: Operator inbox(es) for non-conversational broadcasts.
    notification_recipients: List[Dict[str, Any]] = []
    #: ``[(host, [[k, v], ...]), ...]`` auth headers for media fetch.
    header_rules: List[Any] = []
    #: Wire-protocol version announced on ``ready``.
    #:
    #: Defaults to :data:`librefang.sidecar.protocol.PROTOCOL_VERSION` so every adapter declares it without a per-adapter line — the shape of #7140, where the field existed on both sides of the wire, was documented as "current value: 1", and was pinned at 1 in the shared conformance corpus, yet no adapter ever set it and every real ``ready`` frame carried ``null``.
    #: Override only when an adapter deliberately speaks an older protocol.
    protocol_version: Optional[int] = protocol.PROTOCOL_VERSION

    def ready_event(self) -> Dict[str, Any]:
        return protocol.ready(
            self.capabilities,
            self.account_id,
            self.suppress_error_responses,
            self.notification_recipients,
            self.header_rules,
            self.protocol_version,
        )

    async def on_send(self, cmd: Send) -> None:
        """Deliver an outbound message to the platform. Required."""
        raise NotImplementedError(
            "override on_send() to deliver messages to your platform"
        )

    async def on_command(self, cmd: Command) -> None:
        """Dispatch a command. Default routes ``send`` to
        :meth:`on_send`; typing/reaction/interactive/stream_* are no-ops
        unless you override this. ``ready_ack``/``shutdown`` are handled
        by the framework and never reach here."""
        if isinstance(cmd, Send):
            await self.on_send(cmd)

    async def produce(self, emit: EmitFn) -> None:
        """Optional: pull inbound platform events and ``emit(event)``
        (build events with :mod:`librefang.sidecar.protocol`). Default:
        nothing — for command/webhook-only adapters."""
        return

    async def on_shutdown(self) -> None:
        """Optional cleanup on a clean shutdown."""
        return


async def with_backoff(
    fn: Callable[[], Awaitable[None]],
    *,
    initial: float = 1.0,
    maximum: float = 30.0,
    factor: float = 2.0,
) -> None:
    """Retry ``fn`` with exponential backoff until it returns without
    raising. For *platform* reconnection only — process restart is the
    daemon's job (see module docstring). Propagates cancellation."""
    delay = initial
    while True:
        try:
            await fn()
            return
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - transport errors vary
            log.warn("operation failed; backing off",
                     error=str(e), delay=delay)
            await asyncio.sleep(delay)
            delay = min(delay * factor, maximum)


def describe_main(adapter):
    """Print the adapter's SCHEMA as JSON to stdout and return 0.

    Returns 2 (no schema declared) — same exit code as the missing-token
    case in telegram.py so the daemon's describe-cache logic can treat
    "no schema" identically to "describe failed" and fall back.
    """
    import json as _json
    schema = getattr(adapter, "SCHEMA", None)
    if schema is None:
        # Log via stderr (stdout is reserved for the JSON payload).
        log.error("adapter has no SCHEMA attribute; --describe failed",
                  adapter=type(adapter).__name__)
        return 2
    sys.stdout.write(_json.dumps(schema.to_dict()))
    sys.stdout.flush()
    return 0


async def run(
    adapter: SidecarAdapter,
    *,
    line_source: LineSource,
    emit: EmitFn,
    ready_interval: float = 2.0,
    ready_max_attempts: int = 5,
) -> None:
    """Drive an adapter against injectable I/O. ``line_source`` returns
    the next stdin line or ``None`` at EOF; ``emit`` writes one event.
    Returns when LibreFang sends ``shutdown`` or stdin reaches EOF.

    ``ready_max_attempts`` bounds the un-acked ``ready`` re-announce
    (``0`` = re-announce forever). After the cap the loop stops
    re-announcing but the run continues — a pre-#5219 daemon that never
    sends ``ready_ack`` still gets the first ``ready`` and the adapter
    keeps serving without flooding stdout."""
    acked = asyncio.Event()
    stop = asyncio.Event()
    producer_error: Optional[BaseException] = None
    reader_error: Optional[BaseException] = None

    async def ready_loop() -> None:
        attempts = 0
        while not acked.is_set() and not stop.is_set():
            emit(adapter.ready_event())
            attempts += 1
            try:
                await asyncio.wait_for(acked.wait(), timeout=ready_interval)
            except asyncio.TimeoutError:
                pass  # re-announce; ack is idempotent
            if (not acked.is_set() and ready_max_attempts > 0
                    and attempts >= ready_max_attempts):
                log.warn(
                    "no ready_ack received; assuming a pre-#5219 daemon "
                    "(no ack command) and stopping ready re-announce — "
                    "the adapter keeps running on the first ready",
                    attempts=attempts,
                )
                return

    async def producer() -> None:
        nonlocal producer_error
        try:
            await adapter.produce(emit)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log.error("producer crashed", error=str(e))
            producer_error = e
            stop.set()

    async def reader() -> None:
        nonlocal reader_error
        try:
            while not stop.is_set():
                line = await line_source()
                if line is None:
                    stop.set()
                    return
                line = line.strip()
                if not line:
                    continue
                try:
                    cmd = protocol.parse_command(line)
                except json.JSONDecodeError as e:
                    emit(protocol.error(f"invalid JSON: {e}"))
                    continue
                if isinstance(cmd, ReadyAck):
                    acked.set()
                    continue
                if isinstance(cmd, Shutdown):
                    stop.set()
                    return
                try:
                    await adapter.on_command(cmd)
                except Exception as e:  # noqa: BLE001
                    # The bare except is load-bearing: we cannot raise out
                    # of `reader()` without taking down the entire sidecar
                    # process for one bad command. But the previous
                    # `error=str(e)` lost the traceback, which has burned
                    # us repeatedly — the four channel reply-routing
                    # regressions fixed in #5417→#5423→#5431→#5439 all
                    # took multiple post-merge review passes to surface
                    # because the original failure looked like
                    # ``on_command failed error="missing parent fullname"``
                    # with no clue WHERE it was raised. Include the
                    # traceback as a structured field so debuggers can
                    # grep for the offending adapter / line directly.
                    #
                    # The richer long-term fix — surfacing on_command
                    # failures as a structured JSON-RPC error response so
                    # the bridge can correlate the failure to the original
                    # cmd id and the agent's error-handling lane fires —
                    # is tracked in a follow-up; it needs a protocol
                    # change (Response.error field) that this hotfix
                    # doesn't ship.
                    log.error(
                        "on_command failed",
                        error=str(e),
                        error_type=type(e).__name__,
                        traceback=traceback.format_exc(),
                        cmd_type=type(cmd).__name__,
                    )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            reader_error = e
            log.error(
                "reader crashed",
                error=str(e),
                error_type=type(e).__name__,
                traceback=traceback.format_exc(),
            )
            stop.set()

    tasks = [
        asyncio.ensure_future(ready_loop()),
        asyncio.ensure_future(producer()),
        asyncio.ensure_future(reader()),
    ]
    try:
        await stop.wait()
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        try:
            await adapter.on_shutdown()
        except Exception as e:  # noqa: BLE001
            log.error("on_shutdown failed", error=str(e))
    # A producer crash is a fatal, unhandled adapter failure. Surface it
    # as a regular Exception (with the original error chained); the
    # `run_stdio` wrapper converts that into a nonzero process exit so it
    # is distinguishable from a clean `shutdown`/EOF (cleanup above still
    # ran). The daemon supervisor restarts on any non-shutdown exit; the
    # nonzero code makes the failure explicit to operators and any
    # non-supervised runner. Raising a plain Exception here (rather than
    # SystemExit) keeps `run` library-friendly: callers driving it from
    # their own event loop can catch and inspect the failure without
    # tripping interpreter-level exit semantics — and the assertion is
    # portable across Python versions (a SystemExit raised from inside an
    # awaited sub-Task short-circuits the event loop on <3.11, making the
    # contract unobservable to test code on supported runtimes).
    if producer_error is not None:
        raise ProducerCrashed(
            "sidecar producer crashed; cleanup ran"
        ) from producer_error
    if reader_error is not None:
        raise ReaderCrashed(
            "sidecar reader crashed; cleanup ran"
        ) from reader_error


def run_stdio(adapter: SidecarAdapter, *, ready_interval: float = 2.0,
              ready_max_attempts: int = 5) -> None:
    """Wire ``run`` to real stdio and run it to completion. stdout is
    written under a lock and carries only protocol frames; stdin is read
    on a daemon thread (portable, unlike async stdin). See :func:`run`
    for ``ready_max_attempts``.

    With ``--describe`` in ``sys.argv``, emit the adapter's SCHEMA JSON to
    stdout and exit (via :func:`describe_main`) before starting the
    stdio JSON-RPC loop. Note: this branch is only reachable if the
    caller already managed to *construct* the adapter — adapters that
    require env vars in ``__init__`` cannot be described via this path.
    Use :func:`run_stdio_main` (which takes the class, not an instance)
    from your ``__main__`` block instead; this fallback exists only to
    defend direct ``run_stdio(instance)`` callers.

    A fatal reader or producer error from ``run`` exits the process with
    status 1 so the daemon supervisor (and any non-supervised runner) sees a
    nonzero exit code, distinguishable from a clean ``shutdown``/EOF. The
    exit happens before ``asyncio.run`` unwinds — see
    :func:`_run_stdio_or_exit`."""
    if "--describe" in sys.argv[1:]:
        raise SystemExit(describe_main(adapter))
    asyncio.run(_run_stdio_or_exit(adapter, ready_interval=ready_interval,
                                   ready_max_attempts=ready_max_attempts))


async def _run_stdio_or_exit(adapter: SidecarAdapter, *,
                             ready_interval: float,
                             ready_max_attempts: int) -> None:
    """Run the stdio loop, exiting the process on a fatal reader or
    producer crash.

    A poll parked past its deadline owns an executor thread that never
    returns. ``asyncio.run`` joins the default executor while unwinding
    (300 s on 3.12+, unbounded on 3.10/3.11), and
    ``concurrent.futures.thread`` joins every worker at interpreter exit, so
    raising out of the loop would log the crash and hang instead of exiting.
    ``os._exit`` is the only exit that does not wait on that thread; flushing
    first keeps the "producer crashed" log line on stderr.
    """
    try:
        await _run_stdio(adapter, ready_interval=ready_interval,
                         ready_max_attempts=ready_max_attempts)
    except (ProducerCrashed, ReaderCrashed):
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


def run_stdio_main(adapter_class):
    """Entry point for ``if __name__ == "__main__"`` blocks.

    Dispatches ``--describe`` from ``sys.argv`` against the adapter CLASS
    (using its class-level ``SCHEMA``) without instantiating it — so
    adapters whose ``__init__`` requires env vars (TELEGRAM_BOT_TOKEN,
    NTFY_TOPIC, etc.) still describe correctly when the daemon spawns
    them with no env to enumerate the form fields.

    For the normal serve path, instantiates the class and hands off
    to :func:`run_stdio`.
    """
    if "--describe" in sys.argv[1:]:
        raise SystemExit(describe_main(adapter_class))
    run_stdio(adapter_class())


# ---------------------------------------------------------------------------
# Byte-level stdin reading + reader watchdog
#
# The reader used to be ``for line in sys.stdin``, which hides every
# byte inside TextIOWrapper's read-ahead. On 2026-11-18 a 314KB
# FileData frame written atomically into the stdin pipe was consumed by
# that buffer and never reached the command loop — no error, no log,
# no dispatch. Reading at the byte level makes every buffered byte
# observable, the max-line guard bounds a corrupt stream, and the
# watchdog reports stalls loudly instead of silently.
# ---------------------------------------------------------------------------

#: Bytes fetched per read(2); a frame larger than this arrives across
#: several reads, exactly as it arrives across several pipe writes.
READ_CHUNK_BYTES = 65536

#: Default cap on a single unterminated line. 64 MiB is far above the
#: largest legitimate frame (a 44KB file serialises to ~314KB) but
#: bounds a runaway or corrupt writer.
DEFAULT_MAX_LINE_BYTES = 64 * 1024 * 1024

#: Default time an unterminated partial line may sit in the reader
#: buffer before the watchdog reports it.
DEFAULT_READER_STALL_SECS = 30.0

#: Default time a parsed line may sit in the command queue before the
#: watchdog reports the dispatch loop as wedged.
DEFAULT_QUEUE_STALL_SECS = 30.0


def _env_int(name: str, default: int, override: Optional[int] = None) -> int:
    if override is not None:
        return override
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        log.warn("ignoring invalid env override", env=name, value=raw)
        return default
    if value <= 0:
        log.warn("ignoring non-positive env override", env=name, value=raw)
        return default
    return value


def _env_float(name: str, default: float,
               override: Optional[float] = None) -> float:
    if override is not None:
        return override
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        log.warn("ignoring invalid env override", env=name, value=raw)
        return default
    if value <= 0:
        log.warn("ignoring non-positive env override", env=name, value=raw)
        return default
    return value


class _ReaderProgress:
    """Bytes currently buffered without a newline, and since when.

    Written only by the reader thread and read only by the watchdog
    thread; single-attribute writes are atomic under the GIL.
    """

    def __init__(self) -> None:
        self.partial_bytes = 0
        self.partial_since: Optional[float] = None
        self.last_report: Optional[float] = None

    def observe(self, buffered: int) -> None:
        now = time.monotonic()
        if buffered > 0:
            if self.partial_since is None:
                self.partial_since = now
        else:
            self.partial_since = None
        self.partial_bytes = buffered


def _stdin_read_fn() -> Callable[[], bytes]:
    """Byte-level reader for the real stdin.

    The pump must own every byte between the pipe and the command
    loop, so it reads the file descriptor directly instead of going
    through TextIOWrapper's read-ahead — the layer where a partial
    frame could previously sit, unobservable by anyone.
    """
    try:
        fd = sys.stdin.fileno()
    except OSError:
        raise  # transport-level failure: surface it verbatim
    except Exception as e:  # noqa: BLE001
        raise OSError("stdin is not a real file descriptor; "
                      "byte-level reading is unavailable") from e
    return lambda: os.read(fd, READ_CHUNK_BYTES)


def _pump_stdin_lines(read_fn: Callable[[], bytes],
                      put_line: Callable[[str], None], *,
                      max_line_bytes: int,
                      progress: Optional[_ReaderProgress] = None) -> None:
    """Read stdin byte-by-byte and hand complete lines to ``put_line``.

    Guarantees the old ``for line in sys.stdin`` loop did not make:

    * a line larger than any pipe buffer is reassembled byte-exactly
      from however many reads it arrives in;
    * an unterminated line that exceeds ``max_line_bytes`` is dropped
      with a loud log and the stream resynced at the next newline;
    * an undecodable line is dropped loudly instead of killing the
      reader;
    * EOF with a partial line is reported instead of silently
      discarding the truncated frame.
    """
    buf = bytearray()
    resyncing = False
    while True:
        chunk = read_fn()
        if not chunk:
            break
        buf += chunk
        while True:
            nl = buf.find(b"\n")
            if nl < 0:
                break
            raw = bytes(buf[:nl])
            del buf[:nl + 1]
            if resyncing:
                # Tail of a dropped over-long line: skip through the
                # newline that terminated it.
                resyncing = False
                continue
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError as e:
                log.error("reader: dropping undecodable stdin line",
                          error=str(e), line_bytes=len(raw))
                continue
            put_line(line)
        if progress is not None:
            progress.observe(len(buf))
        if buf and not resyncing and len(buf) > max_line_bytes:
            log.error("reader: stdin line exceeded max_line_bytes; dropping "
                      "the buffered bytes and resyncing at the next newline",
                      buffered_bytes=len(buf), max_line_bytes=max_line_bytes)
            resyncing = True
            buf.clear()
            if progress is not None:
                progress.observe(0)
    if buf:
        log.error("reader: EOF with a partial line — the final frame was "
                  "truncated", buffered_bytes=len(buf))


def _start_watchdog(progress: _ReaderProgress, *,
                    reader_stall_secs: float,
                    queue: Optional["asyncio.Queue[Any]"],
                    queue_stall_secs: float,
                    stop: threading.Event,
                    check_interval: float = 1.0) -> threading.Thread:
    """Watch for the two silent-wedge shapes and report them loudly.

    * bytes buffered with no newline for ``reader_stall_secs`` — the
      writer died mid-frame (the 2026-11-18 18:40Z shape);
    * parsed lines queued but not dispatched for
      ``queue_stall_secs`` — the command loop is wedged.
    """

    def _watch() -> None:
        queue_nonempty_since: Optional[float] = None
        while not stop.is_set():
            now = time.monotonic()
            since = progress.partial_since
            if progress.partial_bytes > 0 and since is not None:
                age = now - since
                last = progress.last_report
                if age >= reader_stall_secs and (
                        last is None or now - last >= reader_stall_secs):
                    progress.last_report = now
                    log.error("reader: stdin bytes buffered with no newline "
                              "— the writer may have died mid-frame",
                              buffered_bytes=progress.partial_bytes,
                              stalled_secs=round(age, 1),
                              hint="the daemon-side write did not complete; "
                                   "a partial line cannot be dispatched")
            if queue is not None:
                qsize = queue.qsize()
                if qsize == 0:
                    queue_nonempty_since = None
                elif queue_nonempty_since is None:
                    queue_nonempty_since = now
                elif now - queue_nonempty_since >= queue_stall_secs:
                    log.error("reader: command lines queued but not "
                              "dispatched — the command loop may be wedged",
                              queued=qsize,
                              stalled_secs=round(
                                  now - queue_nonempty_since, 1))
                    # Re-arm so the backlog is reported at most once
                    # per interval, not once per check.
                    queue_nonempty_since = now
            stop.wait(check_interval)

    t = threading.Thread(target=_watch, daemon=True,
                         name="sidecar-reader-watchdog")
    t.start()
    return t


async def _run_stdio(adapter: SidecarAdapter, *,
                     ready_interval: float,
                     ready_max_attempts: int,
                     max_line_bytes: Optional[int] = None,
                     reader_stall_secs: Optional[float] = None,
                     queue_stall_secs: Optional[float] = None) -> None:
    loop = asyncio.get_event_loop()
    queue: "asyncio.Queue[Any]" = asyncio.Queue()

    max_line_bytes = _env_int("LIBREFANG_SIDECAR_MAX_LINE_BYTES",
                              DEFAULT_MAX_LINE_BYTES, max_line_bytes)
    reader_stall_secs = _env_float("LIBREFANG_SIDECAR_READER_STALL_SECS",
                                    DEFAULT_READER_STALL_SECS,
                                    reader_stall_secs)
    queue_stall_secs = _env_float("LIBREFANG_SIDECAR_QUEUE_STALL_SECS",
                                  DEFAULT_QUEUE_STALL_SECS,
                                  queue_stall_secs)

    progress = _ReaderProgress()

    def _reader_thread() -> None:
        try:
            read_fn = _stdin_read_fn()
            put_line = lambda line: loop.call_soon_threadsafe(  # noqa: E731
                queue.put_nowait, line)
            _pump_stdin_lines(read_fn, put_line,
                              max_line_bytes=max_line_bytes,
                              progress=progress)
        except Exception as e:  # noqa: BLE001
            loop.call_soon_threadsafe(
                queue.put_nowait, _ReaderThreadFailure(e),
            )
        else:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=_reader_thread, daemon=True,
                     name="sidecar-stdin-reader").start()

    watchdog_stop = threading.Event()
    _start_watchdog(progress,
                    reader_stall_secs=reader_stall_secs,
                    queue=queue,
                    queue_stall_secs=queue_stall_secs,
                    stop=watchdog_stop)

    async def line_source() -> Optional[str]:
        item = await queue.get()
        if isinstance(item, _ReaderThreadFailure):
            raise item.error
        return item

    write_lock = threading.Lock()

    def emit(event: Dict[str, Any]) -> None:
        data = json.dumps(event) + "\n"
        with write_lock:
            sys.stdout.write(data)
            sys.stdout.flush()

    try:
        await run(adapter, line_source=line_source, emit=emit,
                  ready_interval=ready_interval,
                  ready_max_attempts=ready_max_attempts)
    finally:
        watchdog_stop.set()
