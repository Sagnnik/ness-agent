from __future__ import annotations

import asyncio
import json
import os
import shutil
from collections import defaultdict
from contextlib import suppress
from pathlib import Path
from typing import Any


class CodexUnavailable(RuntimeError):
    pass


def _package_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("ness-agent")
    except PackageNotFoundError:
        return "0.0.0"


class CodexAppServer:
    """Small JSONL/JSON-RPC client for the system Codex app-server."""

    def __init__(self, codex_home: Path) -> None:
        # path to codex home directory ~/.config/ness-agent/codex/ different from ~/.codex
        self.codex_home = codex_home
        # Codex app server background process
        self._process: asyncio.subprocess.Process | None = None
        # Async Task for reading the stdout of the codex app server
        self._reader_task: asyncio.Task[None] | None = None
        # Dict stores request ID and it corresponding future. Just need to await them.
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}

        # Dict of notifications ("turn/started", "turn/completed", etc.) and corresponding queues.
        self._notifications: dict[str, list[asyncio.Queue[dict[str, Any]]]] = (
            defaultdict(list)
        )
        # store the event {"turn/start" : params dict, ...}. Creates a pub/sub pattern with notifications of the same method
        self._notification_history: dict[str, list[dict[str, Any]]] = defaultdict(list)

        self._next_id = 1  # keep track of the next request ID
        self._lock = asyncio.Lock()  # only one request

    async def start(self) -> None:
        # check process and codex executable exists
        if self._process is not None and self._process.returncode is None:
            return
        executable = shutil.which("codex")
        if executable is None:
            raise CodexUnavailable("Codex CLI is not installed or is not on PATH.")

        # create codex home directory with owner read, write, execute permissions
        self.codex_home.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self.codex_home, 0o700)
        except OSError:
            pass
        env = dict(os.environ)
        env["CODEX_HOME"] = str(self.codex_home)
        # start codex app server
        # stdio is default while websockets is experimental
        self._process = await asyncio.create_subprocess_exec(
            executable,
            "app-server",
            "--stdio",
            "-c",
            'cli_auth_credentials_store="file"',  # file | auto | keyring
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,  # might need to be kept for debugging. let's see
            env=env,
        )
        self._reader_task = asyncio.create_task(
            self._read_loop()
        )  # task to read the stdout of the _process
        # start the jrpc handshake
        # Lifecycle: initialize request -> initialize response -> initialized notification -> thread/start -> turn/start ...
        await self._request_once(
            "initialize",
            {
                "clientInfo": {
                    "name": "ness-agent",
                    "title": "Ness Agent",
                    "version": _package_version(),
                },
                "capabilities": {"experimentalApi": True},
            },
        )
        await self.notify("initialized", {})  # send the initialized notification

    async def _read_loop(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        # read the stdout of the _process line by line; gives exactly one protocol message.
        # OpenAI documents stdio transport as newline-delimited JSON
        while line := await process.stdout.readline():
            try:
                message = json.loads(line)
            except (UnicodeDecodeError, ValueError):
                continue
            if not isinstance(message, dict):
                continue

            request_id = message.get("id")

            # if the request ID is in the pending dict, set the future result
            if isinstance(request_id, int) and request_id in self._pending:
                future = self._pending.pop(request_id)
                if not future.done():
                    future.set_result(message)
                continue  # RPC response; go to next line

            # Event notification (not RPC response - no request ID)
            # add to the event history and notify the waiting tasks
            method = message.get("method")
            if isinstance(method, str):  # check for event eg. "turn/start"
                params = dict(message.get("params") or {})
                # store the event history upto 20 events
                self._notification_history[method].append(params)
                self._notification_history[method] = self._notification_history[method][
                    -20:
                ]
                for queue in tuple(self._notifications.get(method, ())):
                    # add the param to the queue
                    queue.put_nowait(params)

        error = CodexUnavailable("Codex app-server stopped unexpectedly.")
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)
        self._pending.clear()

    async def request(
        self, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if self._process is None:
            await self.start()
        try:
            return await self._request_once(method, params)
        except (CodexUnavailable, BrokenPipeError, ConnectionResetError, TimeoutError):
            await self.restart()
            return await self._request_once(method, params)

    async def _request_once(
        self, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        async with self._lock:
            process = self._process
            if (
                process is None
                or process.stdin is None
                or process.returncode is not None
            ):
                raise CodexUnavailable("Codex app-server is unavailable.")

            request_id = self._next_id
            self._next_id += 1

            # create and store the future
            future: asyncio.Future[dict[str, Any]] = (
                asyncio.get_running_loop().create_future()
            )
            self._pending[request_id] = future

            payload = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params or {},
            }
            process.stdin.write(
                (json.dumps(payload, separators=(",", ":")) + "\n").encode()
            )  # separators to not include whitespaces
            await process.stdin.drain()
        response = await asyncio.wait_for(future, timeout=120)
        if response.get("error"):
            error = response["error"]
            message = error.get("message") if isinstance(error, dict) else str(error)
            raise RuntimeError(f"Codex {method} failed: {message}")
        result = response.get("result")
        return dict(result) if isinstance(result, dict) else {}

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise CodexUnavailable("Codex app-server is unavailable.")
        payload = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        process.stdin.write(
            (json.dumps(payload, separators=(",", ":")) + "\n").encode()
        )
        await process.stdin.drain()

    async def wait_notification(
        self, method: str, *, predicate=None, timeout: float = 300
    ) -> dict[str, Any]:
        # Event waiting api
        # check the event history first
        for params in reversed(self._notification_history.get(method, ())):
            if predicate is None or predicate(
                params
            ):  # check True/False for the predicate
                return params
        # create the queue _read_loop() -> [event, event] -> wait_notification()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._notifications[method].append(queue)  # subscribe to the method
        try:
            while True:
                params = await asyncio.wait_for(queue.get(), timeout=timeout)
                if predicate is None or predicate(params):
                    return params
        finally:
            self._notifications[method].remove(queue)  # remove subscriber

    async def restart(self) -> None:
        await self.close()
        await self.start()

    async def close(self) -> None:
        process, self._process = self._process, None
        task, self._reader_task = self._reader_task, None
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=3)
            except TimeoutError:
                process.kill()
                await process.wait()
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        error = CodexUnavailable("Codex app-server closed.")
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)
        self._pending.clear()
