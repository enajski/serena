"""GDScript language server integration for Serena.

This module provides a SolidLSP-compatible wrapper around the GDScript language
server that ships with the Godot game engine. Godot exposes its LSP server over
TCP or (in newer versions) STDIO. Serena requires a STDIO interface, so we
translate between STDIO and the TCP-based protocol when necessary.
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from overrides import override

from solidlsp.ls import SolidLanguageServer
from solidlsp.ls_config import LanguageServerConfig
from solidlsp.ls_logger import LanguageServerLogger
from solidlsp.ls_exceptions import SolidLSPException
from solidlsp.lsp_protocol_handler.lsp_types import InitializeParams
from solidlsp.lsp_protocol_handler.server import ProcessLaunchInfo
from solidlsp.settings import SolidLSPSettings

_LOG = logging.getLogger(__name__)


class GDScriptLanguageServer(SolidLanguageServer):
    """SolidLanguageServer implementation for GDScript projects."""

    _STDIO_TRANSPORT = "stdio"
    _TCP_TRANSPORT = "tcp"
    _AUTO_TRANSPORT = "auto"

    def __init__(
        self,
        config: LanguageServerConfig,
        logger: LanguageServerLogger,
        repository_root_path: str,
        solidlsp_settings: SolidLSPSettings,
    ) -> None:
        godot_executable = self._resolve_godot_executable()
        extra_args = self._resolve_additional_arguments()
        transport = self._determine_transport(godot_executable)

        if transport == self._STDIO_TRANSPORT:
            cmd = self._build_stdio_command(godot_executable, repository_root_path, extra_args)
        else:
            cmd = self._build_proxy_command(godot_executable, repository_root_path, extra_args)

        logger.log(
            f"Using Godot executable '{godot_executable}' with {transport} transport",
            logging.INFO,
        )

        super().__init__(
            config,
            logger,
            repository_root_path,
            ProcessLaunchInfo(cmd=cmd, cwd=repository_root_path),
            "gdscript",
            solidlsp_settings,
        )
        self.server_ready = threading.Event()

    @staticmethod
    def _resolve_godot_executable() -> str:
        """Return the path to the Godot binary or raise if not available."""

        candidate_env_vars = [
            "SERENA_GODOT_BIN",
            "GODOT_BIN",
            "GODOT4_BIN",
        ]
        for env_var in candidate_env_vars:
            value = os.environ.get(env_var)
            if not value:
                continue
            candidate = shutil.which(value) if not os.path.isabs(value) else value
            if candidate and os.path.exists(candidate):
                return candidate

        for executable in ["godot", "godot4", "godot4-bin"]:
            candidate = shutil.which(executable)
            if candidate:
                return candidate

        raise RuntimeError(
            "Could not locate the Godot executable. Set GODOT_BIN (or SERENA_GODOT_BIN) "
            "to the absolute path of your Godot binary."
        )

    @staticmethod
    def _resolve_additional_arguments() -> list[str]:
        """Parse optional extra CLI arguments for the Godot process."""

        args_env = os.environ.get("SERENA_GODOT_LSP_ARGS") or os.environ.get("GODOT_LSP_ARGS")
        if not args_env:
            return []
        return shlex.split(args_env)

    def _determine_transport(self, godot_executable: str) -> str:
        """Determine which transport Serena should use."""

        configured = os.environ.get("SERENA_GODOT_LSP_TRANSPORT") or os.environ.get("GODOT_LSP_TRANSPORT")
        if configured:
            configured = configured.lower()
            if configured == 'websocket':
                return self._TCP_TRANSPORT
            if configured not in {self._STDIO_TRANSPORT, self._TCP_TRANSPORT, self._AUTO_TRANSPORT}:
                raise ValueError(
                    "Unrecognised Godot LSP transport '%s'. Supported values: stdio, tcp, auto" % configured
                )
            if configured != self._AUTO_TRANSPORT:
                return configured

        return self._STDIO_TRANSPORT if self._supports_stdio_transport(godot_executable) else self._TCP_TRANSPORT

    @staticmethod
    def _supports_stdio_transport(godot_executable: str) -> bool:
        """Check whether the installed Godot binary advertises STDIO transport."""

        try:
            result = subprocess.run(
                [godot_executable, "--help"],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False

        help_text = (result.stdout or "") + (result.stderr or "")
        return "lsp=stdio" in help_text.lower()

    @staticmethod
    def _build_stdio_command(godot_executable: str, repository_root_path: str, extra_args: Iterable[str]) -> list[str]:
        """Create the command list for STDIO-capable Godot binaries."""

        cmd = [
            godot_executable,
            "--headless",
            "--editor",
            "--path",
            repository_root_path,
            "--lsp=stdio",
        ]
        cmd.extend(extra_args)
        return cmd

    @staticmethod
    def _build_proxy_command(godot_executable: str, repository_root_path: str, extra_args: Iterable[str]) -> list[str]:
        """Create the proxy command that bridges STDIO to the TCP-based LSP."""

        cmd = [
            sys.executable,
            "-m",
            "solidlsp.language_servers.gdscript_language_server",
            "--proxy",
            "--godot-bin",
            godot_executable,
            "--project-path",
            repository_root_path,
        ]
        if extra_args:
            cmd.append("--")
            cmd.extend(extra_args)
        return cmd

    @staticmethod
    def _get_initialize_params(repository_absolute_path: str) -> InitializeParams:
        root_uri = Path(repository_absolute_path).as_uri()
        initialize_params: InitializeParams = {  # type: ignore
            "processId": os.getpid(),
            "clientInfo": {"name": "Serena", "version": "0.1"},
            "rootPath": repository_absolute_path,
            "rootUri": root_uri,
            "workspaceFolders": [
                {
                    "uri": root_uri,
                    "name": os.path.basename(repository_absolute_path),
                }
            ],
            "capabilities": {
                "workspace": {
                    "workspaceFolders": True,
                    "didChangeConfiguration": {"dynamicRegistration": True},
                },
                "textDocument": {
                    "synchronization": {"didSave": True, "dynamicRegistration": True},
                    "documentSymbol": {
                        "hierarchicalDocumentSymbolSupport": True,
                        "dynamicRegistration": True,
                    },
                    "references": {"dynamicRegistration": True},
                },
            },
        }
        return initialize_params

    @override
    def is_ignored_dirname(self, dirname: str) -> bool:
        return super().is_ignored_dirname(dirname) or dirname in {".godot", ".import", ".mono", ".cache"}


    @override
    def _shutdown(self, timeout: float = 5.0):
        original_shutdown = self.server.shutdown

        def safe_shutdown():
            try:
                original_shutdown()
            except SolidLSPException as exc:
                cause = getattr(exc, 'cause', None)
                if getattr(cause, 'code', None) == -32601:
                    self.logger.log("Godot language server does not implement shutdown; proceeding without it.", logging.DEBUG)
                    try:
                        self.server.notify.exit()
                        self.logger.log("Sent exit notification after missing shutdown response.", logging.DEBUG)
                    except Exception as exit_exc:
                        self.logger.log(f"Failed to send exit notification: {exit_exc}", logging.DEBUG)
                else:
                    raise

        self.server.shutdown = safe_shutdown  # type: ignore[attr-defined]
        try:
            super()._shutdown(timeout)
        finally:
            self.server.shutdown = original_shutdown  # type: ignore[attr-defined]

    def _start_server(self):  # noqa: D401 - follows SolidLanguageServer interface
        """Start the Godot-backed language server."""

        def do_nothing(_params):
            return None

        def window_log_message(msg):
            self.logger.log(f"LSP: window/logMessage: {msg}", logging.INFO)

        def workspace_configuration(_params):
            return []

        self.server.on_request("workspace/configuration", workspace_configuration)
        self.server.on_request("client/registerCapability", do_nothing)
        self.server.on_notification("window/logMessage", window_log_message)
        self.server.on_notification("textDocument/publishDiagnostics", do_nothing)
        self.server.on_notification("$/progress", do_nothing)

        self.server.start()
        initialize_params = self._get_initialize_params(self.repository_root_path)

        self.logger.log("Initializing GDScript language server", logging.INFO)
        self.server.send.initialize(initialize_params)
        self.server.notify.initialized({})
        self.completions_available.set()
        self.server_ready.set()
        self.server_ready.wait()

        # No additional readiness signals required; Godot starts serving immediately after initialization.


# --------------------------------------------------------------------------------------
# Proxy implementation (STDIO <-> TCP bridge)
# --------------------------------------------------------------------------------------

@dataclass
class _ProxyArgs:
    godot_bin: str
    project_path: str
    connect_timeout: float
    startup_timeout: float
    explicit_port: Optional[int]
    godot_args: list[str]


def _parse_proxy_args(argv: list[str]) -> _ProxyArgs:
    parser = argparse.ArgumentParser(description="STDIO/TCP bridge for the Godot GDScript language server")
    parser.add_argument("--godot-bin", required=True)
    parser.add_argument("--project-path", required=True)
    parser.add_argument("--connect-timeout", type=float, default=30.0)
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument("--port", type=int, default=0, help="Port to expose the Godot LSP on (0 chooses a free port)")
    parser.add_argument("--", dest="godot_args", nargs=argparse.REMAINDER, help="Additional arguments for Godot")

    namespace = parser.parse_args(argv)
    godot_args = namespace.godot_args or []
    if godot_args and godot_args[0] == "--":
        godot_args = godot_args[1:]

    return _ProxyArgs(
        godot_bin=namespace.godot_bin,
        project_path=namespace.project_path,
        connect_timeout=namespace.connect_timeout,
        startup_timeout=namespace.startup_timeout,
        explicit_port=namespace.port if namespace.port > 0 else None,
        godot_args=godot_args,
    )


def _reserve_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.getsockname()[1]


def _ensure_default_godot_args(args: _ProxyArgs, port: int) -> tuple[list[str], int]:
    godot_args = list(args.godot_args)
    if not godot_args:
        godot_args = ["--headless", "--editor"]

    # Ensure project path is set
    if "--path" not in godot_args and not any(item.startswith("--path=") for item in godot_args):
        godot_args.extend(["--path", args.project_path])

    port_value: Optional[int] = None

    def _extract_port(value: str) -> Optional[int]:
        try:
            return int(value)
        except ValueError:
            return None

    idx = 0
    while idx < len(godot_args):
        token = godot_args[idx]
        if token in {"--lsp-port", "--lsp_port"}:
            if idx + 1 < len(godot_args):
                port_value = _extract_port(godot_args[idx + 1])
                idx += 1
        elif token.startswith("--lsp-port="):
            port_value = _extract_port(token.split("=", 1)[1])
        elif token.startswith("--lsp_port="):
            port_value = _extract_port(token.split("=", 1)[1])
        idx += 1

    if port_value is None:
        port_value = port
        godot_args.extend(["--lsp-port", str(port)])

    if not any(token == "--lsp" or token.startswith("--lsp=") for token in godot_args):
        godot_args.append("--lsp")

    return godot_args, port_value


def _drain_stream(stream, prefix: str, stop_event: threading.Event) -> None:
    for line in iter(stream.readline, b""):
        if stop_event.is_set():
            break
        try:
            decoded = line.decode("utf-8", errors="replace")
        except Exception:
            decoded = repr(line)
        sys.stderr.write(f"[godot {prefix}] {decoded}")
        sys.stderr.flush()
    stop_event.set()


def _read_lsp_message(stream) -> Optional[bytes]:
    headers: dict[str, str] = {}
    while True:
        line = stream.readline()
        if not line:
            return None
        try:
            decoded = line.decode("ascii", errors="ignore")
        except Exception:
            decoded = line.decode("utf-8", errors="ignore")
        decoded = decoded.strip()
        if not decoded:
            break
        key, _, value = decoded.partition(":")
        headers[key.lower()] = value.strip()

    content_length = int(headers.get("content-length", "0"))
    payload = b""
    while len(payload) < content_length:
        chunk = stream.read(content_length - len(payload))
        if not chunk:
            break
        payload += chunk
    if len(payload) < content_length:
        return None
    return payload


def _send_lsp_message(stream, payload: bytes) -> None:
    header = f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
    stream.write(header)
    stream.write(payload)
    stream.flush()


def _run_proxy(argv: list[str]) -> int:
    args = _parse_proxy_args(argv)
    port = args.explicit_port or _reserve_free_port()
    godot_args, port = _ensure_default_godot_args(args, port)

    sys.stderr.write("Starting Godot language server via: {} {}\n".format(args.godot_bin, ' '.join(godot_args)))
    sys.stderr.flush()

    godot_proc = subprocess.Popen(
        [args.godot_bin, *godot_args],
        cwd=args.project_path,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stop_event = threading.Event()

    stdout_thread = threading.Thread(target=_drain_stream, args=(godot_proc.stdout, "stdout", stop_event), daemon=True)
    stderr_thread = threading.Thread(target=_drain_stream, args=(godot_proc.stderr, "stderr", stop_event), daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    deadline = time.monotonic() + args.startup_timeout
    sock: socket.socket | None = None
    last_error: Optional[Exception] = None

    while time.monotonic() < deadline:
        if godot_proc.poll() is not None:
            stop_event.set()
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)
            return godot_proc.returncode or 1
        try:
            sock = socket.create_connection(("127.0.0.1", port), timeout=args.connect_timeout)
            sock.settimeout(None)
            break
        except OSError as exc:  # pragma: no cover - depends on runtime availability
            last_error = exc
            time.sleep(0.5)

    if sock is None:
        stop_event.set()
        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)
        if godot_proc.poll() is None:
            godot_proc.terminate()
            try:
                godot_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                godot_proc.kill()
        error_message = f"Failed to connect to Godot language server at 127.0.0.1:{port}: {last_error}"
        sys.stderr.write(error_message + "\n")
        return 1

    stdin_buffer = sys.stdin.buffer
    stdout_buffer = sys.stdout.buffer
    sock_file = sock.makefile("rb")

    def _forward_stdin() -> None:
        try:
            while not stop_event.is_set():
                payload = _read_lsp_message(stdin_buffer)
                if payload is None:
                    break
                header = f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
                sock.sendall(header + payload)
        finally:
            stop_event.set()
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()

    def _forward_socket() -> None:
        try:
            while not stop_event.is_set():
                payload = _read_lsp_message(sock_file)
                if payload is None:
                    break
                _send_lsp_message(stdout_buffer, payload)
        finally:
            stop_event.set()

    stdin_thread = threading.Thread(target=_forward_stdin, daemon=True)
    socket_thread = threading.Thread(target=_forward_socket, daemon=True)
    stdin_thread.start()
    socket_thread.start()

    while not stop_event.is_set():
        if godot_proc.poll() is not None:
            stop_event.set()
            break
        time.sleep(0.1)

    stdin_thread.join(timeout=1)
    socket_thread.join(timeout=1)
    stdout_thread.join(timeout=1)
    stderr_thread.join(timeout=1)

    if godot_proc.poll() is None:
        godot_proc.terminate()
        try:
            godot_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            godot_proc.kill()
            godot_proc.wait(timeout=2)

    return godot_proc.returncode or 0


def _main(argv: list[str]) -> int:
    if len(argv) > 0 and argv[0] == "--proxy":
        return _run_proxy(argv[1:])
    parser = argparse.ArgumentParser(description="GDScript language server helper")
    parser.add_argument("--proxy", action="store_true", help="Run STDIO/TCP proxy")
    args, remainder = parser.parse_known_args(argv)
    if args.proxy:
        return _run_proxy(remainder)
    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(_main(sys.argv[1:]))
