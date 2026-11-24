"""
Zero-dependency ThreadingHTTPServer with CORS, static routing, and graceful shutdown.
"""

from __future__ import annotations

import json
import mimetypes
import os
import signal
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .api_handlers import ApiDispatcher
from ..account_manager import AccountManager
from ..notebook_manager import NotebookManager
from ..kaggle_runner import KaggleRunner
from ..storage import StorageEngine


STATIC_DIR = Path(__file__).parent / "static"


class KaggleHubRequestHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests for Kaggle Hub REST API and Static UI."""

    # Server instance references dispatcher
    dispatcher: ApiDispatcher

    def log_message(self, format: str, *args: Any) -> None:
        """Override to provide quiet or formatted terminal logging."""
        # Suppress noise during test execution or normal static asset polling
        if "GET /static/" in (args[0] if args else ""):
            return
        # sys.stdout.write(f"[HTTP] {self.address_string()} - {format % args}\n")

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight requests."""
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query = parsed_url.query

        # Check API routes
        if path.startswith("/api/"):
            status_code, response_data = self.dispatcher.dispatch("GET", path, query_string=query)
            self._send_json(status_code, response_data)
            return

        # Check Static assets
        self._serve_static(path)

    def do_POST(self) -> None:
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query = parsed_url.query

        content_length = int(self.headers.get("Content-Length", 0))
        body_data = {}
        if content_length > 0:
            try:
                body_bytes = self.rfile.read(content_length)
                body_data = json.loads(body_bytes.decode("utf-8"))
            except Exception:
                body_data = {}

        if path.startswith("/api/"):
            status_code, response_data = self.dispatcher.dispatch("POST", path, query_string=query, body_data=body_data)
            self._send_json(status_code, response_data)
            return

        self._send_json(404, {"error": "Not Found"})

    def do_DELETE(self) -> None:
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query = parsed_url.query

        if path.startswith("/api/"):
            status_code, response_data = self.dispatcher.dispatch("DELETE", path, query_string=query)
            self._send_json(status_code, response_data)
            return

        self._send_json(404, {"error": "Not Found"})

    def _send_json(self, status_code: int, data: Any) -> None:
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status_code)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, req_path: str) -> None:
        # Normalize request path
        if req_path in ["", "/", "/index.html"]:
            file_path = STATIC_DIR / "index.html"
        elif req_path.startswith("/static/"):
            rel = req_path[len("/static/"):].lstrip("/")
            file_path = (STATIC_DIR / rel).resolve()
        elif req_path in ["/app.js", "/styles.css"]:
            file_path = (STATIC_DIR / req_path.lstrip("/")).resolve()
        else:
            file_path = (STATIC_DIR / req_path.lstrip("/")).resolve()

        # Prevent directory traversal
        try:
            file_path.relative_to(STATIC_DIR.resolve())
        except ValueError:
            self._send_json(403, {"error": "Forbidden"})
            return

        if not file_path.is_file():
            self._send_json(404, {"error": f"File not found: {req_path}"})
            return

        content_type, _ = mimetypes.guess_type(str(file_path))
        if not content_type:
            if file_path.suffix == ".css":
                content_type = "text/css"
            elif file_path.suffix == ".js":
                content_type = "application/javascript"
            else:
                content_type = "text/plain"

        try:
            with open(file_path, "rb") as f:
                content = f.read()

            self.send_response(HTTPStatus.OK)
            self._send_cors_headers()
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self._send_json(500, {"error": f"Failed to read file: {e}"})


class KaggleHubServer:
    """Zero-dependency ThreadingHTTPServer wrapper with lifecycle control."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        account_manager: AccountManager | None = None,
        notebook_manager: NotebookManager | None = None,
        runner: KaggleRunner | None = None,
        storage: StorageEngine | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.storage = storage or StorageEngine()
        self.account_manager = account_manager or AccountManager(storage=self.storage)
        self.notebook_manager = notebook_manager or NotebookManager()
        self.runner = runner or KaggleRunner(
            account_manager=self.account_manager,
            notebook_manager=self.notebook_manager,
            storage=self.storage,
        )

        self.dispatcher = ApiDispatcher(
            account_manager=self.account_manager,
            notebook_manager=self.notebook_manager,
            runner=self.runner,
            storage=self.storage,
        )

        # Create handler class with injected dispatcher
        class Handler(KaggleHubRequestHandler):
            pass

        Handler.dispatcher = self.dispatcher

        self.httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        # In case port was 0, retrieve bound port
        self.server_port = self.httpd.server_port
        self._server_thread: threading.Thread | None = None
        self._is_running = False

    def start_background(self) -> None:
        """Starts server in a background daemon thread (useful for testing & embeddings)."""
        if self._is_running:
            return
        self._is_running = True
        self._server_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._server_thread.start()

    def serve_forever(self) -> None:
        """Starts server blocking current thread."""
        self._is_running = True
        try:
            self.httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Gracefully stop server."""
        if self._is_running:
            self._is_running = False
            self.httpd.shutdown()
            self.httpd.server_close()
            if self._server_thread and self._server_thread.is_alive():
                self._server_thread.join(timeout=2)


def run_server(host: str = "127.0.0.1", port: int = 8080, open_browser: bool = False) -> int:
    """CLI runner helper for serving the web dashboard."""
    server = KaggleHubServer(host=host, port=port)
    url = f"http://{host}:{server.server_port}"
    print(f"\n Kaggle Hub Web Dashboard is running at: {url}")
    print(" Press Ctrl+C to stop the server.\n")

    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    def sig_handler(sig: Any, frame: Any) -> None:
        print("\nShutting down Kaggle Hub server...")
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    server.serve_forever()
    return 0
