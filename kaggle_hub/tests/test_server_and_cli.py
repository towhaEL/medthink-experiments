"""
Unit and integration tests for Kaggle Hub CLI and Web Server REST API.
Zero external dependencies.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from kaggle_hub import cli
from kaggle_hub.account_manager import AccountManager
from kaggle_hub.models import Account, JobRecord, JobStatus, RunOptions
from kaggle_hub.notebook_manager import NotebookManager
from kaggle_hub.kaggle_runner import KaggleRunner
from kaggle_hub.storage import StorageEngine
from kaggle_hub.server.http_server import KaggleHubServer


class TestCliCommands(unittest.TestCase):
    """Test CLI argument parsing and execution handlers."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.base_dir = Path(self.temp_dir)
        self.storage = StorageEngine(base_dir=self.base_dir)
        self.acc_mgr = AccountManager(storage=self.storage, base_dir=self.base_dir)

        # Create dummy notebook for testing
        self.nb_dir = self.base_dir / "test_notebooks"
        self.nb_dir.mkdir(parents=True, exist_ok=True)
        self.dummy_nb = self.nb_dir / "00_test_nb.ipynb"
        nb_json = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["# Test Pipeline Notebook\n", "Introductory description."],
                },
                {
                    "cell_type": "code",
                    "execution_count": 1,
                    "metadata": {},
                    "outputs": [],
                    "source": ["import os\n", "print('Running test')"],
                },
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        with open(self.dummy_nb, "w", encoding="utf-8") as f:
            json.dump(nb_json, f)

        self.nb_mgr = NotebookManager(notebooks_dir=self.nb_dir, staging_dir=self.base_dir / "staging")
        self.runner = KaggleRunner(
            account_manager=self.acc_mgr,
            notebook_manager=self.nb_mgr,
            storage=self.storage,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cli_account_lifecycle(self) -> None:
        # Patch managers in cli.main
        with patch("kaggle_hub.cli.StorageEngine", return_value=self.storage), \
             patch("kaggle_hub.cli.AccountManager", return_value=self.acc_mgr):

            # 1. account list when empty
            out = io.StringIO()
            with patch("sys.stdout", out):
                code = cli.main(["account", "list"])
            self.assertEqual(code, 0)
            self.assertIn("No Kaggle accounts configured yet", out.getvalue())

            # 2. account add
            out = io.StringIO()
            with patch("sys.stdout", out):
                code = cli.main([
                    "account", "add",
                    "--username", "alice",
                    "--key", "key1234567890",
                    "--label", "Alice CoAuthor",
                    "--active"
                ])
            self.assertEqual(code, 0)
            self.assertIn("Added API Key account", out.getvalue())

            # Verify active account
            active = self.acc_mgr.get_active_account()
            self.assertIsNotNone(active)
            self.assertEqual(active.username, "alice")

            # 3. account add second account
            code = cli.main([
                "account", "add",
                "--username", "bob",
                "--key", "bobkey1234567890",
                "--label", "Bob GPU",
            ])
            self.assertEqual(code, 0)

            # 4. account switch
            out = io.StringIO()
            with patch("sys.stdout", out):
                code = cli.main(["account", "switch", "bob"])
            self.assertEqual(code, 0)
            self.assertEqual(self.acc_mgr.get_active_account().username, "bob")

            # 5. account list with accounts
            out = io.StringIO()
            with patch("sys.stdout", out):
                code = cli.main(["account", "list"])
            self.assertEqual(code, 0)
            self.assertIn("bob", out.getvalue())
            self.assertIn("alice", out.getvalue())

            # 6. account test (mock subprocess)
            with patch("subprocess.run") as mock_proc:
                mock_proc.return_value = MagicMock(returncode=0, stdout="Success", stderr="")
                out = io.StringIO()
                with patch("sys.stdout", out):
                    code = cli.main(["account", "test"])
                self.assertEqual(code, 0)
                self.assertIn("[PASS]", out.getvalue())

            # 7. account remove
            out = io.StringIO()
            with patch("sys.stdout", out):
                code = cli.main(["account", "remove", "alice"])
            self.assertEqual(code, 0)
            self.assertIsNone(self.acc_mgr.get_account("alice"))

    def test_cli_notebook_commands(self) -> None:
        with patch("kaggle_hub.cli.StorageEngine", return_value=self.storage), \
             patch("kaggle_hub.cli.AccountManager", return_value=self.acc_mgr), \
             patch("kaggle_hub.cli.NotebookManager", return_value=self.nb_mgr):

            # notebook list
            out = io.StringIO()
            with patch("sys.stdout", out):
                code = cli.main(["notebook", "list"])
            self.assertEqual(code, 0)
            self.assertIn("00_test_nb.ipynb", out.getvalue())
            self.assertIn("Test Pipeline Notebook", out.getvalue())

            # notebook inspect
            out = io.StringIO()
            with patch("sys.stdout", out):
                code = cli.main(["notebook", "inspect", "00_test_nb.ipynb"])
            self.assertEqual(code, 0)
            output = out.getvalue()
            self.assertIn("Notebook Inspection", output)
            self.assertIn("Total Cells", output)

    def test_cli_run_commands(self) -> None:
        self.acc_mgr.add_account("Tester", "testuser", credentials={"key": "key123"}, make_active=True)

        with patch("kaggle_hub.cli.StorageEngine", return_value=self.storage), \
             patch("kaggle_hub.cli.AccountManager", return_value=self.acc_mgr), \
             patch("kaggle_hub.cli.NotebookManager", return_value=self.nb_mgr), \
             patch("kaggle_hub.cli.KaggleRunner", return_value=self.runner):

            # Mock push subprocess
            with patch("subprocess.run") as mock_sub:
                mock_sub.return_value = MagicMock(returncode=0, stdout="Kernel version 1 successfully pushed", stderr="")
                out = io.StringIO()
                with patch("sys.stdout", out):
                    code = cli.main(["run", "push", "00_test_nb.ipynb", "--no-gpu"])
                self.assertEqual(code, 0)
                self.assertIn("pushed successfully", out.getvalue())

            # run list
            out = io.StringIO()
            with patch("sys.stdout", out):
                code = cli.main(["run", "list"])
            self.assertEqual(code, 0)
            self.assertIn("testuser", out.getvalue())

            # run status
            with patch("subprocess.run") as mock_sub:
                mock_sub.return_value = MagicMock(returncode=0, stdout="Your kernel has completed", stderr="")
                out = io.StringIO()
                with patch("sys.stdout", out):
                    code = cli.main(["run", "status"])
                self.assertEqual(code, 0)
                self.assertIn("COMPLETE", out.getvalue())

            # run logs
            with patch("subprocess.run") as mock_sub:
                mock_sub.return_value = MagicMock(returncode=0, stdout="Log line 1\nLog line 2", stderr="")
                out = io.StringIO()
                with patch("sys.stdout", out):
                    code = cli.main(["run", "logs"])
                self.assertEqual(code, 0)
                self.assertIn("Log line 1", out.getvalue())


class TestHttpServerAndApi(unittest.TestCase):
    """Test HTTP Server and REST API endpoints."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.mkdtemp()
        cls.base_dir = Path(cls.temp_dir)
        cls.storage = StorageEngine(base_dir=cls.base_dir)
        cls.acc_mgr = AccountManager(storage=cls.storage, base_dir=cls.base_dir)

        # Add initial test account
        cls.acc_mgr.add_account(
            label="Main Account",
            username="main_user",
            auth_type="api_key",
            credentials={"key": "testkey123"},
            make_active=True,
        )

        # Create dummy notebook
        cls.nb_dir = cls.base_dir / "notebooks"
        cls.nb_dir.mkdir(parents=True, exist_ok=True)
        dummy_nb = cls.nb_dir / "01_sample.ipynb"
        with open(dummy_nb, "w", encoding="utf-8") as f:
            json.dump({
                "cells": [
                    {"cell_type": "markdown", "source": ["# Sample Step 1"]},
                    {"cell_type": "code", "source": ["print(123)"]},
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }, f)

        cls.nb_mgr = NotebookManager(notebooks_dir=cls.nb_dir, staging_dir=cls.base_dir / "staging")
        cls.runner = KaggleRunner(
            account_manager=cls.acc_mgr,
            notebook_manager=cls.nb_mgr,
            storage=cls.storage,
        )

        # Start server on ephemeral port (port=0 lets OS choose free port)
        cls.server = KaggleHubServer(
            host="127.0.0.1",
            port=0,
            account_manager=cls.acc_mgr,
            notebook_manager=cls.nb_mgr,
            runner=cls.runner,
            storage=cls.storage,
        )
        cls.server.start_background()
        cls.port = cls.server.server_port
        cls.base_url = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def _request(
        self,
        endpoint: str,
        method: str = "GET",
        data: dict | None = None,
    ) -> tuple[int, dict | str]:
        url = f"{self.base_url}{endpoint}"
        body = json.dumps(data).encode("utf-8") if data else None
        headers = {"Content-Type": "application/json"} if data else {}
        req = Request(url, data=body, headers=headers, method=method)

        try:
            with urlopen(req, timeout=5) as response:
                content_type = response.headers.get("Content-Type", "")
                raw = response.read()
                if "application/json" in content_type:
                    return response.status, json.loads(raw.decode("utf-8"))
                return response.status, raw.decode("utf-8")
        except HTTPError as e:
            raw = e.read()
            try:
                err_data = json.loads(raw.decode("utf-8"))
                return e.code, err_data
            except Exception:
                return e.code, raw.decode("utf-8")

    def test_api_status(self) -> None:
        status, data = self._request("/api/status")
        self.assertEqual(status, 200)
        self.assertEqual(data["status"], "ok")
        self.assertIn("kaggle_cli", data)
        self.assertIsNotNone(data["active_account"])
        self.assertEqual(data["active_account"]["username"], "main_user")

    def test_api_accounts_crud_and_switch(self) -> None:
        # GET /api/accounts
        status, data = self._request("/api/accounts")
        self.assertEqual(status, 200)
        self.assertEqual(len(data["accounts"]), 1)

        # POST /api/accounts (Add new account)
        status, data = self._request("/api/accounts", method="POST", data={
            "label": "Secondary GPU",
            "username": "second_user",
            "auth_type": "api_key",
            "api_key": "secondkey123",
            "make_active": False,
        })
        self.assertEqual(status, 201)
        acc_id = data["account"]["id"]
        self.assertEqual(data["account"]["username"], "second_user")

        # POST /api/accounts/:id/switch
        status, data = self._request(f"/api/accounts/{acc_id}/switch", method="POST")
        self.assertEqual(status, 200)
        self.assertEqual(data["active_account_id"], acc_id)

        # POST /api/accounts/:id/test (mock subprocess)
        with patch("subprocess.run") as mock_sub:
            mock_sub.return_value = MagicMock(returncode=0, stdout="Authorized", stderr="")
            status, data = self._request(f"/api/accounts/{acc_id}/test", method="POST")
            self.assertEqual(status, 200)
            self.assertTrue(data["valid"])

        # DELETE /api/accounts/:id
        status, data = self._request(f"/api/accounts/{acc_id}", method="DELETE")
        self.assertEqual(status, 200)
        self.assertTrue(data["success"])

    def test_api_notebooks(self) -> None:
        # GET /api/notebooks
        status, data = self._request("/api/notebooks")
        self.assertEqual(status, 200)
        self.assertGreaterEqual(len(data["notebooks"]), 1)
        self.assertEqual(data["notebooks"][0]["name"], "01_sample.ipynb")

        # GET /api/notebooks/:name
        status, data = self._request("/api/notebooks/01_sample.ipynb")
        self.assertEqual(status, 200)
        self.assertEqual(data["notebook"]["slug"], "medthink-01-sample")
        self.assertGreaterEqual(len(data["cell_preview"]), 1)

    def test_api_runs_lifecycle(self) -> None:
        # POST /api/runs (mock push)
        with patch("subprocess.run") as mock_push:
            mock_push.return_value = MagicMock(returncode=0, stdout="Kernel pushed", stderr="")
            status, data = self._request("/api/runs", method="POST", data={
                "notebook": "01_sample.ipynb",
                "accelerator": "NvidiaTeslaT4",
                "enable_gpu": True,
                "auto_download": False,
            })
            self.assertEqual(status, 201)
            job = data["job"]
            job_id = job["id"]
            self.assertEqual(job["status"], JobStatus.RUNNING)

        # GET /api/runs
        status, data = self._request("/api/runs")
        self.assertEqual(status, 200)
        self.assertGreaterEqual(len(data["jobs"]), 1)

        # GET /api/runs/:id
        with patch("subprocess.run") as mock_status:
            mock_status.return_value = MagicMock(returncode=0, stdout="Your kernel has completed", stderr="")
            status, data = self._request(f"/api/runs/{job_id}?refresh=true")
            self.assertEqual(status, 200)
            self.assertEqual(data["job"]["status"], JobStatus.COMPLETE)

        # GET /api/runs/:id/logs
        with patch("subprocess.run") as mock_logs:
            mock_logs.return_value = MagicMock(returncode=0, stdout="Sample execution logs...", stderr="")
            status, data = self._request(f"/api/runs/{job_id}/logs")
            self.assertEqual(status, 200)
            self.assertIn("Sample execution logs", data["logs"])

    def test_static_asset_serving(self) -> None:
        # GET /
        status, html = self._request("/")
        self.assertEqual(status, 200)
        self.assertIn("Kaggle Hub", html)
        self.assertIn("class=\"tabs\"", html)

        # GET /styles.css
        status, css = self._request("/styles.css")
        self.assertEqual(status, 200)
        self.assertIn(":root", css)

        # GET /app.js
        status, js = self._request("/app.js")
        self.assertEqual(status, 200)
        self.assertIn("STATE", js)

        # Traversal attack prevention
        status, _ = self._request("/../../etc/passwd")
        self.assertIn(status, [400, 403, 404])


if __name__ == "__main__":
    unittest.main()
