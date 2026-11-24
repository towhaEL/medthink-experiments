"""Web panel API: credential redaction, GPU usage estimate, run-time cap."""

import json
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from kaggle_hub.account_manager import AccountManager
from kaggle_hub.kaggle_runner import KaggleRunner
from kaggle_hub.models import JobRecord, JobStatus
from kaggle_hub.notebook_manager import NotebookManager
from kaggle_hub.server.api_handlers import ApiDispatcher
from kaggle_hub.storage import StorageEngine


class TestWebApi(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        nb_dir = self.tmp / "nbs"
        nb_dir.mkdir()
        (nb_dir / "demo.ipynb").write_text(json.dumps(
            {"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}))
        self.storage = StorageEngine(base_dir=self.tmp / "hub")
        self.mgr = AccountManager(storage=self.storage)
        self.runner = KaggleRunner(
            account_manager=self.mgr,
            notebook_manager=NotebookManager(notebooks_dir=nb_dir, staging_dir=self.tmp / "st"),
            storage=self.storage)
        self.api = ApiDispatcher(account_manager=self.mgr, notebook_manager=self.runner.notebook_manager,
                                 runner=self.runner, storage=self.storage)
        self.acc = self.mgr.add_account("A", "alpha", "oauth_token",
                                        {"refresh_token": "SECRET", "username": "alpha"}, make_active=True)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_no_credentials_reach_the_browser(self):
        responses = [
            self.api.dispatch("GET", "/api/accounts"),
            self.api.dispatch("GET", f"/api/accounts/{self.acc.id}"),
            self.api.dispatch("POST", f"/api/accounts/{self.acc.id}/switch"),
            self.api.dispatch("GET", "/api/status"),
            self.api.dispatch("POST", "/api/accounts",
                              body_data={"username": "beta", "api_key": "KEYSECRET"}),
        ]
        for status, body in responses:
            self.assertLess(status, 300, body)
            dumped = json.dumps(body)
            self.assertNotIn("SECRET", dumped)

    def test_gpu_usage_counts_gpu_runs_in_window_only(self):
        now = datetime.now(timezone.utc)
        iso = lambda h: (now - timedelta(hours=h)).isoformat()

        def job(i, status, start, end=None, acc="NvidiaTeslaT4", gpu=True):
            return JobRecord(id=i, account_id=self.acc.id, username="alpha", kernel_slug=i,
                             kernel_id=f"alpha/{i}", title=i, notebook_path="", staging_path="",
                             accelerator=acc, enable_gpu=gpu, status=status,
                             started_at=iso(start), completed_at=iso(end) if end is not None else None)

        for j in (job("done", JobStatus.COMPLETE, 5, 3),          # 2h
                  job("live", JobStatus.RUNNING, 1),               # ~1h, still running
                  job("cpu", JobStatus.COMPLETE, 5, 1, acc="None", gpu=False),
                  job("old", JobStatus.COMPLETE, 24 * 9, 24 * 8),   # outside 7 days
                  job("edge", JobStatus.COMPLETE, 24 * 7 + 1, 24 * 7 - 1)):  # 1h inside
            self.runner.save_job(j)

        u = self.runner.gpu_usage_by_account(days=7)[self.acc.id]
        self.assertAlmostEqual(u["gpu_hours"], 4.0, delta=0.05)
        self.assertEqual(u["active_jobs"], 1)

    @patch("subprocess.run")
    def test_launch_passes_runtime_cap_and_pushes_once(self, run):
        run.return_value = MagicMock(returncode=0, stdout="pushed", stderr="")
        with patch.object(self.runner, "run_job_async") as watcher:
            status, body = self.api.dispatch("POST", "/api/runs", body_data={
                "notebook": "demo.ipynb", "account_id": self.acc.id, "max_runtime_hours": "2.5"})
        self.assertEqual(status, 201, body)
        pushes = [c for c in run.call_args_list if c[0][0][1:3] == ["kernels", "push"]]
        self.assertEqual(len(pushes), 1)
        cmd = pushes[0][0][0]
        self.assertEqual(cmd[cmd.index("-t") + 1], "9000")
        watcher.assert_called_once()
        self.assertFalse(watcher.call_args.kwargs.get("push", False))


if __name__ == "__main__":
    unittest.main()
