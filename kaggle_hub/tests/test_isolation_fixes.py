"""
Regression tests for multi-account isolation and run-lifecycle fixes.

  * OAuth accounts get a private HOME (kagglesdk ignores KAGGLE_CONFIG_DIR)
  * CLI-refreshed OAuth tokens are not clobbered, and are synced back
  * `kernels push -t` (a kernel RUN-TIME limit) is only sent when asked for
  * Watching an already-pushed job never pushes it a second time
  * Browser login is sandboxed and refuses an already-configured account
"""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from kaggle_hub.account_manager import AccountManager
from kaggle_hub.kaggle_runner import KaggleRunner
from kaggle_hub.models import JobStatus, RunOptions
from kaggle_hub.notebook_manager import NotebookManager
from kaggle_hub.storage import StorageEngine


def _oauth(user, refresh="r1", access="a1"):
    return {"refresh_token": refresh, "access_token": access,
            "access_token_expiration": "", "username": user, "scopes": []}


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.storage = StorageEngine(base_dir=self.tmp / "hub")
        self.mgr = AccountManager(storage=self.storage)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestOAuthIsolation(_Base):
    def test_private_home_points_sdk_path_at_account_credentials(self):
        a = self.mgr.add_account("A", "alpha", "oauth_token", _oauth("alpha", "ra"))
        b = self.mgr.add_account("B", "beta", "oauth_token", _oauth("beta", "rb"))
        env_a = self.mgr.get_env_for_account(a.id)
        env_b = self.mgr.get_env_for_account(b.id)

        self.assertNotEqual(env_a["HOME"], env_b["HOME"])
        for acc, env, refresh in ((a, env_a, "ra"), (b, env_b, "rb")):
            # Exactly the path kagglesdk.KaggleCredentials uses.
            sdk_path = Path(env["HOME"]) / ".kaggle" / "credentials.json"
            self.assertEqual(json.loads(sdk_path.read_text())["refresh_token"], refresh)
            self.assertEqual(sdk_path.resolve(), (self.mgr.envs_dir / acc.id / "credentials.json").resolve())

    def test_host_tokens_do_not_leak(self):
        a = self.mgr.add_account("A", "alpha", "oauth_token", _oauth("alpha"))
        k = self.mgr.add_account("K", "kappa", "api_key", {"key": "k"})
        with patch.dict(os.environ, {"KAGGLE_API_TOKEN": "host", "KAGGLE_KEY": "hostkey",
                                     "KAGGLE_USERNAME": "host"}):
            env_a = self.mgr.get_env_for_account(a.id)
            env_k = self.mgr.get_env_for_account(k.id)
        self.assertNotIn("KAGGLE_API_TOKEN", env_a)
        self.assertNotIn("KAGGLE_KEY", env_a)
        self.assertEqual(env_a["KAGGLE_USERNAME"], "alpha")
        self.assertNotIn("KAGGLE_API_TOKEN", env_k)
        self.assertEqual((env_k["KAGGLE_USERNAME"], env_k["KAGGLE_KEY"]), ("kappa", "k"))

    def test_refreshed_access_token_is_kept_and_synced_back(self):
        a = self.mgr.add_account("A", "alpha", "oauth_token", _oauth("alpha", "r1", "old"))
        cred = self.mgr.envs_dir / a.id / "credentials.json"
        cred.write_text(json.dumps(_oauth("alpha", "r1", "fresh")))      # CLI refresh

        self.mgr.get_env_for_account(a.id)                                # must not clobber
        self.assertEqual(json.loads(cred.read_text())["access_token"], "fresh")

        self.mgr.pull_refreshed_credentials(self.mgr.get_account(a.id))
        self.assertEqual(self.mgr.get_account(a.id).credentials["access_token"], "fresh")

    def test_new_login_replaces_file(self):
        a = self.mgr.add_account("A", "alpha", "oauth_token", _oauth("alpha", "r1"))
        self.mgr.update_account(a.id, credentials=_oauth("alpha", "r2"))
        cred = self.mgr.envs_dir / a.id / "credentials.json"
        self.assertEqual(json.loads(cred.read_text())["refresh_token"], "r2")

    @patch("subprocess.run")
    def test_test_account_rejects_wrong_identity(self, run):
        a = self.mgr.add_account("A", "alpha", "oauth_token", _oauth("alpha"))
        run.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),                         # kernels list
            MagicMock(returncode=0, stdout="- username: someone_else\n", stderr=""),  # config view
        ]
        ok, msg = self.mgr.test_account(a.id)
        self.assertFalse(ok)
        self.assertIn("someone_else", msg)


class TestBrowserLogin(_Base):
    def _fake_login(self, username):
        def run(cmd, env=None, timeout=None, **kw):
            self.assertEqual(cmd[:3], ["kaggle", "auth", "login"])
            self.assertNotEqual(env["HOME"], str(Path.home()))          # sandboxed
            self.assertIn(f"HOME={Path.home()}", env["BROWSER"])          # real browser profile
            target = Path(env["HOME"]) / ".kaggle" / "credentials.json"
            target.write_text(json.dumps(_oauth(username, "new")))
            return subprocess.CompletedProcess(cmd, 0)
        return run

    def test_login_adds_account_and_cleans_up(self):
        with patch("subprocess.run", side_effect=self._fake_login("gamma")):
            acc = self.mgr.login_oauth(label="G")
        self.assertEqual(acc.username, "gamma")
        self.assertEqual(acc.auth_type, "oauth_token")
        self.assertFalse(list(self.mgr.envs_dir.glob("pending_*")))

    def test_login_refuses_duplicate_account(self):
        self.mgr.add_account("A", "alpha", "oauth_token", _oauth("alpha"))
        with patch("subprocess.run", side_effect=self._fake_login("alpha")):
            with self.assertRaises(RuntimeError) as ctx:
                self.mgr.login_oauth()
        self.assertIn("already configured", str(ctx.exception))
        self.assertEqual(len(self.mgr.list_accounts()), 1)


class TestRunLifecycle(_Base):
    def setUp(self):
        super().setUp()
        nb_dir = self.tmp / "nbs"
        nb_dir.mkdir()
        (nb_dir / "demo.ipynb").write_text(json.dumps(
            {"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}))
        self.nb = NotebookManager(notebooks_dir=nb_dir, staging_dir=self.tmp / "staging")
        self.runner = KaggleRunner(account_manager=self.mgr, notebook_manager=self.nb,
                                   storage=self.storage)
        self.mgr.add_account("A", "alpha", "api_key", {"key": "k"}, make_active=True)
        self.job = self.runner.submit_job(RunOptions(notebook=str(nb_dir / "demo.ipynb")))

    @patch("subprocess.run")
    def test_push_has_no_runtime_limit_by_default(self, run):
        run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        self.runner.push_job(self.job.id)
        self.assertNotIn("-t", run.call_args[0][0])

    @patch("subprocess.run")
    def test_push_runtime_limit_when_requested(self, run):
        run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        self.runner.push_job(self.job.id, timeout=7200)
        cmd = run.call_args[0][0]
        self.assertEqual(cmd[cmd.index("-t") + 1], "7200")

    @patch("subprocess.run")
    def test_rejected_push_with_exit_code_zero_is_an_error(self, run):
        run.return_value = MagicMock(
            returncode=0, stderr="",
            stdout="Kernel push error: Maximum batch GPU session count of 2 reached.")
        job = self.runner.push_job(self.job.id)
        self.assertEqual(job.status, JobStatus.ERROR)
        self.assertIn("Maximum batch GPU session", job.error_message)

    @patch("subprocess.run")
    def test_failed_status_query_keeps_running_status(self, run):
        run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        self.runner.push_job(self.job.id)
        run.return_value = MagicMock(
            returncode=1, stdout="",
            stderr="HTTPSConnectionPool: Max retries exceeded (NameResolutionError)")
        job = self.runner.check_status(self.job.id)
        self.assertEqual(job.status, JobStatus.RUNNING)
        run.return_value = MagicMock(returncode=0, stderr="",
                                     stdout='x has status "KernelWorkerStatus.ERROR"')
        self.assertEqual(self.runner.check_status(self.job.id).status, JobStatus.ERROR)

    def test_watch_after_push_does_not_push_again(self):
        with patch.object(self.runner, "push_job") as push, \
             patch.object(self.runner, "check_status") as status:
            status.return_value = MagicMock(status=JobStatus.COMPLETE)
            self.runner.run_job_sync(self.job.id, watch=True, auto_download=False,
                                     poll_interval=0, push=False)
            push.assert_not_called()
            self.runner.run_job_async(self.job.id, auto_download=False, poll_interval=0).join(2)
            push.assert_not_called()


if __name__ == "__main__":
    unittest.main()
