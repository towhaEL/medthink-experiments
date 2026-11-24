"""
Unit tests for core Kaggle Hub modules:
- config
- models
- storage
- account_manager
- notebook_manager
- kaggle_runner
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from kaggle_hub import config
from kaggle_hub.account_manager import AccountManager
from kaggle_hub.kaggle_runner import KaggleRunner
from kaggle_hub.models import (
    Account,
    JobRecord,
    JobStatus,
    NotebookInfo,
    RunOptions,
)
from kaggle_hub.notebook_manager import NotebookManager
from kaggle_hub.storage import StorageEngine


class TestConfig(unittest.TestCase):
    def test_accelerator_normalization(self):
        self.assertEqual(config.normalize_accelerator("t4"), "NvidiaTeslaT4")
        self.assertEqual(config.normalize_accelerator("T4"), "NvidiaTeslaT4")
        self.assertEqual(config.normalize_accelerator("t4x2"), "NvidiaTeslaT4")
        self.assertEqual(config.normalize_accelerator("p100"), "NvidiaTeslaP100")
        self.assertEqual(config.normalize_accelerator("tpu"), "TpuV5E8")
        self.assertEqual(config.normalize_accelerator("cpu"), "None")
        self.assertEqual(config.normalize_accelerator("None"), "None")
        self.assertEqual(config.normalize_accelerator(None), "NvidiaTeslaT4")

    def test_ensure_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config.ensure_dirs(tmp_path)
            self.assertTrue((tmp_path / "envs").is_dir())
            self.assertTrue((tmp_path / "staging").is_dir())


class TestModels(unittest.TestCase):
    def test_account_roundtrip(self):
        acc = Account(
            id="acc_test_1",
            label="Test User",
            username="testuser",
            auth_type="api_key",
            credentials={"key": "secret123"},
        )
        d = acc.to_dict()
        reconstructed = Account.from_dict(d)
        self.assertEqual(reconstructed.id, "acc_test_1")
        self.assertEqual(reconstructed.username, "testuser")
        self.assertEqual(reconstructed.credentials.get("key"), "secret123")

    def test_notebook_info_roundtrip(self):
        nb = NotebookInfo(
            name="00_test.ipynb",
            path="/tmp/00_test.ipynb",
            slug="medthink-00-test",
            title="MedThink 00 Test",
            total_cells=10,
            code_cells=8,
            markdown_cells=2,
            file_size_bytes=1024,
        )
        d = nb.to_dict()
        reconstructed = NotebookInfo.from_dict(d)
        self.assertEqual(reconstructed.slug, "medthink-00-test")
        self.assertEqual(reconstructed.total_cells, 10)

    def test_job_record_roundtrip(self):
        job = JobRecord(
            id="job_001",
            account_id="acc_test_1",
            username="testuser",
            kernel_slug="medthink-00-test",
            kernel_id="testuser/medthink-00-test",
            title="MedThink Test",
            notebook_path="/tmp/00_test.ipynb",
            staging_path="/tmp/staging",
            status=JobStatus.RUNNING,
        )
        d = job.to_dict()
        reconstructed = JobRecord.from_dict(d)
        self.assertEqual(reconstructed.status, JobStatus.RUNNING)
        self.assertEqual(reconstructed.kernel_id, "testuser/medthink-00-test")


class TestStorage(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.storage = StorageEngine(base_dir=Path(self.tmpdir))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_permissions_and_atomic_write(self):
        # Base dir permissions should be 0700
        stat_mode = os.stat(self.tmpdir).st_mode & 0o777
        self.assertEqual(stat_mode, 0o700)

        # Save accounts and check file permissions
        data = {"active_account_id": "acc_1", "accounts": [{"id": "acc_1"}]}
        self.storage.save_accounts(data)
        self.assertTrue(self.storage.accounts_file.exists())
        file_mode = os.stat(self.storage.accounts_file).st_mode & 0o777
        self.assertEqual(file_mode, 0o600)

        loaded = self.storage.load_accounts()
        self.assertEqual(loaded.get("active_account_id"), "acc_1")

    def test_jobs_persistence(self):
        jobs_data = {"jobs": [{"id": "job_1", "status": "QUEUED"}]}
        self.storage.save_jobs(jobs_data)
        loaded = self.storage.load_jobs()
        self.assertEqual(len(loaded.get("jobs", [])), 1)
        self.assertEqual(loaded["jobs"][0]["id"], "job_1")


class TestAccountManager(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.storage = StorageEngine(base_dir=Path(self.tmpdir))
        self.mgr = AccountManager(storage=self.storage)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_add_and_get_account_api_key(self):
        acc = self.mgr.add_account(
            label="Primary API",
            username="researcher1",
            auth_type="api_key",
            credentials={"key": "kaggle_key_xyz"},
            make_active=True,
        )
        self.assertEqual(acc.username, "researcher1")
        self.assertEqual(self.mgr.get_active_account_id(), acc.id)

        # Verify isolated credentials file created
        cred_file = self.mgr.envs_dir / acc.id / "kaggle.json"
        self.assertTrue(cred_file.exists())
        file_mode = os.stat(cred_file).st_mode & 0o777
        self.assertEqual(file_mode, 0o600)

        with open(cred_file, "r") as f:
            saved = json.load(f)
        self.assertEqual(saved.get("key"), "kaggle_key_xyz")

    def test_add_and_get_account_oauth(self):
        acc = self.mgr.add_account(
            label="OAuth User",
            username="oauth_user",
            auth_type="oauth_token",
            credentials={"access_token": "token123", "refresh_token": "ref456"},
        )
        cred_file = self.mgr.envs_dir / acc.id / "credentials.json"
        self.assertTrue(cred_file.exists())
        with open(cred_file, "r") as f:
            saved = json.load(f)
        self.assertEqual(saved.get("access_token"), "token123")

    def test_switch_active_symlink(self):
        acc1 = self.mgr.add_account(
            label="User 1",
            username="user1",
            credentials={"key": "k1"},
            make_active=True,
        )
        acc2 = self.mgr.add_account(
            label="User 2",
            username="user2",
            credentials={"key": "k2"},
            make_active=False,
        )
        self.assertEqual(self.mgr.get_active_account_id(), acc1.id)
        self.assertTrue(self.mgr.active_symlink.is_symlink())
        self.assertEqual(self.mgr.active_symlink.resolve(), (self.mgr.envs_dir / acc1.id).resolve())

        # Switch to acc2
        self.mgr.set_active_account(acc2.id)
        self.assertEqual(self.mgr.get_active_account_id(), acc2.id)
        self.assertEqual(self.mgr.active_symlink.resolve(), (self.mgr.envs_dir / acc2.id).resolve())

    def test_get_env_for_account(self):
        acc = self.mgr.add_account(
            label="User",
            username="envuser",
            auth_type="api_key",
            credentials={"key": "secret_key"},
            make_active=True,
        )
        env = self.mgr.get_env_for_account(acc.id)
        self.assertIn("KAGGLE_CONFIG_DIR", env)
        self.assertEqual(env["KAGGLE_CONFIG_DIR"], str(self.mgr.envs_dir / acc.id))
        self.assertEqual(env.get("KAGGLE_USERNAME"), "envuser")
        self.assertEqual(env.get("KAGGLE_KEY"), "secret_key")

    @patch("subprocess.run")
    def test_test_account_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="ref title", stderr="")
        acc = self.mgr.add_account(
            label="Valid User",
            username="validuser",
            credentials={"key": "abc"},
            make_active=True,
        )
        valid, msg = self.mgr.test_account(acc.id)
        self.assertTrue(valid)
        self.assertIn("successful", msg)
        reloaded = self.mgr.get_account(acc.id)
        self.assertEqual(reloaded.status, "valid")
        self.assertIsNotNone(reloaded.last_used_at)

    def test_delete_account(self):
        acc = self.mgr.add_account(label="Del", username="deluser", make_active=True)
        acc_dir = self.mgr.envs_dir / acc.id
        self.assertTrue(acc_dir.exists())

        deleted = self.mgr.delete_account(acc.id)
        self.assertTrue(deleted)
        self.assertIsNone(self.mgr.get_account(acc.id))
        self.assertFalse(acc_dir.exists())

    def test_auto_import_credentials(self):
        home_kaggle = Path(self.tmpdir) / ".kaggle"
        home_kaggle.mkdir(mode=0o700, parents=True)
        oauth_data = {
            "username": "imported_oauth_user",
            "access_token": "token_abc",
            "refresh_token": "refresh_def",
        }
        with open(home_kaggle / "credentials.json", "w") as f:
            json.dump(oauth_data, f)

        with patch("pathlib.Path.home", return_value=Path(self.tmpdir)):
            imported = self.mgr.import_existing_credentials()
            self.assertEqual(len(imported), 1)
            self.assertEqual(imported[0].username, "imported_oauth_user")
            self.assertEqual(imported[0].auth_type, "oauth_token")
            # Should now be active
            self.assertEqual(self.mgr.get_active_account_id(), imported[0].id)


class TestNotebookManager(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.staging_dir = Path(self.tmpdir) / "staging"
        self.nb_mgr = NotebookManager(
            notebooks_dir=config.DEFAULT_NOTEBOOKS_DIR,
            staging_dir=self.staging_dir,
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_slug_derivation(self):
        self.assertEqual(
            self.nb_mgr.derive_slug("00_data_prep.ipynb"),
            "medthink-00-data-prep",
        )
        self.assertEqual(
            self.nb_mgr.derive_slug("my_custom_model.ipynb"),
            "my-custom-model",
        )

    def test_list_and_inspect_actual_notebooks(self):
        notebooks = self.nb_mgr.list_notebooks()
        self.assertGreaterEqual(len(notebooks), 1)

        first = notebooks[0]
        self.assertTrue(first.name.endswith(".ipynb"))
        self.assertGreater(first.total_cells, 0)
        self.assertTrue(len(first.slug) >= 5)

    def test_generate_metadata_and_staging(self):
        acc = Account(id="acc_1", label="User", username="towha")
        opt = RunOptions(
            notebook="00_data_prep.ipynb",
            enable_gpu=True,
            enable_internet=True,
            accelerator="NvidiaTeslaT4",
        )
        # Find 00_data_prep.ipynb in DEFAULT_NOTEBOOKS_DIR
        nb_path = config.DEFAULT_NOTEBOOKS_DIR / "00_data_prep.ipynb"
        if not nb_path.exists():
            # Create a mock notebook for test
            nb_path = Path(self.tmpdir) / "mock.ipynb"
            with open(nb_path, "w") as f:
                json.dump({"cells": [{"cell_type": "markdown", "source": ["# Mock Title\n"]}]}, f)

        meta = self.nb_mgr.generate_kernel_metadata(opt, acc, nb_path)
        self.assertEqual(meta["language"], "python")
        self.assertEqual(meta["kernel_type"], "notebook")
        self.assertEqual(meta["enable_gpu"], "true")
        self.assertEqual(meta["enable_internet"], "true")

        staging_path = self.nb_mgr.prepare_staging("test_job_123", nb_path, opt, acc)
        self.assertTrue(staging_path.is_dir())
        self.assertTrue((staging_path / "kernel-metadata.json").exists())
        self.assertTrue((staging_path / nb_path.name).exists())


class TestKaggleRunner(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.storage = StorageEngine(base_dir=Path(self.tmpdir))
        self.acc_mgr = AccountManager(storage=self.storage)
        self.nb_mgr = NotebookManager(
            notebooks_dir=config.DEFAULT_NOTEBOOKS_DIR,
            staging_dir=Path(self.tmpdir) / "staging",
        )
        self.runner = KaggleRunner(
            account_manager=self.acc_mgr,
            notebook_manager=self.nb_mgr,
            storage=self.storage,
        )
        self.account = self.acc_mgr.add_account(
            label="Runner Test",
            username="towha_test",
            credentials={"key": "key_xyz"},
            make_active=True,
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("subprocess.run")
    def test_job_submission_and_push(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Kernel version 1 successfully pushed",
            stderr="",
        )

        opt = RunOptions(
            notebook="00_data_prep.ipynb",
            account_id=self.account.id,
            accelerator="NvidiaTeslaT4",
        )
        job = self.runner.submit_job(opt)
        self.assertEqual(job.status, JobStatus.QUEUED)
        self.assertTrue(Path(job.staging_path).exists())

        # Push job
        pushed = self.runner.push_job(job.id)
        self.assertEqual(pushed.status, JobStatus.RUNNING)
        self.assertIsNotNone(pushed.started_at)

    @patch("subprocess.run")
    def test_check_status_parsing(self, mock_run):
        opt = RunOptions(notebook="00_data_prep.ipynb", account_id=self.account.id)
        job = self.runner.submit_job(opt)

        # Mock status "Your kernel has completed"
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='Your kernel "towha_test/medthink-00-data-prep" has completed',
            stderr="",
        )
        updated = self.runner.check_status(job.id)
        self.assertEqual(updated.status, JobStatus.COMPLETE)
        self.assertIsNotNone(updated.completed_at)

    @patch("subprocess.run")
    def test_get_logs(self, mock_run):
        opt = RunOptions(notebook="00_data_prep.ipynb", account_id=self.account.id)
        job = self.runner.submit_job(opt)

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Epoch 1/10\nloss: 0.42\nDone training.",
            stderr="",
        )
        logs = self.runner.get_logs(job.id)
        self.assertIn("Done training", logs)
        reloaded = self.runner.get_job(job.id)
        self.assertIn("Done training", reloaded.last_log_snippet)

    @patch("subprocess.run")
    def test_download_outputs(self, mock_run):
        opt = RunOptions(notebook="00_data_prep.ipynb", account_id=self.account.id)
        job = self.runner.submit_job(opt)
        target_dir = Path(self.tmpdir) / "custom_outputs"

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Output URL: https://...",
            stderr="",
        )
        out_dir = self.runner.download_outputs(job.id, output_dir=target_dir)
        self.assertEqual(out_dir, str(target_dir))
        reloaded = self.runner.get_job(job.id)
        self.assertEqual(reloaded.output_dir, str(target_dir))
        self.assertEqual(reloaded.status, JobStatus.DOWNLOADED)


    def test_corrupted_storage_recovery(self):
        # Corrupt storage file with non-JSON text
        with open(self.storage.accounts_file, "w") as f:
            f.write("GARBAGE DATA {{{")

        # Storage should safely recover with default structure
        loaded = self.storage.load_accounts()
        self.assertIn("accounts", loaded)
        self.assertEqual(loaded["accounts"], [])

    def test_runner_async_execution(self):
        with patch.object(self.runner, "run_job_sync") as mock_sync:
            thread = self.runner.run_job_async("mock_job_id", auto_download=False)
            thread.join(timeout=2.0)
            mock_sync.assert_called_once()


if __name__ == "__main__":
    unittest.main()
