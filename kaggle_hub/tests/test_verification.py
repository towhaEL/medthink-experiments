"""
Independent verification test suite for Kaggle Hub core backend.
Validates:
1. Security permissions (0700 on dirs, 0600 on sensitive credentials & storage files).
2. Real notebook discovery against actual notebooks directory.
3. Staging directory creation and kernel-metadata.json validity for 00_data_prep.ipynb and 02_train_closed_end.ipynb.
4. Environment isolation and get_process_env behavior with zero leakage between accounts.
5. Account switching and symlink integrity across full CRUD lifecycle.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path

from kaggle_hub import config
from kaggle_hub.account_manager import AccountManager
from kaggle_hub.models import Account, RunOptions
from kaggle_hub.notebook_manager import NotebookManager
from kaggle_hub.storage import StorageEngine


class TestSecurityPermissions(unittest.TestCase):
    """Verify strict UNIX permissions (0700 on dirs, 0600 on credential files)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="kh_perm_test_")
        self.base_dir = Path(self.tmpdir)
        self.storage = StorageEngine(base_dir=self.base_dir)
        self.mgr = AccountManager(storage=self.storage)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _get_mode(self, path: Path) -> int:
        return os.stat(path).st_mode & 0o777

    def test_directory_permissions_0700(self):
        # Base dir must be 0700
        self.assertEqual(self._get_mode(self.base_dir), 0o700)
        # envs dir must be 0700
        self.assertEqual(self._get_mode(self.mgr.envs_dir), 0o700)

    def test_account_credentials_permissions_0600(self):
        # API key account
        acc_api = self.mgr.add_account(
            label="API Key Account",
            username="sec_user_api",
            auth_type="api_key",
            credentials={"key": "super_secret_api_key"},
        )
        api_dir = self.mgr.envs_dir / acc_api.id
        api_file = api_dir / "kaggle.json"

        self.assertEqual(self._get_mode(api_dir), 0o700)
        self.assertEqual(self._get_mode(api_file), 0o600)

        # OAuth token account
        acc_oauth = self.mgr.add_account(
            label="OAuth Account",
            username="sec_user_oauth",
            auth_type="oauth_token",
            credentials={"access_token": "oauth_tok_xyz", "refresh_token": "refresh_xyz"},
        )
        oauth_dir = self.mgr.envs_dir / acc_oauth.id
        oauth_file = oauth_dir / "credentials.json"

        self.assertEqual(self._get_mode(oauth_dir), 0o700)
        self.assertEqual(self._get_mode(oauth_file), 0o600)

    def test_storage_files_permissions_0600(self):
        self.storage.save_accounts({"active_account_id": None, "accounts": []})
        self.assertTrue(self.storage.accounts_file.exists())
        self.assertEqual(self._get_mode(self.storage.accounts_file), 0o600)

        self.storage.save_jobs({"jobs": []})
        self.assertTrue(self.storage.jobs_file.exists())
        self.assertEqual(self._get_mode(self.storage.jobs_file), 0o600)


class TestRealNotebookDiscovery(unittest.TestCase):
    """Verify discovery, cell parsing, slug derivation against real repo notebooks."""

    def setUp(self):
        self.nb_dir = config.DEFAULT_NOTEBOOKS_DIR
        self.nb_mgr = NotebookManager(notebooks_dir=self.nb_dir)

    def test_real_notebooks_discovered(self):
        notebooks = self.nb_mgr.list_notebooks()
        names = [nb.name for nb in notebooks]

        expected = [
            "00_data_prep.ipynb",
            "01_extract_detr_features.ipynb",
            "02_train_closed_end.ipynb",
            "03_train_open_end.ipynb",
            "04_two_stage_reasoning.ipynb",
            "05_results_report.ipynb",
        ]

        for exp in expected:
            self.assertIn(exp, names, f"Expected notebook {exp} not found in {names}")

        self.assertEqual(len(notebooks), 6)

    def test_notebook_cell_counts_and_types(self):
        notebooks = {nb.name: nb for nb in self.nb_mgr.list_notebooks()}

        for name, nb in notebooks.items():
            self.assertGreater(nb.total_cells, 0, f"{name} has 0 total cells")
            self.assertGreater(nb.code_cells, 0, f"{name} has 0 code cells")
            self.assertGreater(nb.markdown_cells, 0, f"{name} has 0 markdown cells")
            self.assertEqual(
                nb.total_cells,
                nb.code_cells + nb.markdown_cells,
                f"Mismatch in cell counts for {name}",
            )
            self.assertGreater(nb.file_size_bytes, 1000, f"{name} file size abnormally small")

    def test_slug_and_title_spec(self):
        nb00, _ = self.nb_mgr.inspect_notebook("00_data_prep.ipynb")
        self.assertEqual(nb00.slug, "medthink-00-data-prep")
        self.assertIn("Data Preparation", nb00.title)

        nb02, _ = self.nb_mgr.inspect_notebook("02_train_closed_end.ipynb")
        self.assertEqual(nb02.slug, "medthink-02-train-closed-end")
        self.assertIn("Closed-End", nb02.title)


class TestStagingAndMetadataValidity(unittest.TestCase):
    """Verify staging directory creation and strict kernel-metadata.json schema."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="kh_stage_test_")
        self.staging_dir = Path(self.tmpdir) / "staging"
        self.nb_mgr = NotebookManager(
            notebooks_dir=config.DEFAULT_NOTEBOOKS_DIR,
            staging_dir=self.staging_dir,
        )
        self.test_account = Account(
            id="acc_verifier",
            label="Verifier Account",
            username="verifier_author",
            auth_type="api_key",
            credentials={"key": "ver_key_123"},
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_staging_00_data_prep(self):
        opt = RunOptions(
            notebook="00_data_prep.ipynb",
            account_id=self.test_account.id,
            accelerator="NvidiaTeslaT4",
            enable_gpu=True,
            enable_internet=True,
        )

        staging_path = self.nb_mgr.prepare_staging(
            job_id="test_job_00",
            notebook_path="00_data_prep.ipynb",
            options=opt,
            account=self.test_account,
        )

        self.assertTrue(staging_path.is_dir())
        meta_file = staging_path / "kernel-metadata.json"
        code_file = staging_path / "00_data_prep.ipynb"

        self.assertTrue(meta_file.exists())
        self.assertTrue(code_file.exists())

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # Validate Kaggle kernel-metadata.json fields
        self.assertEqual(meta["id"], "verifier_author/medthink-00-data-prep")
        self.assertEqual(meta["code_file"], "00_data_prep.ipynb")
        self.assertEqual(meta["language"], "python")
        self.assertEqual(meta["kernel_type"], "notebook")
        self.assertEqual(meta["is_private"], "true")
        self.assertEqual(meta["enable_gpu"], "true")
        self.assertEqual(meta["enable_tpu"], "false")
        self.assertEqual(meta["enable_internet"], "true")
        self.assertIsInstance(meta["dataset_sources"], list)
        self.assertIsInstance(meta["competition_sources"], list)
        self.assertIsInstance(meta["kernel_sources"], list)
        self.assertIsInstance(meta["model_sources"], list)

        # Verify staged notebook is valid json and matches original cell count
        with open(code_file, "r", encoding="utf-8") as f:
            staged_nb = json.load(f)
        self.assertIn("cells", staged_nb)
        self.assertEqual(len(staged_nb["cells"]), 20)

    def test_staging_02_train_closed_end_with_sources(self):
        opt = RunOptions(
            notebook="02_train_closed_end.ipynb",
            account_id=self.test_account.id,
            accelerator="NvidiaTeslaP100",
            enable_gpu=True,
            enable_internet=False,
            dataset_sources=["towha/medthink-preprocessed-vqa"],
            competition_sources=["med-vqa-challenge"],
            kernel_sources=["verifier_author/medthink-00-data-prep"],
        )

        staging_path = self.nb_mgr.prepare_staging(
            job_id="test_job_02",
            notebook_path="02_train_closed_end.ipynb",
            options=opt,
            account=self.test_account,
        )

        meta_file = staging_path / "kernel-metadata.json"
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.assertEqual(meta["id"], "verifier_author/medthink-02-train-closed-end")
        self.assertEqual(meta["code_file"], "02_train_closed_end.ipynb")
        self.assertEqual(meta["enable_gpu"], "true")
        self.assertEqual(meta["enable_internet"], "false")
        self.assertEqual(meta["dataset_sources"], ["towha/medthink-preprocessed-vqa"])
        self.assertEqual(meta["competition_sources"], ["med-vqa-challenge"])
        self.assertEqual(meta["kernel_sources"], ["verifier_author/medthink-00-data-prep"])


class TestEnvironmentIsolation(unittest.TestCase):
    """Verify get_process_env sets KAGGLE_CONFIG_DIR per-account with no cross-leakage."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="kh_env_test_")
        self.base_dir = Path(self.tmpdir)
        self.storage = StorageEngine(base_dir=self.base_dir)
        self.mgr = AccountManager(storage=self.storage)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_multi_account_environment_isolation(self):
        acc_a = self.mgr.add_account(
            label="Account Alpha",
            username="author_alpha",
            auth_type="api_key",
            credentials={"key": "key_alpha_secret_111"},
            make_active=True,
        )
        acc_b = self.mgr.add_account(
            label="Account Beta",
            username="author_beta",
            auth_type="oauth_token",
            credentials={"access_token": "tok_beta_oauth_222", "refresh_token": "ref_beta_333"},
            make_active=False,
        )

        # Call via both get_env_for_account and alias get_process_env
        env_a = self.mgr.get_process_env(acc_a.id)
        env_b = self.mgr.get_process_env(acc_b.id)

        # 1. Config dirs must point strictly to isolated account directories
        expected_dir_a = str(self.mgr.envs_dir / acc_a.id)
        expected_dir_b = str(self.mgr.envs_dir / acc_b.id)

        self.assertEqual(env_a["KAGGLE_CONFIG_DIR"], expected_dir_a)
        self.assertEqual(env_b["KAGGLE_CONFIG_DIR"], expected_dir_b)
        self.assertNotEqual(env_a["KAGGLE_CONFIG_DIR"], env_b["KAGGLE_CONFIG_DIR"])

        # 2. Credential variables in env_a
        self.assertEqual(env_a["KAGGLE_USERNAME"], "author_alpha")
        self.assertEqual(env_a["KAGGLE_KEY"], "key_alpha_secret_111")

        # 3. Credential variables in env_b (OAuth)
        self.assertEqual(env_b["KAGGLE_USERNAME"], "author_beta")
        # KAGGLE_KEY must NOT be present in env_b (prevent leakage from Alpha or host)
        self.assertNotIn("KAGGLE_KEY", env_b)

        # 4. Verify credentials on disk
        cred_file_a = Path(env_a["KAGGLE_CONFIG_DIR"]) / "kaggle.json"
        cred_file_b = Path(env_b["KAGGLE_CONFIG_DIR"]) / "credentials.json"
        self.assertTrue(cred_file_a.exists())
        self.assertTrue(cred_file_b.exists())

        with open(cred_file_a, "r") as f:
            data_a = json.load(f)
            self.assertEqual(data_a["username"], "author_alpha")
            self.assertEqual(data_a["key"], "key_alpha_secret_111")

        with open(cred_file_b, "r") as f:
            data_b = json.load(f)
            self.assertEqual(data_b["username"], "author_beta")
            self.assertEqual(data_b["access_token"], "tok_beta_oauth_222")

        # 5. Check PATH contains local bin
        self.assertTrue(any(".local/bin" in p for p in env_a["PATH"].split(os.pathsep)))
        self.assertTrue(any(".local/bin" in p for p in env_b["PATH"].split(os.pathsep)))

    def test_ambient_env_var_clearing(self):
        # Set ambient KAGGLE_KEY in os.environ to test that OAuth account suppresses it
        orig_key = os.environ.get("KAGGLE_KEY")
        try:
            os.environ["KAGGLE_KEY"] = "host_ambient_leak_key"
            acc_oauth = self.mgr.add_account(
                label="OAuth User",
                username="oauth_user_clean",
                auth_type="oauth_token",
                credentials={"access_token": "token123"},
            )
            env_oauth = self.mgr.get_process_env(acc_oauth.id)
            self.assertNotIn("KAGGLE_KEY", env_oauth, "Ambient KAGGLE_KEY leaked into OAuth environment!")
        finally:
            if orig_key is not None:
                os.environ["KAGGLE_KEY"] = orig_key
            else:
                os.environ.pop("KAGGLE_KEY", None)


class TestAccountSwitchingAndSymlinkIntegrity(unittest.TestCase):
    """Verify switching active accounts updates active symlink and survives deletes."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="kh_sym_test_")
        self.base_dir = Path(self.tmpdir)
        self.storage = StorageEngine(base_dir=self.base_dir)
        self.mgr = AccountManager(storage=self.storage)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_symlink_lifecycle_and_switching(self):
        # 1. Add first account -> should auto become active
        acc1 = self.mgr.add_account(label="Acc 1", username="user1", credentials={"key": "k1"})
        self.assertEqual(self.mgr.get_active_account_id(), acc1.id)
        self.assertTrue(self.mgr.active_symlink.is_symlink())
        self.assertEqual(
            self.mgr.active_symlink.resolve(),
            (self.mgr.envs_dir / acc1.id).resolve(),
        )

        # 2. Add second account without make_active -> active stays acc1
        acc2 = self.mgr.add_account(label="Acc 2", username="user2", credentials={"key": "k2"})
        self.assertEqual(self.mgr.get_active_account_id(), acc1.id)
        self.assertEqual(
            self.mgr.active_symlink.resolve(),
            (self.mgr.envs_dir / acc1.id).resolve(),
        )

        # 3. Explicitly switch to acc2
        self.mgr.set_active_account(acc2.id)
        self.assertEqual(self.mgr.get_active_account_id(), acc2.id)
        self.assertEqual(
            self.mgr.active_symlink.resolve(),
            (self.mgr.envs_dir / acc2.id).resolve(),
        )

        # 4. Add third account and switch to it
        acc3 = self.mgr.add_account(label="Acc 3", username="user3", credentials={"key": "k3"}, make_active=True)
        self.assertEqual(self.mgr.get_active_account_id(), acc3.id)
        self.assertEqual(
            self.mgr.active_symlink.resolve(),
            (self.mgr.envs_dir / acc3.id).resolve(),
        )

        # 5. Delete active account (acc3) -> should fallback to remaining account
        deleted = self.mgr.delete_account(acc3.id)
        self.assertTrue(deleted)
        new_active = self.mgr.get_active_account_id()
        self.assertIn(new_active, [acc1.id, acc2.id])
        self.assertEqual(
            self.mgr.active_symlink.resolve(),
            (self.mgr.envs_dir / new_active).resolve(),
        )

        # 6. Delete remaining accounts until none left
        self.mgr.delete_account(acc1.id)
        self.mgr.delete_account(acc2.id)
        self.assertIsNone(self.mgr.get_active_account_id())
        self.assertFalse(self.mgr.active_symlink.exists())
        self.assertFalse(self.mgr.active_symlink.is_symlink())


if __name__ == "__main__":
    unittest.main()
