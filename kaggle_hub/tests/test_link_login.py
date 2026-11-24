"""
Sign in with a link + verification code (the web panel's multi-account path).

A fake `kaggle` executable on PATH mimics `kaggle auth login --no-launch-browser`:
it prints an authorize URL, reads the code from stdin, and on code "good"
saves credentials to $HOME/.kaggle/credentials.json (as kagglesdk does).
"""

import json
import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kaggle_hub.account_manager import AccountManager
from kaggle_hub.server.api_handlers import ApiDispatcher
from kaggle_hub.storage import StorageEngine

FAKE_KAGGLE = r'''#!/usr/bin/env python3
import json, os, sys
assert sys.argv[1:] == ["auth", "login", "--no-launch-browser"], sys.argv
print("\nGo to the following link in your browser:\n")
print("  https://www.kaggle.com/api/v1/oauth2/authorize?response_type=code&state=s")
print("\nOnce finished, enter the verification code provided in your browser: ", end="")
code = input().strip()
if code != "good":
    print("Invalid verification code")
    sys.exit(1)
user = os.environ.get("FAKE_KAGGLE_USER", "gamma")
path = os.path.join(os.environ["HOME"], ".kaggle", "credentials.json")
with open(path, "w") as f:
    json.dump({"refresh_token": "r", "access_token": "a", "access_token_expiration": "",
               "username": user, "scopes": []}, f)
print(f"You are now logged in as [{user}]")
'''


class TestLinkLogin(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        bin_dir = self.tmp / "bin"
        bin_dir.mkdir()
        fake = bin_dir / "kaggle"
        fake.write_text(FAKE_KAGGLE)
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
        self.env = patch.dict(os.environ, {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"})
        self.env.start()
        self.mgr = AccountManager(storage=StorageEngine(base_dir=self.tmp / "hub"))

    def tearDown(self):
        self.env.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _pending(self):
        return list(self.mgr.envs_dir.glob("pending_*"))

    def test_start_returns_link_and_finish_adds_account(self):
        d = self.mgr.start_link_login(label="G", wait=10)
        self.assertIn("oauth2/authorize", d["url"])
        self.assertTrue(d["url"].startswith("https://"))
        acc = self.mgr.finish_link_login(d["session"], "good")
        self.assertEqual((acc.username, acc.label, acc.auth_type), ("gamma", "G", "oauth_token"))
        self.assertEqual(self._pending(), [])
        # the session is consumed
        with self.assertRaises(RuntimeError):
            self.mgr.finish_link_login(d["session"], "good")

    def test_wrong_code_is_reported_and_cleaned_up(self):
        d = self.mgr.start_link_login(wait=10)
        with self.assertRaises(RuntimeError) as ctx:
            self.mgr.finish_link_login(d["session"], "bad")
        self.assertIn("rejected", str(ctx.exception))
        self.assertEqual(self.mgr.list_accounts(), [])
        self.assertEqual(self._pending(), [])

    def test_duplicate_account_is_refused_with_private_window_hint(self):
        self.mgr.add_account("G", "gamma", "oauth_token",
                             {"refresh_token": "x", "username": "gamma"})
        d = self.mgr.start_link_login(wait=10)
        with self.assertRaises(RuntimeError) as ctx:
            self.mgr.finish_link_login(d["session"], "good")
        self.assertIn("already configured", str(ctx.exception))
        self.assertIn("private window", str(ctx.exception))
        self.assertEqual(len(self.mgr.list_accounts()), 1)

    def test_cancel_kills_process_and_removes_sandbox(self):
        d = self.mgr.start_link_login(wait=10)
        proc = self.mgr._link_logins[d["session"]]["proc"]
        self.assertTrue(self.mgr.cancel_link_login(d["session"]))
        proc.wait(timeout=5)
        self.assertEqual(self._pending(), [])
        self.assertFalse(self.mgr.cancel_link_login(d["session"]))

    def test_api_routes(self):
        api = ApiDispatcher(account_manager=self.mgr, storage=self.mgr.storage)
        with patch.object(self.mgr, "test_account", return_value=(True, "ok")):
            code, d = api._handle_accounts("POST", ["login", "start"], {}, {"label": "G"})
            self.assertEqual(code, 200)
            code, d = api._handle_accounts("POST", ["login", "finish"], {},
                                           {"session": d["session"], "code": "good"})
        self.assertEqual(code, 201)
        self.assertEqual(d["account"]["username"], "gamma")
        self.assertFalse(d["account"].get("credentials"))      # tokens never reach the browser
        code, d = api._handle_accounts("POST", ["login", "finish"], {}, {"session": "nope", "code": "x"})
        self.assertEqual(code, 409)


if __name__ == "__main__":
    unittest.main()
