"""
Multi-account management and credential sandboxing for Kaggle Hub.
Zero external dependencies.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from . import config
from .models import Account, utc_now_iso
from .storage import StorageEngine


class AccountManager:
    """Manages multi-account credentials, isolated execution environments, and symlinking."""

    def __init__(
        self,
        storage: StorageEngine | None = None,
        base_dir: Path | None = None,
    ) -> None:
        self.storage = storage or StorageEngine(base_dir=base_dir)
        self.base_dir = self.storage.base_dir
        self.envs_dir = self.base_dir / "envs"
        self.active_symlink = self.base_dir / "active"
        self._link_logins: dict[str, dict] = {}     # session id -> pending link sign-in
        self._ensure_setup()

    def _ensure_setup(self) -> None:
        """Initialize directory layout."""
        self.envs_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            os.chmod(self.envs_dir, 0o700)
        except OSError:
            pass

    def _sync_env_credentials(self, account: Account) -> Path:
        """Write isolated Kaggle credentials into ~/.kaggle_hub/envs/<acc_id>/."""
        acc_env_dir = self.envs_dir / account.id
        acc_env_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            os.chmod(acc_env_dir, 0o700)
        except OSError:
            pass

        if account.auth_type == "oauth_token":
            cred_file = acc_env_dir / "credentials.json"
            cred_data = dict(account.credentials)
            if "username" not in cred_data:
                cred_data["username"] = account.username
            # The Kaggle CLI refreshes the access token in place, so the file
            # can be newer than the stored copy. Only overwrite it when the
            # refresh token itself changed (a new login) or it is missing.
            current = self._read_json(cred_file)
            if current.get("refresh_token") != cred_data.get("refresh_token"):
                with open(cred_file, "w", encoding="utf-8") as f:
                    json.dump(cred_data, f, indent=2)
                try:
                    os.chmod(cred_file, 0o600)
                except OSError:
                    pass
        else:
            # Default to api_key (kaggle.json)
            cred_file = acc_env_dir / "kaggle.json"
            key = account.credentials.get("key", "")
            with open(cred_file, "w", encoding="utf-8") as f:
                json.dump({"username": account.username, "key": key}, f, indent=2)
            try:
                os.chmod(cred_file, 0o600)
            except OSError:
                pass

        self._ensure_private_home(acc_env_dir)
        return acc_env_dir

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _ensure_private_home(acc_env_dir: Path) -> Path:
        """
        Give the account its own HOME whose ~/.kaggle points at its env dir.

        The Kaggle SDK (kagglesdk 0.1.x) loads and saves OAuth credentials at
        the hard-coded path ~/.kaggle/credentials.json and ignores
        KAGGLE_CONFIG_DIR. Without a per-account HOME, every OAuth account
        silently runs as whoever is logged in to the real ~/.kaggle.
        """
        home = acc_env_dir / "home"
        home.mkdir(mode=0o700, parents=True, exist_ok=True)
        link = home / ".kaggle"
        if link.is_symlink() and Path(os.readlink(link)) == acc_env_dir:
            return home
        if link.is_symlink() or link.is_file():
            link.unlink()
        elif link.is_dir():
            shutil.rmtree(link)
        os.symlink(acc_env_dir, link)
        return home

    def pull_refreshed_credentials(self, account: Account) -> None:
        """Copy CLI-refreshed OAuth tokens from the env dir back into storage."""
        if account.auth_type != "oauth_token":
            return
        on_disk = self._read_json(self.envs_dir / account.id / "credentials.json")
        if on_disk.get("refresh_token") and on_disk != account.credentials:
            doc = self.storage.load_accounts()
            for a in doc.get("accounts", []):
                if a.get("id") == account.id:
                    a["credentials"] = on_disk
                    break
            self.storage.save_accounts(doc)

    def _update_active_symlink(self, account_id: str | None) -> None:
        """Atomically update the active symlink to point to the active account env."""
        if not account_id:
            if self.active_symlink.is_symlink() or self.active_symlink.exists():
                try:
                    self.active_symlink.unlink()
                except OSError:
                    pass
            return

        target_dir = self.envs_dir / account_id
        if not target_dir.exists():
            return

        tmp_symlink = self.base_dir / f".active_tmp_{uuid.uuid4().hex[:6]}"
        try:
            os.symlink(target_dir, tmp_symlink)
            os.replace(tmp_symlink, self.active_symlink)
        except OSError:
            if tmp_symlink.exists():
                try:
                    tmp_symlink.unlink()
                except OSError:
                    pass
            # Fallback to direct symlink
            if self.active_symlink.is_symlink() or self.active_symlink.exists():
                try:
                    self.active_symlink.unlink()
                except OSError:
                    pass
            try:
                os.symlink(target_dir, self.active_symlink)
            except OSError:
                pass

    def list_accounts(self) -> list[Account]:
        """List all configured accounts."""
        doc = self.storage.load_accounts()
        return [Account.from_dict(a) for a in doc.get("accounts", [])]

    def get_active_account_id(self) -> str | None:
        """Get the ID of the currently active account."""
        doc = self.storage.load_accounts()
        return doc.get("active_account_id")

    def get_active_account(self) -> Account | None:
        """Get the Account model of the currently active account."""
        active_id = self.get_active_account_id()
        if not active_id:
            return None
        return self.get_account(active_id)

    def get_account(self, id_or_username: str) -> Account | None:
        """Find an account by its unique ID or username."""
        accounts = self.list_accounts()
        for acc in accounts:
            if acc.id == id_or_username or acc.username.lower() == id_or_username.lower():
                return acc
        return None

    def add_account(
        self,
        label: str,
        username: str,
        auth_type: str = "api_key",
        credentials: dict[str, Any] | None = None,
        make_active: bool = False,
        quota_notes: str = "30h weekly GPU",
        notes: str = "",
        account_id: str | None = None,
    ) -> Account:
        """Register a new Kaggle account and initialize its environment."""
        username = username.strip()
        if not username:
            raise ValueError("Username cannot be empty")

        clean_slug = "".join(c if c.isalnum() else "_" for c in username.lower())
        uid = account_id or f"acc_{clean_slug}_{uuid.uuid4().hex[:6]}"

        account = Account(
            id=uid,
            label=label or username,
            username=username,
            auth_type=auth_type,
            credentials=credentials or {},
            created_at=utc_now_iso(),
            last_used_at=None,
            status="valid",
            quota_notes=quota_notes,
            notes=notes,
        )

        # Write credentials to sandbox environment
        self._sync_env_credentials(account)

        # Update storage
        doc = self.storage.load_accounts()
        accounts_list = doc.get("accounts", [])

        # Remove existing if same ID
        accounts_list = [a for a in accounts_list if a.get("id") != account.id]
        accounts_list.append(account.to_dict())
        doc["accounts"] = accounts_list

        # If first account or make_active requested, set active
        if make_active or not doc.get("active_account_id") or len(accounts_list) == 1:
            doc["active_account_id"] = account.id
            self._update_active_symlink(account.id)

        self.storage.save_accounts(doc)
        return account

    def update_account(
        self,
        id_or_username: str,
        label: str | None = None,
        credentials: dict[str, Any] | None = None,
        quota_notes: str | None = None,
        notes: str | None = None,
        status: str | None = None,
    ) -> Account | None:
        """Update existing account fields."""
        doc = self.storage.load_accounts()
        accounts_list = doc.get("accounts", [])
        target_dict = None
        for a in accounts_list:
            if a.get("id") == id_or_username or a.get("username", "").lower() == id_or_username.lower():
                target_dict = a
                break

        if not target_dict:
            return None

        if label is not None:
            target_dict["label"] = label
        if credentials is not None:
            target_dict["credentials"] = credentials
        if quota_notes is not None:
            target_dict["quota_notes"] = quota_notes
        if notes is not None:
            target_dict["notes"] = notes
        if status is not None:
            target_dict["status"] = status

        account = Account.from_dict(target_dict)
        if credentials is not None:
            self._sync_env_credentials(account)

        self.storage.save_accounts(doc)
        return account

    def set_active_account(self, id_or_username: str) -> Account:
        """Switch active account and update ~/.kaggle_hub/active symlink."""
        account = self.get_account(id_or_username)
        if not account:
            raise ValueError(f"Account '{id_or_username}' not found")

        doc = self.storage.load_accounts()
        doc["active_account_id"] = account.id
        self.storage.save_accounts(doc)
        self._update_active_symlink(account.id)
        return account

    def delete_account(self, id_or_username: str) -> bool:
        """Delete an account and cleanup its environment."""
        account = self.get_account(id_or_username)
        if not account:
            return False

        doc = self.storage.load_accounts()
        doc["accounts"] = [a for a in doc.get("accounts", []) if a.get("id") != account.id]

        if doc.get("active_account_id") == account.id:
            remaining = doc["accounts"]
            doc["active_account_id"] = remaining[0]["id"] if remaining else None
            self._update_active_symlink(doc["active_account_id"])

        self.storage.save_accounts(doc)

        # Cleanup environment directory
        acc_env_dir = self.envs_dir / account.id
        if acc_env_dir.exists():
            shutil.rmtree(acc_env_dir, ignore_errors=True)

        return True

    def get_env_for_account(self, id_or_username: str | None = None) -> dict[str, str]:
        """Construct isolated environment variables for Kaggle subprocess execution."""
        if id_or_username:
            account = self.get_account(id_or_username)
        else:
            account = self.get_active_account()

        if not account:
            raise ValueError(
                f"No active or matching account found for '{id_or_username or 'active'}'"
            )

        env = os.environ.copy()
        acc_env_dir = self.envs_dir / account.id

        # Make sure credentials file (and the private HOME) are present
        self._sync_env_credentials(account)

        env["KAGGLE_CONFIG_DIR"] = str(acc_env_dir)
        # The SDK resolves ~/.kaggle/credentials.json and ~/.kaggle/access_token
        # from HOME, so HOME must be per-account too (see _ensure_private_home).
        real_home = str(Path.home())
        env["HOME"] = str(acc_env_dir / "home")
        # A host-level token outranks every other auth method in the CLI and
        # would make every account act as the host user.
        env.pop("KAGGLE_API_TOKEN", None)

        if account.auth_type == "api_key":
            env["KAGGLE_USERNAME"] = account.username
            key = account.credentials.get("key", "")
            if key:
                env["KAGGLE_KEY"] = key
            else:
                env.pop("KAGGLE_KEY", None)
        else:
            # Legacy key auth outranks OAuth and needs username AND key: clear
            # the key so a host KAGGLE_KEY cannot hijack this account. The
            # username alone is inert, and pinning it overrides a host value.
            env.pop("KAGGLE_KEY", None)
            env["KAGGLE_USERNAME"] = account.username

        # Ensure standard binary search paths are available in PATH
        path_elements = env.get("PATH", "").split(os.pathsep)
        user_local_bin = str(Path(real_home) / ".local" / "bin")
        for p in [user_local_bin, "/usr/local/bin", "/usr/bin", "/bin"]:
            if p not in path_elements:
                path_elements.insert(0, p)
        env["PATH"] = os.pathsep.join(path_elements)

        return env

    # Alias for API compatibility across test runners and CLI callers
    get_process_env = get_env_for_account

    def run_kaggle(
        self,
        account: Account,
        args: list[str],
        timeout: int = 60,
    ) -> subprocess.CompletedProcess:
        """Run a `kaggle ...` command as `account`, then persist refreshed tokens."""
        env = self.get_env_for_account(account.id)
        try:
            return subprocess.run(
                ["kaggle", *args], env=env, capture_output=True, text=True, timeout=timeout,
            )
        finally:
            self.pull_refreshed_credentials(account)

    def whoami(self, id_or_username: str | None = None) -> tuple[str | None, str]:
        """Return (authenticated username, raw output) via `kaggle config view`."""
        account = self.get_account(id_or_username) if id_or_username else self.get_active_account()
        if not account:
            return None, "No account specified or currently active."
        proc = self.run_kaggle(account, ["config", "view"], timeout=30)
        out = (proc.stdout or "") + (proc.stderr or "")
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("- username:"):
                name = line.split(":", 1)[1].strip()
                return (name if name and name != "None" else None), out
        return None, out

    def test_account(self, id_or_username: str | None = None) -> tuple[bool, str]:
        """Test Kaggle authentication by running `kaggle kernels list --mine`."""
        if id_or_username:
            account = self.get_account(id_or_username)
        else:
            account = self.get_active_account()

        if not account:
            return False, "No account specified or currently active."

        try:
            proc = self.run_kaggle(
                account, ["kernels", "list", "--mine", "--page-size", "1"], timeout=30,
            )

            if proc.returncode == 0:
                # A working login is not enough: it must be THIS account's.
                # A leaked host credential authenticates fine as someone else.
                actual, _ = self.whoami(account.id)
                if actual and actual.lower() != account.username.lower():
                    self.update_account(account.id, status="invalid")
                    return False, (
                        f"Authenticated as '{actual}', not '{account.username}'. "
                        "Credentials are leaking between accounts."
                    )
                self.update_account(
                    account.id,
                    status="valid",
                )
                # Update last used
                doc = self.storage.load_accounts()
                for a in doc.get("accounts", []):
                    if a.get("id") == account.id:
                        a["last_used_at"] = utc_now_iso()
                        break
                self.storage.save_accounts(doc)
                return True, f"Authentication successful for user '{account.username}'"
            else:
                err_msg = (proc.stderr or proc.stdout).strip()
                self.update_account(account.id, status="invalid")
                return False, f"Authentication failed: {err_msg}"
        except Exception as e:
            self.update_account(account.id, status="invalid")
            return False, f"Validation check failed: {str(e)}"

    def login_oauth(
        self,
        label: str | None = None,
        make_active: bool = False,
        no_browser: bool = False,
        timeout: int = 600,
    ) -> Account:
        """
        Add an account through Kaggle's browser OAuth login, fully sandboxed.

        `kaggle auth login` runs with HOME set to a fresh sandbox, so the token
        lands there and never touches (or replaces) the real ~/.kaggle login.
        The consent page opens in the user's normal browser profile -- whichever
        Kaggle user is signed in there is the account that gets added -- by
        launching the browser with the REAL home via $BROWSER; a sandboxed
        HOME would otherwise start an empty browser profile.
        """
        pending, env = self._new_login_sandbox()
        try:
            cmd = ["kaggle", "auth", "login"]
            if no_browser:
                cmd.append("--no-launch-browser")
            # Inherit the terminal: the user may need to see the URL or paste a code.
            proc = subprocess.run(cmd, env=env, timeout=timeout)
            return self._save_login(pending, proc.returncode, label, make_active)
        finally:
            shutil.rmtree(pending, ignore_errors=True)

    # -- sign-in shared helpers -------------------------------------------------

    def _new_login_sandbox(self) -> tuple[Path, dict]:
        """A throwaway env dir + environment in which `kaggle auth login` saves there."""
        pending = self.envs_dir / f"pending_{uuid.uuid4().hex[:8]}"
        pending.mkdir(mode=0o700, parents=True)
        home = self._ensure_private_home(pending)
        env = os.environ.copy()
        for k in ("KAGGLE_API_TOKEN", "KAGGLE_KEY", "KAGGLE_USERNAME"):
            env.pop(k, None)
        env["HOME"] = str(home)
        env["KAGGLE_CONFIG_DIR"] = str(pending)
        env["BROWSER"] = f"env HOME={shlex.quote(str(Path.home()))} xdg-open %s"
        env["PYTHONUNBUFFERED"] = "1"          # so a piped URL prompt arrives at once
        return pending, env

    def _save_login(self, pending: Path, returncode: int, label: str | None,
                    make_active: bool) -> Account:
        creds = self._read_json(pending / "credentials.json")
        if returncode != 0 or not creds.get("refresh_token"):
            raise RuntimeError("Kaggle login did not complete (no credentials were saved).")
        username = (creds.get("username") or "").strip()
        if not username:
            raise RuntimeError("Login succeeded but returned no username.")
        existing = self.get_account(username)
        if existing:
            raise RuntimeError(
                f"'{username}' is already configured (id {existing.id}). Kaggle signed in as "
                "the account your browser is logged in to. Use 'Sign in with a link' and open "
                "the link in a private window, then sign in there as the other account."
            )
        return self.add_account(
            label=label or username,
            username=username,
            auth_type="oauth_token",
            credentials=creds,
            make_active=make_active,
            notes="Added via sandboxed `kaggle auth login`",
        )

    # -- sign-in with a link + verification code (works from the web panel) ----
    #
    # `kaggle auth login --no-launch-browser` prints an authorize URL and reads
    # the verification code from stdin. Splitting that into start/finish lets a
    # web page show the link (open it in any browser profile or private window,
    # so any Kaggle account can be chosen) and send the code back.

    LINK_LOGIN_TTL = 15 * 60

    def start_link_login(self, label: str | None = None, make_active: bool = False,
                         wait: float = 30.0) -> dict:
        """Start a link login. Returns {"session": id, "url": authorize URL}."""
        import threading
        import time

        self._expire_link_logins()
        pending, env = self._new_login_sandbox()
        try:
            proc = subprocess.Popen(
                ["kaggle", "auth", "login", "--no-launch-browser"], env=env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
        except OSError as e:
            shutil.rmtree(pending, ignore_errors=True)
            raise RuntimeError(f"Could not start the Kaggle CLI: {e}") from e

        out: list[str] = []
        found = threading.Event()

        def drain():                    # keep reading so the CLI never blocks on a full pipe
            for line in proc.stdout:
                out.append(line)
                if "oauth2/authorize" in line:
                    found.set()
            found.set()

        threading.Thread(target=drain, daemon=True).start()
        found.wait(wait)
        url = next((l.strip() for l in out if "oauth2/authorize" in l), None)
        if not url:
            proc.kill()
            shutil.rmtree(pending, ignore_errors=True)
            tail = "".join(out).strip()[-300:]
            raise RuntimeError(f"Kaggle CLI did not print a sign-in link. {tail}".strip())

        sid = uuid.uuid4().hex
        self._link_logins[sid] = dict(proc=proc, pending=pending, out=out, label=label,
                                      make_active=make_active, started=time.time())
        return {"session": sid, "url": url}

    def finish_link_login(self, session: str, code: str, timeout: int = 60) -> Account:
        """Send the verification code for a started link login and save the account."""
        sess = self._link_logins.pop(session, None)
        if not sess:
            raise RuntimeError("This sign-in has expired or was cancelled. Start again.")
        proc, pending = sess["proc"], sess["pending"]
        try:
            code = (code or "").strip()
            if not code:
                raise RuntimeError("Paste the verification code shown by Kaggle.")
            try:
                proc.stdin.write(code + "\n")
                proc.stdin.close()
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                raise RuntimeError("Kaggle did not accept the code in time. Start again.")
            except (BrokenPipeError, OSError):
                proc.wait(timeout=5)
            try:
                return self._save_login(pending, proc.returncode, sess["label"],
                                        sess["make_active"])
            except RuntimeError as e:
                if proc.returncode != 0:
                    tail = "".join(sess["out"]).strip().splitlines()[-1:] or [""]
                    raise RuntimeError(f"Kaggle rejected the code. {tail[0]}".strip()) from e
                raise
        finally:
            if proc.poll() is None:
                proc.kill()
            shutil.rmtree(pending, ignore_errors=True)

    def cancel_link_login(self, session: str) -> bool:
        sess = self._link_logins.pop(session, None)
        if not sess:
            return False
        if sess["proc"].poll() is None:
            sess["proc"].kill()
        shutil.rmtree(sess["pending"], ignore_errors=True)
        return True

    def _expire_link_logins(self) -> None:
        import time
        for sid, sess in list(self._link_logins.items()):
            if time.time() - sess["started"] > self.LINK_LOGIN_TTL:
                self.cancel_link_login(sid)

    def import_existing_credentials(self) -> list[Account]:
        """Import existing credentials from ~/.kaggle/credentials.json or ~/.kaggle/kaggle.json."""
        imported: list[Account] = []
        home_kaggle = Path.home() / ".kaggle"

        # 1. Check credentials.json (OAuth)
        oauth_file = home_kaggle / "credentials.json"
        if oauth_file.exists():
            try:
                with open(oauth_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                username = data.get("username")
                if username and not self.get_account(username):
                    acc = self.add_account(
                        label=f"{username} (Default OAuth)",
                        username=username,
                        auth_type="oauth_token",
                        credentials=data,
                        notes="Auto-imported from ~/.kaggle/credentials.json",
                    )
                    imported.append(acc)
            except Exception:
                pass

        # 2. Check kaggle.json (API Key)
        api_file = home_kaggle / "kaggle.json"
        if api_file.exists():
            try:
                with open(api_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                username = data.get("username")
                key = data.get("key")
                if username and key and not self.get_account(username):
                    acc = self.add_account(
                        label=f"{username} (Default API Key)",
                        username=username,
                        auth_type="api_key",
                        credentials={"key": key},
                        notes="Auto-imported from ~/.kaggle/kaggle.json",
                    )
                    imported.append(acc)
            except Exception:
                pass

        return imported
