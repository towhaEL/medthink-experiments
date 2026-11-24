"""
REST API request handlers and endpoint dispatchers for Kaggle Hub.
Zero external dependencies.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .. import __version__, config
from ..account_manager import AccountManager
from ..models import JobStatus, RunOptions
from ..notebook_manager import NotebookManager
from ..kaggle_runner import KaggleRunner
from ..storage import StorageEngine


class ApiDispatcher:
    """Dispatches incoming JSON REST requests to core Kaggle Hub managers."""

    def __init__(
        self,
        account_manager: AccountManager | None = None,
        notebook_manager: NotebookManager | None = None,
        runner: KaggleRunner | None = None,
        storage: StorageEngine | None = None,
    ) -> None:
        self.storage = storage or StorageEngine()
        self.account_manager = account_manager or AccountManager(storage=self.storage)
        self.notebook_manager = notebook_manager or NotebookManager()
        self.runner = runner or KaggleRunner(
            account_manager=self.account_manager,
            notebook_manager=self.notebook_manager,
            storage=self.storage,
        )

    def dispatch(
        self,
        method: str,
        path: str,
        query_string: str = "",
        body_data: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """Main dispatcher entrypoint returning (HTTP status code, response dict)."""
        method = method.upper()
        clean_path = path.strip("/").split("/")
        # Path format: /api/<resource>/[sub_resource]/...
        if not clean_path or clean_path[0] != "api":
            return 404, {"error": "Not Found"}

        resource = clean_path[1] if len(clean_path) > 1 else ""
        query = parse_qs(query_string)

        try:
            if resource == "status":
                return self._handle_status(method)

            elif resource == "accounts":
                return self._handle_accounts(method, clean_path[2:], query, body_data or {})

            elif resource == "notebooks":
                return self._handle_notebooks(method, clean_path[2:], query)

            elif resource == "runs":
                return self._handle_runs(method, clean_path[2:], query, body_data or {})

            return 404, {"error": f"Unknown API endpoint: /{'/'.join(clean_path)}"}

        except ValueError as ve:
            return 400, {"error": str(ve)}
        except Exception as ex:
            return 500, {"error": f"Internal Server Error: {str(ex)}"}

    @staticmethod
    def _public(account) -> dict[str, Any]:
        """Account as JSON for the browser: tokens and API keys never leave the server."""
        d = account.to_dict()
        d["credentials"] = {}
        return d

    # -------------------------------------------------------------------------
    # /api/status
    # -------------------------------------------------------------------------
    def _handle_status(self, method: str) -> tuple[int, dict[str, Any]]:
        if method != "GET":
            return 405, {"error": "Method Not Allowed"}

        active_account = self.account_manager.get_active_account()
        accounts = self.account_manager.list_accounts()
        notebooks = self.notebook_manager.list_notebooks()

        # Check kaggle CLI version and path
        kaggle_cli_path = shutil.which("kaggle")
        kaggle_version = "Unknown"
        kaggle_ok = False
        if kaggle_cli_path:
            try:
                proc = subprocess.run(
                    ["kaggle", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if proc.returncode == 0:
                    kaggle_version = proc.stdout.strip()
                    kaggle_ok = True
                else:
                    kaggle_version = proc.stderr.strip() or "Installed (Error running)"
            except Exception as e:
                kaggle_version = f"Error: {e}"

        return 200, {
            "version": __version__,
            "status": "ok",
            "kaggle_cli": {
                "installed": kaggle_ok,
                "path": kaggle_cli_path,
                "version": kaggle_version,
            },
            "active_account": self._public(active_account) if active_account else None,
            "total_accounts": len(accounts),
            "total_notebooks": len(notebooks),
            "base_hub_dir": str(self.storage.base_dir),
            "staging_dir": str(config.STAGING_DIR),
        }

    # -------------------------------------------------------------------------
    # /api/accounts
    # -------------------------------------------------------------------------
    def _handle_accounts(
        self,
        method: str,
        sub_path: list[str],
        query: dict[str, list[str]],
        body: dict[str, Any],
    ) -> tuple[int, dict[str, Any]]:
        # GET /api/accounts
        if not sub_path:
            if method == "GET":
                accounts = self.account_manager.list_accounts()
                active_id = self.account_manager.get_active_account_id()
                usage = self.runner.gpu_usage_by_account(days=7)
                out = []
                for a in accounts:
                    d = self._public(a)
                    d["usage"] = usage.get(a.id, {"gpu_hours": 0.0, "active_jobs": 0})
                    out.append(d)
                return 200, {
                    "accounts": out,
                    "active_account_id": active_id,
                    "usage_window_days": 7,
                }
            elif method == "POST":
                # Add account
                username = body.get("username", "").strip()
                if not username:
                    return 400, {"error": "Missing required 'username' field"}

                auth_type = body.get("auth_type", "api_key")
                credentials = body.get("credentials", {})
                if auth_type == "api_key" and not credentials.get("key"):
                    if "api_key" in body:
                        credentials["key"] = body["api_key"]
                    elif "key" in body:
                        credentials["key"] = body["key"]
                    else:
                        return 400, {"error": "Missing 'api_key' or credentials key"}

                label = body.get("label", username)
                make_active = bool(body.get("make_active", False))
                quota_notes = body.get("quota_notes", "30h weekly GPU")
                notes = body.get("notes", "")

                acc = self.account_manager.add_account(
                    label=label,
                    username=username,
                    auth_type=auth_type,
                    credentials=credentials,
                    make_active=make_active,
                    quota_notes=quota_notes,
                    notes=notes,
                )
                return 201, {"success": True, "account": self._public(acc)}
            else:
                return 405, {"error": "Method Not Allowed"}

        # POST /api/accounts/login -- sandboxed browser sign-in (blocks until done)
        if len(sub_path) == 1 and sub_path[0] == "login":
            if method != "POST":
                return 405, {"error": "Method Not Allowed"}
            try:
                acc = self.account_manager.login_oauth(
                    label=(body.get("label") or "").strip() or None,
                    make_active=bool(body.get("make_active", False)),
                    timeout=int(body.get("timeout", 600)),
                )
            except RuntimeError as e:
                return 409, {"error": str(e)}
            valid, msg = self.account_manager.test_account(acc.id)
            acc = self.account_manager.get_account(acc.id) or acc
            return 201, {"success": True, "account": self._public(acc), "valid": valid, "message": msg}

        # POST /api/accounts/login/{start,finish,cancel} -- sign in with a link + code.
        # The link can be opened in any browser profile or private window, so
        # any Kaggle account can be added, not just the one the browser uses.
        if len(sub_path) == 2 and sub_path[0] == "login":
            if method != "POST":
                return 405, {"error": "Method Not Allowed"}
            step = sub_path[1]
            try:
                if step == "start":
                    return 200, self.account_manager.start_link_login(
                        label=(body.get("label") or "").strip() or None,
                        make_active=bool(body.get("make_active", False)),
                    )
                if step == "finish":
                    acc = self.account_manager.finish_link_login(
                        str(body.get("session", "")), str(body.get("code", "")))
                    valid, msg = self.account_manager.test_account(acc.id)
                    acc = self.account_manager.get_account(acc.id) or acc
                    return 201, {"success": True, "account": self._public(acc),
                                 "valid": valid, "message": msg}
                if step == "cancel":
                    return 200, {"success": self.account_manager.cancel_link_login(
                        str(body.get("session", "")))}
            except RuntimeError as e:
                return 409, {"error": str(e)}
            return 404, {"error": f"Unknown login step '{step}'"}

        # Sub-endpoints: /api/accounts/import
        if len(sub_path) == 1 and sub_path[0] == "import":
            if method == "POST":
                imported = self.account_manager.import_existing_credentials()
                return 200, {
                    "success": True,
                    "imported_count": len(imported),
                    "accounts": [self._public(a) for a in imported],
                }
            return 405, {"error": "Method Not Allowed"}

        acc_id = sub_path[0]

        # DELETE /api/accounts/:id
        if len(sub_path) == 1 and method == "DELETE":
            ok = self.account_manager.delete_account(acc_id)
            if ok:
                return 200, {"success": True, "deleted_id": acc_id}
            return 404, {"error": f"Account '{acc_id}' not found"}

        # GET /api/accounts/:id
        if len(sub_path) == 1 and method == "GET":
            acc = self.account_manager.get_account(acc_id)
            if not acc:
                return 404, {"error": f"Account '{acc_id}' not found"}
            return 200, {"account": self._public(acc)}

        # Action sub-endpoints: /api/accounts/:id/switch or /api/accounts/:id/test
        if len(sub_path) == 2:
            action = sub_path[1]
            if action == "switch" and method == "POST":
                acc = self.account_manager.set_active_account(acc_id)
                return 200, {"success": True, "active_account_id": acc.id, "account": self._public(acc)}

            elif action == "test" and method == "POST":
                acc = self.account_manager.get_account(acc_id)
                if not acc:
                    return 404, {"error": f"Account '{acc_id}' not found"}
                valid, msg = self.account_manager.test_account(acc.id)
                return 200, {"valid": valid, "message": msg, "status": "valid" if valid else "invalid"}

        return 404, {"error": "Endpoint Not Found"}

    # -------------------------------------------------------------------------
    # /api/notebooks
    # -------------------------------------------------------------------------
    def _handle_notebooks(
        self,
        method: str,
        sub_path: list[str],
        query: dict[str, list[str]],
    ) -> tuple[int, dict[str, Any]]:
        if method != "GET":
            return 405, {"error": "Method Not Allowed"}

        # GET /api/notebooks
        if not sub_path:
            custom_dir = query.get("path", [None])[0]
            target_path = Path(custom_dir) if custom_dir else None
            notebooks = self.notebook_manager.list_notebooks(directory=target_path)
            return 200, {
                "notebooks": [n.to_dict() for n in notebooks],
                "directory": str(target_path or self.notebook_manager.notebooks_dir),
            }

        # GET /api/notebooks/:name
        nb_name = sub_path[0]
        try:
            info, data = self.notebook_manager.inspect_notebook(nb_name)
            cells = data.get("cells", [])
            preview = []
            for cell in cells[:5]:
                ctype = cell.get("cell_type", "")
                src = cell.get("source", [])
                src_str = "".join(src) if isinstance(src, list) else src
                preview.append({
                    "cell_type": ctype,
                    "preview": src_str[:200],
                })
            return 200, {
                "notebook": info.to_dict(),
                "cell_preview": preview,
            }
        except FileNotFoundError:
            return 404, {"error": f"Notebook '{nb_name}' not found"}
        except Exception as e:
            return 400, {"error": f"Failed to parse notebook: {e}"}

    # -------------------------------------------------------------------------
    # /api/runs
    # -------------------------------------------------------------------------
    def _handle_runs(
        self,
        method: str,
        sub_path: list[str],
        query: dict[str, list[str]],
        body: dict[str, Any],
    ) -> tuple[int, dict[str, Any]]:
        # GET /api/runs
        if not sub_path:
            if method == "GET":
                limit_str = query.get("limit", ["50"])[0]
                limit = int(limit_str) if limit_str.isdigit() else 50
                acc_filter = query.get("account", [None])[0]
                jobs = self.runner.list_jobs(account_id=acc_filter, limit=limit)
                return 200, {"jobs": [j.to_dict() for j in jobs]}

            elif method == "POST":
                # Launch a new run
                notebook = body.get("notebook")
                if not notebook:
                    return 400, {"error": "Missing 'notebook' parameter"}

                account_id = body.get("account_id")
                title = body.get("title")
                slug = body.get("slug")
                accelerator = body.get("accelerator", "NvidiaTeslaT4")
                enable_gpu = bool(body.get("enable_gpu", True))
                enable_internet = bool(body.get("enable_internet", True))
                auto_download = bool(body.get("auto_download", True))
                # Kernel run-time cap on Kaggle (`kernels push -t`). Empty -> 12h session limit.
                max_hours = body.get("max_runtime_hours")
                max_runtime = int(float(max_hours) * 3600) if max_hours not in (None, "") else None
                dataset_sources = body.get("dataset_sources", [])
                if isinstance(dataset_sources, str):
                    dataset_sources = [d.strip() for d in dataset_sources.split(",") if d.strip()]

                options = RunOptions(
                    notebook=notebook,
                    account_id=account_id,
                    title=title,
                    slug=slug,
                    accelerator=accelerator,
                    enable_gpu=enable_gpu,
                    enable_internet=enable_internet,
                    dataset_sources=dataset_sources,
                    auto_download=auto_download,
                )

                # Submit staging
                job = self.runner.submit_job(options)

                # Push directly
                job = self.runner.push_job(job.id, accelerator=accelerator, timeout=max_runtime)

                # If auto-download is requested, start background watcher
                if auto_download and job.status != JobStatus.ERROR:
                    self.runner.run_job_async(job.id, auto_download=True, poll_interval=10)

                return 201, {"success": True, "job": job.to_dict()}
            else:
                return 405, {"error": "Method Not Allowed"}

        job_id = sub_path[0]

        # GET /api/runs/:id
        if len(sub_path) == 1 and method == "GET":
            # Refresh status if requested or if job is currently active
            refresh = query.get("refresh", ["false"])[0].lower() in ["true", "1"]
            job = self.runner.get_job(job_id)
            if not job:
                return 404, {"error": f"Job '{job_id}' not found"}

            if refresh or job.status in [JobStatus.QUEUED, JobStatus.RUNNING]:
                try:
                    job = self.runner.check_status(job_id)
                except Exception:
                    pass

            return 200, {"job": job.to_dict()}

        # GET /api/runs/:id/logs
        if len(sub_path) == 2 and sub_path[1] == "logs":
            if method != "GET":
                return 405, {"error": "Method Not Allowed"}
            job = self.runner.get_job(job_id)
            if not job:
                return 404, {"error": f"Job '{job_id}' not found"}

            follow = query.get("follow", ["false"])[0].lower() in ["true", "1"]
            logs = self.runner.get_logs(job_id, follow=follow)
            return 200, {
                "job_id": job.id,
                "status": job.status,
                "logs": logs,
            }

        # POST /api/runs/:id/download
        if len(sub_path) == 2 and sub_path[1] == "download":
            if method != "POST":
                return 405, {"error": "Method Not Allowed"}
            custom_dir = body.get("output_path") if body else None
            try:
                out_dir = self.runner.download_outputs(job_id, output_dir=custom_dir)
                job = self.runner.get_job(job_id)
                return 200, {
                    "success": True,
                    "output_dir": out_dir,
                    "job": job.to_dict() if job else None,
                }
            except Exception as e:
                return 500, {"error": f"Download failed: {e}"}

        return 404, {"error": "Endpoint Not Found"}
