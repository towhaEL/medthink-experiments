"""
Headless Kaggle kernel runner, status monitor, and output downloader.
Zero external dependencies.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from . import config
from .account_manager import AccountManager
from .models import Account, JobRecord, JobStatus, RunOptions, utc_now_iso
from .notebook_manager import NotebookManager
from .storage import StorageEngine


class KaggleRunner:
    """Manages Kaggle CLI execution, tracking job status, logs, and artifacts."""

    def __init__(
        self,
        account_manager: AccountManager | None = None,
        notebook_manager: NotebookManager | None = None,
        storage: StorageEngine | None = None,
    ) -> None:
        self.storage = storage or StorageEngine()
        self.account_manager = account_manager or AccountManager(storage=self.storage)
        self.notebook_manager = notebook_manager or NotebookManager()
        self._lock = threading.RLock()

    def list_jobs(self, account_id: str | None = None, limit: int = 50) -> list[JobRecord]:
        """List past and current jobs, newest first."""
        with self._lock:
            doc = self.storage.load_jobs()
            jobs = [JobRecord.from_dict(j) for j in doc.get("jobs", [])]
            if account_id:
                jobs = [j for j in jobs if j.account_id == account_id]
            jobs.sort(key=lambda j: j.created_at or "", reverse=True)
            return jobs[:limit]

    def gpu_usage_by_account(self, days: int = 7) -> dict[str, dict[str, Any]]:
        """
        Approximate accelerator hours per account over the last `days` days.

        Only jobs launched through Kaggle Hub are counted, and a run's end is
        the moment the hub observed it finish, so this is an estimate for
        spreading work across accounts -- not Kaggle's official quota meter.
        """
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        since = now - timedelta(days=days)
        usage: dict[str, dict[str, Any]] = {}

        def parse(ts: str | None):
            if not ts:
                return None
            try:
                return datetime.fromisoformat(ts)
            except ValueError:
                return None

        for job in self.list_jobs(limit=10_000):
            u = usage.setdefault(job.account_id, {"gpu_hours": 0.0, "active_jobs": 0})
            if job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
                u["active_jobs"] += 1
            if not job.enable_gpu or job.accelerator == "None":
                continue
            start = parse(job.started_at)
            if start is None:
                continue
            end = parse(job.completed_at)
            if end is None:
                if job.status not in (JobStatus.QUEUED, JobStatus.RUNNING):
                    continue                      # finished but never observed
                end = now
            start = max(start, since)
            if end > start:
                u["gpu_hours"] += (end - start).total_seconds() / 3600
        for u in usage.values():
            u["gpu_hours"] = round(u["gpu_hours"], 2)
        return usage

    def get_job(self, job_id: str) -> JobRecord | None:
        """Get a single job record by ID."""
        with self._lock:
            doc = self.storage.load_jobs()
            for j in doc.get("jobs", []):
                if j.get("id") == job_id or j.get("kernel_slug") == job_id:
                    return JobRecord.from_dict(j)
            return None

    def save_job(self, job: JobRecord) -> None:
        """Persist updated job record to storage."""
        with self._lock:
            doc = self.storage.load_jobs()
            jobs_list = doc.get("jobs", [])
            updated = False
            for idx, item in enumerate(jobs_list):
                if item.get("id") == job.id:
                    jobs_list[idx] = job.to_dict()
                    updated = True
                    break
            if not updated:
                jobs_list.append(job.to_dict())
            doc["jobs"] = jobs_list
            self.storage.save_jobs(doc)

    def submit_job(self, options: RunOptions) -> JobRecord:
        """Prepare staging directory and register a new queued job."""
        account = None
        if options.account_id:
            account = self.account_manager.get_account(options.account_id)
        if not account:
            account = self.account_manager.get_active_account()

        if not account:
            raise ValueError(
                "No Kaggle account found. Please add or activate an account first."
            )

        # Inspect notebook to obtain slug and title
        nb_info, _ = self.notebook_manager.inspect_notebook(options.notebook)
        slug = options.slug or nb_info.slug
        title = options.title or nb_info.title
        kernel_id = f"{account.username}/{slug}"

        job_ts = time.strftime("%Y%m%d_%H%M%S")
        job_id = f"job_{job_ts}_{slug[:12]}_{uuid.uuid4().hex[:4]}"

        # Prepare staging
        staging_path = self.notebook_manager.prepare_staging(
            job_id=job_id,
            notebook_path=nb_info.path,
            options=options,
            account=account,
        )

        accelerator = config.normalize_accelerator(options.accelerator)

        job = JobRecord(
            id=job_id,
            account_id=account.id,
            username=account.username,
            kernel_slug=slug,
            kernel_id=kernel_id,
            title=title,
            notebook_path=nb_info.path,
            staging_path=str(staging_path),
            accelerator=accelerator,
            enable_gpu=options.enable_gpu,
            enable_internet=options.enable_internet,
            dataset_sources=options.dataset_sources,
            competition_sources=options.competition_sources,
            kernel_sources=options.kernel_sources,
            model_sources=options.model_sources,
            status=JobStatus.QUEUED,
            created_at=utc_now_iso(),
        )

        self.save_job(job)
        return job

    def push_job(
        self,
        job_id: str,
        timeout: int | None = None,
        accelerator: str | None = None,
    ) -> JobRecord:
        """
        Push prepared staging directory to Kaggle via `kaggle kernels push`.

        `timeout` is passed as `kernels push -t`, which is the kernel's maximum
        RUN TIME on Kaggle, not a network timeout. It is omitted unless given,
        so runs get Kaggle's full session limit (12h).
        """
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job '{job_id}' not found")

        account = self.account_manager.get_account(job.account_id)
        if not account:
            raise ValueError(f"Account '{job.account_id}' for job not found")

        target_acc = accelerator or job.accelerator
        target_acc = config.normalize_accelerator(target_acc)

        cmd = ["kernels", "push", "-p", str(job.staging_path)]
        if target_acc and target_acc != "None":
            cmd.extend(["--accelerator", target_acc])
        if timeout:
            cmd.extend(["-t", str(int(timeout))])

        try:
            proc = self.account_manager.run_kaggle(account, cmd, timeout=300)

            stdout = (proc.stdout or "").strip()
            stderr = (proc.stderr or "").strip()
            # The Kaggle CLI exits 0 on rejected pushes and reports them on
            # stdout ("Kernel push error: Maximum batch GPU session count of 2
            # reached."). Trusting the exit code records them as running.
            rejected = "push error" in stdout.lower()

            if proc.returncode == 0 and not rejected:
                job.status = JobStatus.RUNNING
                job.started_at = utc_now_iso()
                job.exit_code = 0
                job.error_message = None
                job.last_log_snippet = stdout
            else:
                job.status = JobStatus.ERROR
                job.exit_code = proc.returncode
                job.error_message = stderr or stdout
                job.last_log_snippet = stderr or stdout

        except Exception as e:
            job.status = JobStatus.ERROR
            job.error_message = f"Failed to execute push: {str(e)}"

        self.save_job(job)
        return job

    def check_status(self, job_id: str) -> JobRecord:
        """Query Kaggle for latest kernel status via `kaggle kernels status`."""
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job '{job_id}' not found")

        account = self.account_manager.get_account(job.account_id)
        if not account:
            return job

        try:
            proc = self.account_manager.run_kaggle(
                account, ["kernels", "status", job.kernel_id], timeout=30,
            )

            output = (proc.stdout or proc.stderr or "").strip()
            lower = output.lower()

            # A failed query (network outage, 404, rate limit) says nothing
            # about the kernel. Parsing its text for "error" used to mark live
            # runs as ERROR, which made the scheduler re-push a running kernel.
            if proc.returncode != 0:
                job.last_log_snippet = f"Status check failed: {output[:300]}"
                self.save_job(job)
                return job

            if "complete" in lower or "has completed" in lower:
                job.status = JobStatus.COMPLETE
                if not job.completed_at:
                    job.completed_at = utc_now_iso()
            elif "running" in lower or "is running" in lower:
                job.status = JobStatus.RUNNING
            elif "queue" in lower or "is queued" in lower:
                job.status = JobStatus.QUEUED
            elif "fail" in lower or "error" in lower:
                job.status = JobStatus.ERROR
                job.error_message = output
                if not job.completed_at:
                    job.completed_at = utc_now_iso()
            elif "cancel" in lower:
                job.status = JobStatus.CANCELLED
                if not job.completed_at:
                    job.completed_at = utc_now_iso()

            job.last_log_snippet = output
            self.save_job(job)

        except Exception as e:
            job.last_log_snippet = f"Status check error: {e}"
            self.save_job(job)

        return job

    def get_logs(self, job_id: str, follow: bool = False) -> str:
        """Fetch kernel logs via `kaggle kernels logs`."""
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job '{job_id}' not found")

        account = self.account_manager.get_account(job.account_id)
        if not account:
            raise ValueError(f"Account '{job.account_id}' not found")

        cmd = ["kernels", "logs", job.kernel_id]
        if follow:
            cmd.append("-f")

        try:
            proc = self.account_manager.run_kaggle(account, cmd, timeout=60)
            logs = proc.stdout or proc.stderr or ""
            # Update last snippet
            lines = logs.strip().splitlines()
            if lines:
                job.last_log_snippet = "\n".join(lines[-10:])
                self.save_job(job)
            return logs
        except Exception as e:
            return f"Error fetching logs: {e}"

    def download_outputs(
        self,
        job_id: str,
        output_dir: str | Path | None = None,
    ) -> str:
        """Download output files from Kaggle to local disk."""
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job '{job_id}' not found")

        account = self.account_manager.get_account(job.account_id)
        if not account:
            raise ValueError(f"Account '{job.account_id}' not found")

        target_dir = Path(output_dir or config.DEFAULT_OUTPUTS_DIR / job.kernel_slug).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)

        proc = self.account_manager.run_kaggle(
            account, ["kernels", "output", job.kernel_id, "-p", str(target_dir), "-o"],
            timeout=1800,
        )

        if proc.returncode == 0:
            job.output_dir = str(target_dir)
            job.status = JobStatus.DOWNLOADED
            self.save_job(job)
            return str(target_dir)
        else:
            err = proc.stderr or proc.stdout
            raise RuntimeError(f"Failed to download outputs: {err}")

    def run_job_sync(
        self,
        job_id: str,
        watch: bool = False,
        auto_download: bool = True,
        poll_interval: int = 10,
        timeout: int = 12 * 3600,
        push: bool = True,
    ) -> JobRecord:
        """
        Optionally push, then optionally monitor the job until completion.

        Pass push=False for a job that was already pushed: every push starts
        a NEW run on Kaggle and spends GPU quota again. `timeout` only bounds
        how long this process watches; the kernel keeps running regardless.
        """
        if push:
            job = self.push_job(job_id)
        else:
            job = self.get_job(job_id)
            if job is None:
                raise ValueError(f"Job '{job_id}' not found")
        if job.status == JobStatus.ERROR or not watch:
            return job

        start_time = time.time()
        while time.time() - start_time < timeout:
            time.sleep(poll_interval)
            job = self.check_status(job_id)
            if job.status in [JobStatus.COMPLETE, JobStatus.ERROR, JobStatus.CANCELLED]:
                break

        if auto_download and job.status == JobStatus.COMPLETE:
            try:
                self.download_outputs(job_id)
                job = self.get_job(job_id) or job
            except Exception:
                pass

        return job

    def run_job_async(
        self,
        job_id: str,
        auto_download: bool = True,
        poll_interval: int = 10,
        timeout: int = 12 * 3600,
        push: bool = False,
    ) -> threading.Thread:
        """
        Monitor (and, with push=True, first push) a job in a daemon thread.

        Defaults to push=False: callers push synchronously first so they can
        report push errors, and a second push would start a duplicate run.
        """
        thread = threading.Thread(
            target=self.run_job_sync,
            args=(job_id, True, auto_download, poll_interval, timeout, push),
            daemon=True,
        )
        thread.start()
        return thread
