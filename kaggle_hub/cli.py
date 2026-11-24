"""
Rich Terminal Command Line Interface for Kaggle Hub.
Zero external dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser
from pathlib import Path
from typing import Any

from . import __version__, config
from .account_manager import AccountManager
from .models import JobStatus, RunOptions
from .notebook_manager import NotebookManager
from .kaggle_runner import KaggleRunner
from .storage import StorageEngine


# ANSI Color formatting
class Style:
    USE_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ

    RESET = "\033[0m" if USE_COLOR else ""
    BOLD = "\033[1m" if USE_COLOR else ""
    DIM = "\033[2m" if USE_COLOR else ""
    RED = "\033[31m" if USE_COLOR else ""
    GREEN = "\033[32m" if USE_COLOR else ""
    YELLOW = "\033[33m" if USE_COLOR else ""
    BLUE = "\033[34m" if USE_COLOR else ""
    MAGENTA = "\033[35m" if USE_COLOR else ""
    CYAN = "\033[36m" if USE_COLOR else ""
    WHITE = "\033[37m" if USE_COLOR else ""


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    """Renders a clean formatted ASCII/Unicode terminal table."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            clean_cell = cell.replace(Style.BOLD, "").replace(Style.GREEN, "").replace(
                Style.RED, "").replace(Style.YELLOW, "").replace(Style.CYAN, "").replace(
                Style.DIM, "").replace(Style.RESET, "").replace(Style.MAGENTA, "")
            if idx < len(col_widths):
                col_widths[idx] = max(col_widths[idx], len(clean_cell))
            else:
                col_widths.append(len(clean_cell))

    # Header
    header_str = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep_str = "-+-".join("-" * col_widths[i] for i in range(len(headers)))

    lines = [
        f"{Style.BOLD}{header_str}{Style.RESET}",
        f"{Style.DIM}{sep_str}{Style.RESET}",
    ]

    for row in rows:
        formatted_cells = []
        for i, cell in enumerate(row):
            clean_len = len(
                cell.replace(Style.BOLD, "").replace(Style.GREEN, "").replace(
                    Style.RED, "").replace(Style.YELLOW, "").replace(Style.CYAN, "").replace(
                    Style.DIM, "").replace(Style.RESET, "").replace(Style.MAGENTA, "")
            )
            padding = " " * (col_widths[i] - clean_len)
            formatted_cells.append(cell + padding)
        lines.append(" | ".join(formatted_cells))

    return "\n".join(lines)


# -------------------------------------------------------------------------
# Account Subcommands
# -------------------------------------------------------------------------

def cmd_account_list(args: argparse.Namespace, mgr: AccountManager) -> int:
    accounts = mgr.list_accounts()
    active_id = mgr.get_active_account_id()

    if not accounts:
        print(f"{Style.YELLOW}No Kaggle accounts configured yet.{Style.RESET}")
        print("Run `python -m kaggle_hub account import` or `python -m kaggle_hub account add` to get started.")
        return 0

    headers = ["Act", "Account ID", "Label", "Username", "Auth", "Status", "Quota", "Last Used"]
    rows = []
    for acc in accounts:
        is_active = (acc.id == active_id)
        act_mark = f"{Style.GREEN}* {Style.RESET}" if is_active else "  "

        status_colored = acc.status
        if acc.status == "valid":
            status_colored = f"{Style.GREEN}valid{Style.RESET}"
        elif acc.status == "invalid":
            status_colored = f"{Style.RED}invalid{Style.RESET}"
        else:
            status_colored = f"{Style.YELLOW}{acc.status}{Style.RESET}"

        last_used = acc.last_used_at[:16].replace("T", " ") if acc.last_used_at else "Never"

        rows.append([
            act_mark,
            acc.id,
            acc.label,
            acc.username,
            acc.auth_type,
            status_colored,
            acc.quota_notes or "-",
            last_used,
        ])

    print(f"\n{Style.BOLD}Configured Kaggle Accounts ({len(accounts)}):{Style.RESET}\n")
    print(format_table(headers, rows))
    print()
    return 0


def cmd_account_add(args: argparse.Namespace, mgr: AccountManager) -> int:
    try:
        acc = mgr.add_account(
            label=args.label or args.username,
            username=args.username,
            auth_type="api_key",
            credentials={"key": args.key},
            make_active=args.active,
            quota_notes=args.quota_notes or "30h weekly GPU",
            notes=args.notes or "",
        )
        print(f"{Style.GREEN}[+] Added API Key account:{Style.RESET} {acc.username} (ID: {acc.id})")
        if args.active:
            print(f"{Style.CYAN}[*] Switched active account to: {acc.username}{Style.RESET}")
        return 0
    except Exception as e:
        print(f"{Style.RED}[!] Failed to add account:{Style.RESET} {e}", file=sys.stderr)
        return 1


def cmd_account_add_oauth(args: argparse.Namespace, mgr: AccountManager) -> int:
    cred_path = Path(args.credentials_file).resolve()
    if not cred_path.is_file():
        print(f"{Style.RED}[!] Credentials file not found:{Style.RESET} {cred_path}", file=sys.stderr)
        return 1

    try:
        with open(cred_path, "r", encoding="utf-8") as f:
            cred_data = json.load(f)

        username = cred_data.get("username")
        if not username:
            print(f"{Style.RED}[!] No 'username' field found in credentials JSON.{Style.RESET}", file=sys.stderr)
            return 1

        acc = mgr.add_account(
            label=args.label or f"{username} (OAuth)",
            username=username,
            auth_type="oauth_token",
            credentials=cred_data,
            make_active=args.active,
            quota_notes=args.quota_notes or "30h weekly GPU",
            notes=args.notes or f"Imported from {cred_path.name}",
        )
        print(f"{Style.GREEN}[+] Added OAuth account:{Style.RESET} {acc.username} (ID: {acc.id})")
        if args.active:
            print(f"{Style.CYAN}[*] Switched active account to: {acc.username}{Style.RESET}")
        return 0
    except Exception as e:
        print(f"{Style.RED}[!] Failed to import OAuth credentials:{Style.RESET} {e}", file=sys.stderr)
        return 1


def cmd_account_login(args: argparse.Namespace, mgr: AccountManager) -> int:
    print("Opening Kaggle sign-in in your browser. Whichever Kaggle account is signed in")
    print("there is the one that gets added; your current ~/.kaggle login is not touched.")
    try:
        acc = mgr.login_oauth(label=args.label, make_active=args.active, no_browser=args.no_browser)
    except Exception as e:
        print(f"{Style.RED}[!] Login failed:{Style.RESET} {e}", file=sys.stderr)
        return 1
    print(f"{Style.GREEN}[+] Added OAuth account:{Style.RESET} {acc.username} (ID: {acc.id})")
    ok, msg = mgr.test_account(acc.id)
    print(f"{Style.GREEN if ok else Style.RED}[{'PASS' if ok else 'FAIL'}] {msg}{Style.RESET}")
    return 0 if ok else 1


def cmd_account_switch(args: argparse.Namespace, mgr: AccountManager) -> int:
    try:
        acc = mgr.set_active_account(args.account)
        print(f"{Style.GREEN}[*] Active Kaggle account switched to:{Style.RESET} {acc.username} ({acc.id})")
        print(f"{Style.DIM}Environment symlink updated: {mgr.active_symlink}{Style.RESET}")
        return 0
    except Exception as e:
        print(f"{Style.RED}[!] Switch failed:{Style.RESET} {e}", file=sys.stderr)
        return 1


def cmd_account_test(args: argparse.Namespace, mgr: AccountManager) -> int:
    target = args.account
    acc = mgr.get_account(target) if target else mgr.get_active_account()

    if not acc:
        print(f"{Style.RED}[!] No account to test. Specify an account or activate one.{Style.RESET}", file=sys.stderr)
        return 1

    print(f"Testing authentication for account '{acc.username}' ({acc.id})...")
    ok, msg = mgr.test_account(acc.id)
    if ok:
        print(f"{Style.GREEN}[PASS] {msg}{Style.RESET}")
        return 0
    else:
        print(f"{Style.RED}[FAIL] {msg}{Style.RESET}")
        return 1


def cmd_account_remove(args: argparse.Namespace, mgr: AccountManager) -> int:
    ok = mgr.delete_account(args.account)
    if ok:
        print(f"{Style.GREEN}[+] Account '{args.account}' removed successfully.{Style.RESET}")
        return 0
    else:
        print(f"{Style.RED}[!] Account '{args.account}' not found.{Style.RESET}", file=sys.stderr)
        return 1


def cmd_account_import(args: argparse.Namespace, mgr: AccountManager) -> int:
    imported = mgr.import_existing_credentials()
    if not imported:
        print(f"{Style.YELLOW}No new Kaggle credentials discovered in ~/.kaggle/{Style.RESET}")
        return 0

    print(f"{Style.GREEN}[+] Successfully imported {len(imported)} account(s):{Style.RESET}")
    for acc in imported:
        print(f"  - {acc.username} ({acc.auth_type}) [ID: {acc.id}]")
    return 0


# -------------------------------------------------------------------------
# Notebook Subcommands
# -------------------------------------------------------------------------

def cmd_notebook_list(args: argparse.Namespace, nb_mgr: NotebookManager) -> int:
    target_path = Path(args.path) if args.path else None
    notebooks = nb_mgr.list_notebooks(directory=target_path)

    if not notebooks:
        print(f"{Style.YELLOW}No notebooks found in {target_path or nb_mgr.notebooks_dir}{Style.RESET}")
        return 0

    headers = ["Notebook File", "Kaggle Slug", "Cells (Code/MD)", "Size", "Derived Title"]
    rows = []
    for nb in notebooks:
        cells_str = f"{nb.total_cells} ({nb.code_cells}/{nb.markdown_cells})"
        size_kb = f"{nb.file_size_bytes / 1024:.1f} KB"
        rows.append([
            f"{Style.BOLD}{nb.name}{Style.RESET}",
            nb.slug,
            cells_str,
            size_kb,
            nb.title[:35],
        ])

    print(f"\n{Style.BOLD}Discovered Notebooks in {target_path or nb_mgr.notebooks_dir} ({len(notebooks)}):{Style.RESET}\n")
    print(format_table(headers, rows))
    print()
    return 0


def cmd_notebook_inspect(args: argparse.Namespace, nb_mgr: NotebookManager) -> int:
    try:
        info, nb_data = nb_mgr.inspect_notebook(args.notebook)
    except Exception as e:
        print(f"{Style.RED}[!] Failed to inspect notebook:{Style.RESET} {e}", file=sys.stderr)
        return 1

    print(f"\n{Style.BOLD}=== Notebook Inspection: {info.name} ==={Style.RESET}")
    print(f"  {Style.CYAN}Path:{Style.RESET}            {info.path}")
    print(f"  {Style.CYAN}Derived Title:{Style.RESET}   {info.title}")
    print(f"  {Style.CYAN}Kernel Slug:{Style.RESET}     {info.slug}")
    print(f"  {Style.CYAN}Total Cells:{Style.RESET}     {info.total_cells} (Code: {info.code_cells}, Markdown: {info.markdown_cells})")
    print(f"  {Style.CYAN}File Size:{Style.RESET}       {info.file_size_bytes / 1024:.1f} KB")
    print(f"  {Style.CYAN}Last Modified:{Style.RESET}   {info.last_modified}")

    cells = nb_data.get("cells", [])
    print(f"\n{Style.BOLD}Cell Preview (first 3 cells):{Style.RESET}")
    for idx, cell in enumerate(cells[:3]):
        ctype = cell.get("cell_type", "unknown")
        src = cell.get("source", [])
        src_text = "".join(src) if isinstance(src, list) else src
        preview = src_text.strip().splitlines()
        first_line = preview[0][:80] if preview else "(empty cell)"
        print(f"  [{idx+1}] [{ctype.upper()}]: {first_line}")
    print()
    return 0


# -------------------------------------------------------------------------
# Run Subcommands
# -------------------------------------------------------------------------

def cmd_run_push(args: argparse.Namespace, runner: KaggleRunner) -> int:
    enable_gpu = not args.no_gpu
    enable_internet = not args.no_internet
    accelerator = "None" if args.no_gpu else (args.accelerator or "NvidiaTeslaT4")

    options = RunOptions(
        notebook=args.notebook,
        account_id=args.account,
        title=args.title,
        slug=args.slug,
        accelerator=accelerator,
        enable_gpu=enable_gpu,
        enable_internet=enable_internet,
        dataset_sources=args.dataset or [],
        auto_download=args.auto_download,
        watch=args.watch,
        timeout=args.timeout or 12 * 3600,
    )

    try:
        print(f"Preparing staging environment for '{args.notebook}'...")
        job = runner.submit_job(options)
        print(f"{Style.GREEN}[+] Job staged:{Style.RESET} {job.id} (Kernel: {job.kernel_id})")

        print(f"Pushing to Kaggle (Accelerator: {job.accelerator}, GPU: {job.enable_gpu})...")
        job = runner.push_job(job.id, accelerator=job.accelerator,
                              timeout=getattr(args, "max_runtime", None))

        if job.status == JobStatus.ERROR:
            print(f"{Style.RED}[!] Push failed:{Style.RESET}\n{job.error_message}", file=sys.stderr)
            return 1

        print(f"{Style.GREEN}[+] Kernel pushed successfully! Status: {job.status}{Style.RESET}")

        if args.watch:
            print(f"Watching job {job.id}... (Ctrl+C to stop watching; execution continues on Kaggle)")
            # Already pushed above: push=False, or Kaggle starts a second run.
            job = runner.run_job_sync(job.id, watch=True, auto_download=args.auto_download,
                                      timeout=options.timeout, push=False)
            print(f"Final status: {job.status}")
            if job.status == JobStatus.COMPLETE and args.auto_download:
                print(f"{Style.GREEN}[+] Outputs downloaded to:{Style.RESET} {job.output_dir}")
        else:
            print(f"Monitor status with: `python -m kaggle_hub run status {job.id}`")

        return 0
    except Exception as e:
        print(f"{Style.RED}[!] Push execution failed:{Style.RESET} {e}", file=sys.stderr)
        return 1


def cmd_run_status(args: argparse.Namespace, runner: KaggleRunner) -> int:
    target = args.job
    if not target:
        jobs = runner.list_jobs(limit=1)
        if not jobs:
            print(f"{Style.YELLOW}No jobs recorded yet.{Style.RESET}")
            return 0
        job_id = jobs[0].id
    else:
        job = runner.get_job(target)
        if not job:
            # Maybe target is a kernel slug/id
            all_jobs = runner.list_jobs(limit=100)
            for j in all_jobs:
                if j.kernel_slug == target or j.kernel_id == target:
                    job = j
                    break
        if not job:
            print(f"{Style.RED}[!] Job '{target}' not found in local history.{Style.RESET}", file=sys.stderr)
            return 1
        job_id = job.id

    job = runner.check_status(job_id)
    print(f"\n{Style.BOLD}Job Status for: {job.id}{Style.RESET}")
    print(f"  Kernel:      {job.kernel_id}")
    print(f"  Account:     {job.username} ({job.account_id})")
    status_color = Style.GREEN if job.status in [JobStatus.COMPLETE, JobStatus.DOWNLOADED] else (
        Style.RED if job.status == JobStatus.ERROR else Style.YELLOW
    )
    print(f"  Status:      {status_color}{job.status}{Style.RESET}")
    print(f"  Accelerator: {job.accelerator}")
    print(f"  Created At:  {job.created_at}")
    print(f"  Snippet:     {job.last_log_snippet or '(no log output)'}")
    if job.output_dir:
        print(f"  Outputs:     {job.output_dir}")
    print()
    return 0


def cmd_run_logs(args: argparse.Namespace, runner: KaggleRunner) -> int:
    target = args.job
    if not target:
        jobs = runner.list_jobs(limit=1)
        if not jobs:
            print(f"{Style.YELLOW}No jobs found.{Style.RESET}")
            return 0
        job_id = jobs[0].id
    else:
        job = runner.get_job(target)
        if not job:
            print(f"{Style.RED}[!] Job '{target}' not found.{Style.RESET}", file=sys.stderr)
            return 1
        job_id = job.id

    logs = runner.get_logs(job_id, follow=args.follow)
    print(logs)
    return 0


def cmd_run_download(args: argparse.Namespace, runner: KaggleRunner) -> int:
    target = args.job
    if not target:
        jobs = runner.list_jobs(limit=1)
        if not jobs:
            print(f"{Style.YELLOW}No jobs found.{Style.RESET}")
            return 0
        job_id = jobs[0].id
    else:
        job = runner.get_job(target)
        if not job:
            print(f"{Style.RED}[!] Job '{target}' not found.{Style.RESET}", file=sys.stderr)
            return 1
        job_id = job.id

    try:
        out_dir = runner.download_outputs(job_id, output_dir=args.path)
        print(f"{Style.GREEN}[+] Artifacts downloaded to:{Style.RESET} {out_dir}")
        return 0
    except Exception as e:
        print(f"{Style.RED}[!] Download failed:{Style.RESET} {e}", file=sys.stderr)
        return 1


def cmd_run_list(args: argparse.Namespace, runner: KaggleRunner) -> int:
    limit = 100 if args.all else (args.limit or 20)
    jobs = runner.list_jobs(limit=limit)

    if not jobs:
        print(f"{Style.YELLOW}No execution jobs recorded yet.{Style.RESET}")
        return 0

    headers = ["Job ID", "Kernel / Title", "Account", "Accelerator", "Status", "Started", "Outputs"]
    rows = []
    for j in jobs:
        st_color = Style.GREEN if j.status in [JobStatus.COMPLETE, JobStatus.DOWNLOADED] else (
            Style.RED if j.status == JobStatus.ERROR else Style.YELLOW
        )
        started_str = j.started_at[:16].replace("T", " ") if j.started_at else "-"
        outputs_str = "Yes" if j.output_dir else "-"
        rows.append([
            j.id,
            j.kernel_slug[:30],
            j.username,
            j.accelerator,
            f"{st_color}{j.status}{Style.RESET}",
            started_str,
            outputs_str,
        ])

    print(f"\n{Style.BOLD}Recent Kaggle Hub Jobs ({len(jobs)}):{Style.RESET}\n")
    print(format_table(headers, rows))
    print()
    return 0


# -------------------------------------------------------------------------
# Serve Subcommand
# -------------------------------------------------------------------------

def cmd_serve(args: argparse.Namespace) -> int:
    from .server.http_server import run_server

    host = args.host or "127.0.0.1"
    port = args.port or 8080

    print(f"{Style.BOLD}Starting Kaggle Hub Web Dashboard on http://{host}:{port}{Style.RESET}")
    return run_server(host=host, port=port, open_browser=args.open)


# -------------------------------------------------------------------------
# Main Argument Parser
# -------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kaggle_hub",
        description=f"Kaggle Hub v{__version__} - Multi-Account Manager & Headless Notebook Runner",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Command category")

    # 1. account
    p_acc = subparsers.add_parser("account", help="Multi-account credential management")
    acc_sub = p_acc.add_subparsers(dest="subcommand", help="Account action")

    # account list
    acc_sub.add_parser("list", help="List all configured accounts")

    # account add
    p_acc_add = acc_sub.add_parser("add", help="Add an API key account")
    p_acc_add.add_argument("--username", "-u", required=True, help="Kaggle username")
    p_acc_add.add_argument("--key", "-k", required=True, help="Kaggle API Key")
    p_acc_add.add_argument("--label", "-l", help="Descriptive label (e.g. 'CoAuthor GPU')")
    p_acc_add.add_argument("--active", action="store_true", help="Set as active immediately")
    p_acc_add.add_argument("--quota-notes", help="Quota notes (e.g. '30h weekly GPU')")
    p_acc_add.add_argument("--notes", help="Optional notes")

    # account add-oauth
    p_acc_oauth = acc_sub.add_parser("add-oauth", help="Add an account from credentials.json")
    p_acc_oauth.add_argument("--credentials-file", "-f", required=True, help="Path to credentials.json")
    p_acc_oauth.add_argument("--label", "-l", help="Descriptive label")
    p_acc_oauth.add_argument("--active", action="store_true", help="Set as active immediately")
    p_acc_oauth.add_argument("--quota-notes", help="Quota notes")
    p_acc_oauth.add_argument("--notes", help="Optional notes")

    # account switch
    p_acc_login = acc_sub.add_parser(
        "login", help="Add an account via browser sign-in (sandboxed `kaggle auth login`)")
    p_acc_login.add_argument("--label", "-l", help="Descriptive label")
    p_acc_login.add_argument("--active", action="store_true", help="Set as active immediately")
    p_acc_login.add_argument("--no-browser", action="store_true",
                             help="Print the sign-in URL and paste the code instead")

    p_acc_sw = acc_sub.add_parser("switch", help="Switch active Kaggle account")
    p_acc_sw.add_argument("account", help="Account ID or Username to activate")

    # account test
    p_acc_test = acc_sub.add_parser("test", help="Test credentials with Kaggle CLI")
    p_acc_test.add_argument("account", nargs="?", help="Account ID or Username (defaults to active)")

    # account remove
    p_acc_rm = acc_sub.add_parser("remove", help="Remove an account")
    p_acc_rm.add_argument("account", help="Account ID or Username to remove")

    # account import
    acc_sub.add_parser("import", help="Auto-import ~/.kaggle/kaggle.json or credentials.json")

    # 2. notebook
    p_nb = subparsers.add_parser("notebook", help="Local notebook discovery & inspection")
    nb_sub = p_nb.add_subparsers(dest="subcommand", help="Notebook action")

    # notebook list
    p_nb_list = nb_sub.add_parser("list", help="List discovered notebooks")
    p_nb_list.add_argument("--path", "-p", help="Directory to scan (defaults to ../notebooks)")

    # notebook inspect
    p_nb_insp = nb_sub.add_parser("inspect", help="Inspect notebook structure & metadata")
    p_nb_insp.add_argument("notebook", help="Notebook filename or path")

    # 3. run
    p_run = subparsers.add_parser("run", help="Execution, push, status, logs & downloads")
    run_sub = p_run.add_subparsers(dest="subcommand", help="Run action")

    # run push
    p_run_push = run_sub.add_parser("push", help="Push notebook to Kaggle for execution")
    p_run_push.add_argument("notebook", help="Notebook filename or path")
    p_run_push.add_argument("--account", "-a", help="Account ID or Username (defaults to active)")
    p_run_push.add_argument("--title", "-t", help="Kernel title on Kaggle")
    p_run_push.add_argument("--slug", "-s", help="Kernel slug")
    p_run_push.add_argument(
        "--accelerator",
        choices=["NvidiaTeslaT4", "NvidiaTeslaP100", "TpuV5E8", "None"],
        default="NvidiaTeslaT4",
        help="Hardware accelerator (default: NvidiaTeslaT4)",
    )
    p_run_push.add_argument("--no-gpu", action="store_true", help="Run on standard CPU")
    p_run_push.add_argument("--no-internet", action="store_true", help="Disable internet access")
    p_run_push.add_argument(
        "--max-runtime", type=int, default=None,
        help="Kill the kernel on Kaggle after this many seconds (default: Kaggle's 12h limit)")
    p_run_push.add_argument(
        "--timeout", type=int, default=12 * 3600,
        help="How long --watch keeps polling, in seconds (the kernel is unaffected)")
    p_run_push.add_argument("--dataset", action="append", help="Attach dataset owner/slug")
    p_run_push.add_argument("--watch", "-w", action="store_true", help="Stream logs until completion")
    p_run_push.add_argument("--auto-download", action="store_true", default=True, help="Auto download outputs")
    p_run_push.add_argument("--no-download", dest="auto_download", action="store_false",
                            help="Do not download outputs after --watch completes")

    # run status
    p_run_st = run_sub.add_parser("status", help="Check status of a job")
    p_run_st.add_argument("job", nargs="?", help="Job ID or Kernel Slug (defaults to latest)")

    # run logs
    p_run_lg = run_sub.add_parser("logs", help="Fetch or stream logs for a job")
    p_run_lg.add_argument("job", nargs="?", help="Job ID (defaults to latest)")
    p_run_lg.add_argument("--follow", "-f", action="store_true", help="Follow logs")

    # run download
    p_run_dl = run_sub.add_parser("download", help="Download output artifacts")
    p_run_dl.add_argument("job", nargs="?", help="Job ID (defaults to latest)")
    p_run_dl.add_argument("--path", "-p", help="Custom output directory")

    # run list
    p_run_ls = run_sub.add_parser("list", help="List past execution jobs")
    p_run_ls.add_argument("--all", action="store_true", help="Show all jobs")
    p_run_ls.add_argument("--limit", "-n", type=int, default=20, help="Max jobs to show")

    # 4. serve
    p_srv = subparsers.add_parser("serve", help="Launch Modern Web UI Dashboard")
    p_srv.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    p_srv.add_argument("--port", type=int, default=8080, help="HTTP port (default: 8080)")
    p_srv.add_argument("--open", action="store_true", help="Open browser on start")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    storage = StorageEngine()
    acc_mgr = AccountManager(storage=storage)
    nb_mgr = NotebookManager()
    runner = KaggleRunner(account_manager=acc_mgr, notebook_manager=nb_mgr, storage=storage)

    if args.command == "account":
        if not args.subcommand or args.subcommand == "list":
            return cmd_account_list(args, acc_mgr)
        elif args.subcommand == "add":
            return cmd_account_add(args, acc_mgr)
        elif args.subcommand == "add-oauth":
            return cmd_account_add_oauth(args, acc_mgr)
        elif args.subcommand == "login":
            return cmd_account_login(args, acc_mgr)
        elif args.subcommand == "switch":
            return cmd_account_switch(args, acc_mgr)
        elif args.subcommand == "test":
            return cmd_account_test(args, acc_mgr)
        elif args.subcommand == "remove":
            return cmd_account_remove(args, acc_mgr)
        elif args.subcommand == "import":
            return cmd_account_import(args, acc_mgr)
        else:
            print(f"Unknown account subcommand: {args.subcommand}", file=sys.stderr)
            return 1

    elif args.command == "notebook":
        if not args.subcommand or args.subcommand == "list":
            return cmd_notebook_list(args, nb_mgr)
        elif args.subcommand == "inspect":
            return cmd_notebook_inspect(args, nb_mgr)
        else:
            print(f"Unknown notebook subcommand: {args.subcommand}", file=sys.stderr)
            return 1

    elif args.command == "run":
        if not args.subcommand or args.subcommand == "list":
            return cmd_run_list(args, runner)
        elif args.subcommand == "push":
            return cmd_run_push(args, runner)
        elif args.subcommand == "status":
            return cmd_run_status(args, runner)
        elif args.subcommand == "logs":
            return cmd_run_logs(args, runner)
        elif args.subcommand == "download":
            return cmd_run_download(args, runner)
        else:
            print(f"Unknown run subcommand: {args.subcommand}", file=sys.stderr)
            return 1

    elif args.command == "serve":
        return cmd_serve(args)

    return 0
