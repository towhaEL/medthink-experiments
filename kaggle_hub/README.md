# Kaggle Hub 🚀

**Multi-Account Kaggle Manager & Headless GPU Notebook Runner**

Kaggle Hub is a self-contained system developed specifically for research teams and multi-author workflows. It allows you to store and switch between multiple Kaggle accounts, distribute experiments across your co-authors' weekly GPU quotas (e.g., 30 hours per account), trigger headless runs on Kaggle GPU directly from your local terminal or browser, stream logs in real-time, and download output weights/metrics automatically.

---

## Key Features

- **Zero External Dependencies**: Built with 100% Python 3 standard library (`http.server.ThreadingHTTPServer`, `sqlite3`/JSON storage, `subprocess`, `urllib`). No virtualenv activation or broken pip packages required!
- **Multi-Account Credential Isolation**: Secure sandboxing in `~/.kaggle_hub/envs/<account_id>/` (enforced `0700` directory and `0600` file permissions). Each execution isolates `KAGGLE_CONFIG_DIR`, preventing concurrent runs from colliding.
- **Dual Interface**:
  - **Modern Web UI Dashboard**: Responsive SPA (Single Page App) on `http://127.0.0.1:8080` with dark mode, live status badges, log streaming viewer, and one-click execution.
  - **Rich Terminal CLI**: Fast ANSI-formatted commands (`python3 -m kaggle_hub ...`).
- **Notebook Execution Engine**: Automatically detects notebooks in `../notebooks/` (`00_data_prep.ipynb` through `05_results_report.ipynb`), extracts titles and stats, generates Kaggle `kernel-metadata.json`, and handles the full lifecycle (`QUEUED` → `RUNNING` → `COMPLETE` / `ERROR`).
- **Artifact Auto-Downloader**: Syncs output models, checkpoints, and evaluation tables directly into `../outputs/<slug>/`.

---

## Quickstart

### 1. Launch the Web UI Dashboard
From the project directory:
```bash
./kaggle_hub/run.sh
# or
python3 -m kaggle_hub serve --port 8080
```
Open **[http://127.0.0.1:8080](http://127.0.0.1:8080)** in your browser!

### 2. Using the CLI
```bash
# List all configured Kaggle accounts
./kaggle_hub/run.sh account list

# List local notebooks available to run
./kaggle_hub/run.sh notebook list

# Inspect notebook metadata and cells
./kaggle_hub/run.sh notebook inspect 00_data_prep.ipynb

# Launch notebook on Kaggle GPU (runs up to Kaggle's 12h session limit)
./kaggle_hub/run.sh run push 00_data_prep.ipynb --watch

# Run on a specific co-author account; cap the run at 2h on Kaggle's side
./kaggle_hub/run.sh run push 02_train_closed_end.ipynb --account codegiveway --max-runtime 7200

# Check running jobs status
./kaggle_hub/run.sh run list

# View logs for a running or completed job
./kaggle_hub/run.sh run logs <job_id_or_kernel_slug>

# Download outputs for a job
./kaggle_hub/run.sh run download <job_id_or_kernel_slug>
```

---

## Multi-Account Setup

### Recommended: browser sign-in (OAuth, Kaggle CLI ≥ 2.x)
```bash
python3 -m kaggle_hub account login --label "CoAuthor B"
```
Opens Kaggle's sign-in page in your normal browser. **Whichever Kaggle account is
signed in there is the one that gets added**, so switch the browser to the
co-author's Kaggle account first. The login runs in a sandbox: your existing
`~/.kaggle` login is never touched, and an account that is already configured is
refused. Use `--no-browser` to get a URL and paste the code by hand.

`account test <user>` checks that the account authenticates **as itself** (via
`kaggle config view`), not merely that some login works.

> **How isolation works.** The Kaggle SDK reads OAuth tokens from the hard-coded
> path `~/.kaggle/credentials.json` and ignores `KAGGLE_CONFIG_DIR`. Every command
> for an account therefore runs with `HOME=~/.kaggle_hub/envs/<id>/home`, whose
> `.kaggle` links to that account's env dir. Token refreshes stay in the sandbox
> and are synced back into `accounts.json`.

### Alternative: API key (`kaggle.json`)
1. Ask your co-author to go to: **[kaggle.com](https://www.kaggle.com)** → Click Profile Avatar → **Settings**.
2. Scroll to the **API** section:
   - Click **Create New Token**. This downloads a file called `kaggle.json` containing:
     ```json
     {"username": "coauthor_username", "key": "32_character_hex_key"}
     ```
3. Add the account via the **Web UI** or the **CLI**:

#### Via Web UI:
- Open `http://localhost:8080` → Navigate to **Account Management** tab.
- Click **Add Account** → Choose either **API Key** or **Upload JSON File**.
- Click **Test Auth** to verify immediate connectivity to Kaggle's servers.

#### Via CLI:
```bash
# Add with username and key
python3 -m kaggle_hub account add --label "CoAuthor A" --username "alice" --key "abcdef0123456789"

# Or import an existing ~/.kaggle/credentials.json or kaggle.json
python3 -m kaggle_hub account import

# Switch the globally active account
python3 -m kaggle_hub account switch alice
```

---

## Directory Structure

```
kaggle_hub/
├── __init__.py               # Package metadata and exports
├── __main__.py               # CLI entrypoint
├── config.py                 # Paths, constants, GPU accelerator mappings
├── models.py                 # Data models (Account, NotebookInfo, JobRecord)
├── storage.py                # Thread-safe atomic JSON store (0600 permissions)
├── account_manager.py        # Multi-account sandboxing and credential switching
├── notebook_manager.py       # Notebook scanner and kernel-metadata generator
├── kaggle_runner.py          # Kaggle CLI wrapper, lifecycle tracker, output downloader
├── cli.py                    # Terminal CLI with ANSI tables and subcommands
├── run.sh                    # Unified launcher script
├── server/
│   ├── http_server.py        # ThreadingHTTPServer with path routing and CORS
│   ├── api_handlers.py       # REST API dispatchers
│   └── static/               # Web UI Dashboard
│       ├── index.html        # SPA layout with tabs and modals
│       ├── app.js            # Reactive JavaScript client
│       └── styles.css        # Modern slate dark theme
└── tests/                    # 50 unit and integration tests (100% pass)
    ├── test_core.py
    ├── test_server_and_cli.py
    ├── test_verification.py
    └── test_server_e2e_verification.py
```

---

## Running Tests

To run the complete 50-test verification suite:
```bash
python3 -m unittest discover kaggle_hub/tests
```
