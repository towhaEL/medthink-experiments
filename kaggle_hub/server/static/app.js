/**
 * Kaggle Hub web panel. Vanilla JS, no dependencies.
 */

const AUTO_ACCOUNT = "__auto__";
const ACTIVE_STATES = ["QUEUED", "RUNNING", "STAGING", "SUBMITTED"];

const S = {
  tab: "accounts",
  accounts: [],
  activeId: null,
  notebooks: [],
  selectedNb: null,
  jobs: [],
  logJobId: null,
  manualAuth: "api_key",
  timers: {},
};

const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------ helpers
function esc(v) {
  return String(v ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

function icon(name, cls = "icon") {
  return `<svg class="${cls}"><use href="#i-${name}"/></svg>`;
}

async function api(path, options = {}) {
  const opts = { ...options };
  if (opts.body && typeof opts.body !== "string") {
    opts.headers = { "Content-Type": "application/json", ...(opts.headers || {}) };
    opts.body = JSON.stringify(opts.body);
  }
  const res = await fetch(path, opts);
  let data = {};
  try { data = await res.json(); } catch (_) { /* empty body */ }
  if (!res.ok) throw new Error(data.error || `${res.status} ${res.statusText}`);
  return data;
}

function toast(message, type = "info") {
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = message;
  $("toasts").appendChild(el);
  setTimeout(() => { el.classList.add("leaving"); setTimeout(() => el.remove(), 260); },
             type === "error" ? 7000 : 4000);
}

function fmtTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (isNaN(d)) return "—";
  const today = new Date().toDateString() === d.toDateString();
  return today
    ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : d.toLocaleDateString([], { month: "short", day: "numeric" }) + " " +
      d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function fmtDuration(startIso, endIso) {
  if (!startIso) return "—";
  const start = new Date(startIso);
  const end = endIso ? new Date(endIso) : new Date();
  let s = Math.max(0, Math.round((end - start) / 1000));
  const h = Math.floor(s / 3600); s -= h * 3600;
  const m = Math.floor(s / 60);
  return h ? `${h}h ${String(m).padStart(2, "0")}m` : `${m}m`;
}

function quotaHours(acc) {
  const m = /(\d+(?:\.\d+)?)\s*h/i.exec(acc.quota_notes || "");
  return m ? parseFloat(m[1]) : 30;
}

function hoursLeft(acc) {
  return Math.max(0, quotaHours(acc) - ((acc.usage && acc.usage.gpu_hours) || 0));
}

function statusBadge(status) {
  const map = {
    RUNNING: "info live", QUEUED: "violet live", STAGING: "violet", SUBMITTED: "violet",
    COMPLETE: "ok", DOWNLOADED: "ok", ERROR: "bad", CANCELLED: "warn",
  };
  return `<span class="badge ${map[status] || ""}">${esc((status || "unknown").toLowerCase())}</span>`;
}

// --------------------------------------------------------------- navigation
function switchTab(tab) {
  S.tab = tab;
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  document.querySelectorAll(".tab-pane").forEach((p) => p.classList.toggle("hidden", p.id !== `tab-${tab}`));
  if (location.hash !== "#" + tab) history.replaceState(null, "", "#" + tab);
  if (tab === "jobs") loadJobs();
  if (tab === "launch") renderAccountSelect();
}

function openModal(id) { $(id).classList.remove("hidden"); }
function closeModal(id) {
  const el = $(id);
  if (!el || el.dataset.busy === "1") return;
  el.classList.add("hidden");
  if (id === "logs-modal") { S.logJobId = null; clearInterval(S.timers.logs); }
  if (id === "login-modal") resetLoginModal();
}

// ------------------------------------------------------------------- status
async function loadStatus() {
  try {
    const d = await api("/api/status");
    const cli = d.kaggle_cli || {};
    $("cli-pill").innerHTML = cli.installed
      ? `<span class="dot ok"></span><span>${esc(cli.version)}</span>`
      : `<span class="dot bad"></span><span>Kaggle CLI not found</span>`;
    $("active-account-text").textContent = d.active_account ? "@" + d.active_account.username : "none";
  } catch (e) {
    $("cli-pill").innerHTML = `<span class="dot bad"></span><span>Server unreachable</span>`;
  }
}

// ----------------------------------------------------------------- accounts
async function loadAccounts() {
  try {
    const d = await api("/api/accounts");
    S.accounts = d.accounts || [];
    S.activeId = d.active_account_id;
    renderAccounts();
    renderAccountSelect();
    renderJobsFilter();
  } catch (e) {
    toast("Could not load accounts: " + e.message, "error");
  }
}

function renderAccounts() {
  const accs = S.accounts;
  const used = accs.reduce((t, a) => t + ((a.usage && a.usage.gpu_hours) || 0), 0);
  const total = accs.reduce((t, a) => t + quotaHours(a), 0);
  const running = accs.reduce((t, a) => t + ((a.usage && a.usage.active_jobs) || 0), 0);
  const valid = accs.filter((a) => a.status === "valid").length;

  $("stat-accounts").textContent = accs.length;
  $("stat-accounts-note").textContent = accs.length ? `${valid} verified` : "add one to start";
  $("stat-used").textContent = used.toFixed(1) + " h";
  $("stat-left").textContent = Math.max(0, total - used).toFixed(1) + " h";
  $("stat-left-note").textContent = `of ${total.toFixed(0)} h weekly`;
  $("stat-running").textContent = running;

  const grid = $("accounts-grid");
  if (!accs.length) {
    grid.innerHTML = `
      <div class="card empty" style="grid-column:1/-1">
        <h3>No Kaggle accounts yet</h3>
        <p>Sign in through the browser, or import the login already in ~/.kaggle.</p>
        <div class="actions">
          <button class="btn" data-action="import">${icon("import")}Import ~/.kaggle</button>
          <button class="btn btn-primary" data-open="login-modal">${icon("login")}Sign in with browser</button>
        </div>
      </div>`;
    return;
  }

  grid.innerHTML = accs.map((a) => {
    const active = a.id === S.activeId;
    const usedH = (a.usage && a.usage.gpu_hours) || 0;
    const quota = quotaHours(a);
    const pct = Math.min(100, (usedH / quota) * 100);
    const meterCls = pct >= 90 ? "bad" : pct >= 70 ? "warn" : "";
    const st = a.status || "untested";
    const stCls = st === "valid" ? "ok" : st === "invalid" ? "bad" : "";
    const jobs = (a.usage && a.usage.active_jobs) || 0;
    return `
      <article class="card acct ${active ? "is-active" : ""}">
        <div class="acct-head">
          <div class="avatar">${esc((a.username || "?")[0].toUpperCase())}</div>
          <div style="min-width:0">
            <div class="acct-name" title="${esc(a.label || a.username)}">${esc(a.label || a.username)}</div>
            <div class="acct-user">@${esc(a.username)}</div>
          </div>
          <div class="acct-badges">
            <span class="badge ${stCls}">${esc(st)}</span>
            ${active ? '<span class="badge info">active</span>' : ""}
          </div>
        </div>

        <div>
          <div class="meter-row"><span class="muted">GPU · last 7 days</span><strong>${usedH.toFixed(1)} / ${quota} h</strong></div>
          <div class="meter ${meterCls}"><span style="width:${pct.toFixed(1)}%"></span></div>
        </div>

        <dl class="meta">
          <dt>Sign-in</dt><dd>${a.auth_type === "oauth_token" ? "Browser (OAuth)" : "API key"}</dd>
          <dt>Running jobs</dt><dd>${jobs}</dd>
          <dt>Last checked</dt><dd>${esc(fmtTime(a.last_used_at))}</dd>
          ${a.notes ? `<dt>Notes</dt><dd title="${esc(a.notes)}">${esc(a.notes)}</dd>` : ""}
        </dl>

        <div class="acct-foot">
          ${active ? "" : `<button class="btn btn-sm" data-acc="switch" data-id="${esc(a.id)}">${icon("star")}Make active</button>`}
          <button class="btn btn-sm" data-acc="test" data-id="${esc(a.id)}">${icon("check")}Test</button>
          <button class="btn btn-sm" data-acc="launch" data-id="${esc(a.id)}">${icon("rocket")}Launch</button>
          <span class="spacer"></span>
          <button class="btn btn-sm btn-ghost btn-danger" data-acc="delete" data-id="${esc(a.id)}" title="Remove account">${icon("trash")}</button>
        </div>
      </article>`;
  }).join("");
}

async function accountAction(action, id, btn) {
  const acc = S.accounts.find((a) => a.id === id);
  if (!acc) return;
  if (action === "launch") {
    switchTab("launch");
    $("f-account").value = id;
    updateAccountHint();
    return;
  }
  if (action === "delete") {
    if (!confirm(`Remove @${acc.username} from Kaggle Hub?\n\nThis deletes its stored sign-in on this machine only; the Kaggle account itself is not affected.`)) return;
  }
  const old = btn ? btn.innerHTML : "";
  if (btn) { btn.disabled = true; if (action === "test") btn.textContent = "Testing…"; }
  try {
    if (action === "switch") {
      await api(`/api/accounts/${encodeURIComponent(id)}/switch`, { method: "POST" });
      toast(`Active account: @${acc.username}`, "success");
    } else if (action === "test") {
      const d = await api(`/api/accounts/${encodeURIComponent(id)}/test`, { method: "POST" });
      toast(d.message, d.valid ? "success" : "error");
    } else if (action === "delete") {
      await api(`/api/accounts/${encodeURIComponent(id)}`, { method: "DELETE" });
      toast(`Removed @${acc.username}`, "info");
    }
  } catch (e) {
    toast(e.message, "error");
  } finally {
    if (btn) { btn.disabled = false; btn.innerHTML = old; }
    loadAccounts();
    loadStatus();
  }
}

async function startLogin() {
  const modal = $("login-modal");
  const btn = $("btn-login-start");
  modal.dataset.busy = "1";
  btn.disabled = true;
  setLoginView("wait");
  try {
    const d = await api("/api/accounts/login", {
      method: "POST",
      body: { label: $("login-label").value.trim(), make_active: $("login-active").checked },
    });
    toast(`Added @${d.account.username}. ${d.message}`, d.valid ? "success" : "error");
    modal.dataset.busy = "0";
    closeModal("login-modal");
    $("login-label").value = "";
  } catch (e) {
    toast("Sign-in failed: " + e.message, "error");
  } finally {
    modal.dataset.busy = "0";
    btn.disabled = false;
    setLoginView("form");
    loadAccounts();
    loadStatus();
  }
}

// Sign in with a link: the server starts `kaggle auth login --no-launch-browser`
// and hands back its link; the user opens it in any browser window (a private
// one picks a different Kaggle account) and pastes the verification code back.
let loginSession = null;

function setLoginView(view) {
  $("login-form-view").classList.toggle("hidden", view !== "form");
  $("login-wait-view").classList.toggle("hidden", view !== "wait");
  $("login-link-view").classList.toggle("hidden", view !== "link");
  $("btn-login-start").classList.toggle("hidden", view !== "form");
  $("btn-login-link").classList.toggle("hidden", view !== "form");
  $("btn-login-finish").classList.toggle("hidden", view !== "link");
}

function resetLoginModal() {
  if (loginSession) {
    api("/api/accounts/login/cancel", { method: "POST", body: { session: loginSession } }).catch(() => {});
    loginSession = null;
  }
  $("login-url").value = "";
  $("login-code").value = "";
  setLoginView("form");
}

async function startLinkLogin() {
  const btn = $("btn-login-link");
  btn.disabled = true;
  try {
    const d = await api("/api/accounts/login/start", {
      method: "POST",
      body: { label: $("login-label").value.trim(), make_active: $("login-active").checked },
    });
    loginSession = d.session;
    $("login-url").value = d.url;
    setLoginView("link");
    $("login-code").focus();
  } catch (e) {
    toast("Could not start sign-in: " + e.message, "error");
  } finally {
    btn.disabled = false;
  }
}

async function copyLoginUrl() {
  const url = $("login-url").value;
  try {
    await navigator.clipboard.writeText(url);
    toast("Link copied. Open it in a private window.", "success");
  } catch {
    $("login-url").select();                // clipboard API blocked: let the user copy
    toast("Press Ctrl+C to copy the selected link.", "info");
  }
}

async function finishLinkLogin() {
  const code = $("login-code").value.trim();
  if (!code) { toast("Paste the verification code from Kaggle first.", "error"); return; }
  const modal = $("login-modal");
  const btn = $("btn-login-finish");
  modal.dataset.busy = "1";
  btn.disabled = true;
  const session = loginSession;
  loginSession = null;                      // the server consumes it, success or not
  try {
    const d = await api("/api/accounts/login/finish", { method: "POST", body: { session, code } });
    toast(`Added @${d.account.username}. ${d.message}`, d.valid ? "success" : "error");
    modal.dataset.busy = "0";
    closeModal("login-modal");
    $("login-label").value = "";
  } catch (e) {
    toast("Sign-in failed: " + e.message, "error");
    setLoginView("form");
  } finally {
    modal.dataset.busy = "0";
    btn.disabled = false;
    loadAccounts();
    loadStatus();
  }
}

function setManualAuth(type) {
  S.manualAuth = type;
  document.querySelectorAll("#manual-modal .seg button")
    .forEach((b) => b.classList.toggle("active", b.dataset.auth === type));
  $("m-key-field").classList.toggle("hidden", type !== "api_key");
  $("m-json-field").classList.toggle("hidden", type === "api_key");
}

async function submitManual(ev) {
  ev.preventDefault();
  const username = $("m-username").value.trim();
  const payload = {
    username,
    label: $("m-label").value.trim() || username,
    auth_type: S.manualAuth,
    quota_notes: $("m-quota").value.trim() || "30h weekly GPU",
    notes: $("m-notes").value.trim(),
    make_active: $("m-active").checked,
  };
  if (S.manualAuth === "api_key") {
    const key = $("m-key").value.trim();
    if (!key) return toast("Enter the API key", "error");
    payload.credentials = { key };
  } else {
    try { payload.credentials = JSON.parse($("m-json").value.trim()); }
    catch (_) { return toast("credentials.json is not valid JSON", "error"); }
  }
  try {
    await api("/api/accounts", { method: "POST", body: payload });
    toast(`Added @${username}. Use Test to verify it.`, "success");
    $("manual-form").reset();
    setManualAuth("api_key");
    closeModal("manual-modal");
    loadAccounts();
    loadStatus();
  } catch (e) {
    toast(e.message, "error");
  }
}

async function importCredentials() {
  try {
    const d = await api("/api/accounts/import", { method: "POST" });
    toast(d.imported_count ? `Imported ${d.imported_count} account(s)` : "Nothing new in ~/.kaggle",
          d.imported_count ? "success" : "info");
    loadAccounts();
    loadStatus();
  } catch (e) {
    toast("Import failed: " + e.message, "error");
  }
}

// ------------------------------------------------------------------- launch
async function loadNotebooks() {
  try {
    const d = await api("/api/notebooks");
    S.notebooks = d.notebooks || [];
    $("nb-dir").textContent = d.directory || "";
    renderNotebooks();
  } catch (e) {
    toast("Could not load notebooks: " + e.message, "error");
  }
}

function renderNotebooks() {
  const sel = $("f-notebook");
  sel.innerHTML = `<option value="" disabled ${S.selectedNb ? "" : "selected"}>Choose a notebook…</option>` +
    S.notebooks.map((n) => `<option value="${esc(n.name)}" ${S.selectedNb === n.name ? "selected" : ""}>${esc(n.name)}</option>`).join("");

  const list = $("notebooks-list");
  if (!S.notebooks.length) {
    list.innerHTML = `<div class="card empty"><h3>No notebooks found</h3><p>Add .ipynb files to the notebooks folder and rescan.</p></div>`;
    return;
  }
  list.innerHTML = S.notebooks.map((n) => {
    const num = (/^(\d+)/.exec(n.name) || [, "nb"])[1];
    return `
      <button type="button" class="card nb ${S.selectedNb === n.name ? "selected" : ""}" data-nb="${esc(n.name)}">
        <span class="nb-num">${esc(num)}</span>
        <span class="nb-body">
          <span class="nb-title">${esc(n.title)}</span>
          <span class="nb-sub mono">${esc(n.name)} · ${n.code_cells} code cells · ${(n.file_size_bytes / 1024).toFixed(0)} KB</span>
        </span>
      </button>`;
  }).join("");
}

function selectNotebook(name) {
  const nb = S.notebooks.find((n) => n.name === name);
  if (!nb) return;
  S.selectedNb = name;
  $("f-notebook").value = name;
  $("f-title").value = nb.title;
  $("f-slug").value = nb.slug;
  $("launch-heading").textContent = nb.title;
  document.querySelectorAll(".nb").forEach((el) => el.classList.toggle("selected", el.dataset.nb === name));
}

function bestAccount() {
  const usable = S.accounts.filter((a) => a.status !== "invalid");
  return usable.sort((a, b) => hoursLeft(b) - hoursLeft(a))[0] || null;
}

function renderAccountSelect() {
  const sel = $("f-account");
  const prev = sel.value;
  const best = bestAccount();
  sel.innerHTML =
    (S.accounts.length > 1 && best
      ? `<option value="${AUTO_ACCOUNT}">Auto: most GPU hours left (@${esc(best.username)})</option>` : "") +
    S.accounts.map((a) => {
      const bad = a.status === "invalid";
      return `<option value="${esc(a.id)}" ${bad ? "disabled" : ""}>@${esc(a.username)} · ${hoursLeft(a).toFixed(1)} h left${a.id === S.activeId ? " · active" : ""}${bad ? " · sign-in invalid" : ""}</option>`;
    }).join("");
  if (prev && [...sel.options].some((o) => o.value === prev)) sel.value = prev;
  else if (S.accounts.length > 1 && best) sel.value = AUTO_ACCOUNT;
  else if (S.activeId) sel.value = S.activeId;
  updateAccountHint();
}

function resolveAccount() {
  const v = $("f-account").value;
  if (v === AUTO_ACCOUNT) return bestAccount();
  return S.accounts.find((a) => a.id === v) || null;
}

function updateAccountHint() {
  const acc = resolveAccount();
  $("f-account-hint").textContent = acc
    ? `Runs as @${acc.username}: ${hoursLeft(acc).toFixed(1)} of ${quotaHours(acc)} GPU hours left this week (estimate).`
    : "Add an account first.";
}

async function submitLaunch(ev) {
  ev.preventDefault();
  const notebook = $("f-notebook").value;
  if (!notebook) return toast("Choose a notebook first", "error");
  const acc = resolveAccount();
  if (!acc) return toast("Choose an account", "error");

  const accel = $("f-accel").value;
  const maxRun = $("f-maxrun").value.trim();
  const btn = $("btn-launch");
  const old = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span>Pushing to Kaggle as @${esc(acc.username)}…`;
  try {
    const d = await api("/api/runs", {
      method: "POST",
      body: {
        notebook,
        account_id: acc.id,
        accelerator: accel,
        enable_gpu: accel !== "None" && !accel.startsWith("Tpu"),
        enable_internet: $("f-internet").checked,
        auto_download: $("f-download").checked,
        title: $("f-title").value.trim() || undefined,
        slug: $("f-slug").value.trim() || undefined,
        dataset_sources: $("f-datasets").value.trim(),
        max_runtime_hours: maxRun || null,
      },
    });
    if (d.job.status === "ERROR") {
      toast("Kaggle rejected the push: " + (d.job.error_message || "unknown error"), "error");
    } else {
      toast(`Pushed ${d.job.kernel_id}`, "success");
      switchTab("jobs");
    }
    loadJobs();
    loadAccounts();
  } catch (e) {
    toast("Push failed: " + e.message, "error");
  } finally {
    btn.disabled = false;
    btn.innerHTML = old;
  }
}

// --------------------------------------------------------------------- jobs
function renderJobsFilter() {
  const sel = $("jobs-filter");
  const prev = sel.value;
  sel.innerHTML = `<option value="">All accounts</option>` +
    S.accounts.map((a) => `<option value="${esc(a.id)}">@${esc(a.username)}</option>`).join("");
  sel.value = prev;
}

async function loadJobs() {
  try {
    const filter = $("jobs-filter").value;
    const d = await api("/api/runs?limit=100" + (filter ? "&account=" + encodeURIComponent(filter) : ""));
    S.jobs = d.jobs || [];
    renderJobs();
  } catch (e) {
    console.error(e);
  }
}

function renderJobs() {
  const active = S.jobs.filter((j) => ACTIVE_STATES.includes(j.status)).length;
  const count = $("active-jobs-count");
  count.textContent = active;
  count.classList.toggle("hidden", !active);

  const body = $("jobs-body");
  if (!S.jobs.length) {
    body.innerHTML = `<tr><td colspan="8" style="text-align:center;padding:40px" class="muted">No jobs yet. Launch a notebook to see it here.</td></tr>`;
    return;
  }
  body.innerHTML = S.jobs.map((j) => {
    const live = ACTIVE_STATES.includes(j.status);
    const lastLine = (j.error_message || j.last_log_snippet || "").trim().split("\n").pop() || "—";
    const done = ["COMPLETE", "DOWNLOADED"].includes(j.status);
    return `
      <tr>
        <td>${statusBadge(j.status)}</td>
        <td><div style="font-weight:600">${esc(j.title || j.kernel_slug)}</div><div class="kid mono small faint" title="${esc(j.kernel_id)}">${esc(j.kernel_id)}</div></td>
        <td class="mono small">@${esc(j.username)}</td>
        <td class="small">${esc(j.accelerator === "None" ? "CPU" : j.accelerator.replace("NvidiaTesla", ""))}</td>
        <td class="num small">${esc(fmtTime(j.started_at || j.created_at))}</td>
        <td class="num small">${live || j.completed_at ? esc(fmtDuration(j.started_at, live ? null : j.completed_at)) : "—"}</td>
        <td><div class="snippet" title="${esc(lastLine)}">${esc(lastLine)}</div></td>
        <td class="right">
          <span class="row-actions">
            <button class="btn btn-sm" data-job="logs" data-id="${esc(j.id)}">${icon("terminal")}Logs</button>
            <button class="btn btn-sm btn-ghost" data-job="status" data-id="${esc(j.id)}" title="Check status on Kaggle">${icon("refresh")}</button>
            <button class="btn btn-sm btn-ghost" data-job="download" data-id="${esc(j.id)}" title="Download outputs" ${done ? "" : "disabled"}>${icon("download")}</button>
          </span>
        </td>
      </tr>`;
  }).join("");
}

async function jobAction(action, id, btn) {
  if (action === "logs") return openLogs(id);
  if (btn) btn.disabled = true;
  try {
    if (action === "status") {
      const d = await api(`/api/runs/${encodeURIComponent(id)}?refresh=true`);
      toast(`${d.job.kernel_slug}: ${d.job.status.toLowerCase()}`, "info");
    } else if (action === "download") {
      toast("Downloading outputs…", "info");
      const d = await api(`/api/runs/${encodeURIComponent(id)}/download`, { method: "POST" });
      toast("Saved to " + d.output_dir, "success");
    }
  } catch (e) {
    toast(e.message, "error");
  } finally {
    if (btn) btn.disabled = false;
    loadJobs();
  }
}

// Kaggle status is only fetched for jobs still in flight, and less often than
// the local list, to stay well clear of API rate limits.
async function refreshLiveStatuses() {
  const live = S.jobs.filter((j) => ACTIVE_STATES.includes(j.status)).slice(0, 6);
  await Promise.all(live.map((j) => api(`/api/runs/${encodeURIComponent(j.id)}`).catch(() => null)));
  if (live.length) { loadJobs(); loadAccounts(); }
}

async function openLogs(id) {
  S.logJobId = id;
  const job = S.jobs.find((j) => j.id === id);
  $("log-job").textContent = job ? job.kernel_id : id;
  $("log-content").textContent = "Fetching logs from Kaggle…";
  openModal("logs-modal");
  await refreshLogs();
  clearInterval(S.timers.logs);
  S.timers.logs = setInterval(() => {
    if (S.logJobId && $("log-follow").checked) refreshLogs();
  }, 15000);
}

async function refreshLogs() {
  const id = S.logJobId;
  if (!id) return;
  try {
    const d = await api(`/api/runs/${encodeURIComponent(id)}/logs`);
    if (id !== S.logJobId) return;
    const box = $("log-content");
    box.textContent = (d.logs || "").trim() || "(Kaggle has not returned any log output yet)";
    $("log-status").outerHTML = statusBadge(d.status).replace("<span ", '<span id="log-status" ');
    $("log-time").textContent = "Updated " + new Date().toLocaleTimeString();
    if ($("log-follow").checked) box.scrollTop = box.scrollHeight;
  } catch (e) {
    $("log-content").textContent = "Could not load logs: " + e.message;
  }
}

// --------------------------------------------------------------------- init
function refreshAll() {
  loadStatus();
  loadAccounts();
  loadNotebooks();
  loadJobs();
}

document.addEventListener("click", (ev) => {
  const t = ev.target.closest("button, [data-close], .modal-bg");
  if (!t) return;
  if (t.classList.contains("modal-bg")) {
    if (ev.target === t) closeModal(t.id);       // click on backdrop only
    return;
  }
  if (t.dataset.tab) return switchTab(t.dataset.tab);
  if (t.dataset.open) return openModal(t.dataset.open);
  if (t.hasAttribute("data-close")) return closeModal(t.closest(".modal-bg").id);
  if (t.dataset.acc) return accountAction(t.dataset.acc, t.dataset.id, t);
  if (t.dataset.job) return jobAction(t.dataset.job, t.dataset.id, t);
  if (t.dataset.nb) return selectNotebook(t.dataset.nb);
  if (t.dataset.auth) return setManualAuth(t.dataset.auth);
  switch (t.dataset.action) {
    case "import": return importCredentials();
    case "rescan": return loadNotebooks();
    case "reload-jobs": return loadJobs();
  }
});

document.addEventListener("keydown", (ev) => {
  if (ev.key !== "Escape") return;
  document.querySelectorAll(".modal-bg:not(.hidden)").forEach((m) => closeModal(m.id));
});

document.addEventListener("DOMContentLoaded", () => {
  $("btn-refresh").addEventListener("click", refreshAll);
  $("btn-login-start").addEventListener("click", startLogin);
  $("btn-login-link").addEventListener("click", startLinkLogin);
  $("btn-login-copy").addEventListener("click", copyLoginUrl);
  $("btn-login-finish").addEventListener("click", finishLinkLogin);
  $("login-code").addEventListener("keydown", (e) => { if (e.key === "Enter") finishLinkLogin(); });
  $("btn-log-refresh").addEventListener("click", refreshLogs);
  $("manual-form").addEventListener("submit", submitManual);
  $("launch-form").addEventListener("submit", submitLaunch);
  $("f-notebook").addEventListener("change", (e) => selectNotebook(e.target.value));
  $("f-account").addEventListener("change", updateAccountHint);
  $("f-accel").addEventListener("change", (e) =>
    $("p100-note").classList.toggle("hidden", e.target.value !== "NvidiaTeslaP100"));
  $("jobs-filter").addEventListener("change", loadJobs);

  refreshAll();
  const fromHash = location.hash.slice(1);
  if (["accounts", "launch", "jobs"].includes(fromHash)) switchTab(fromHash);

  S.timers.jobs = setInterval(() => {
    if ($("jobs-auto").checked && !document.hidden) loadJobs();
  }, 10000);
  S.timers.live = setInterval(() => {
    if ($("jobs-auto").checked && !document.hidden) refreshLiveStatuses();
  }, 30000);
});
