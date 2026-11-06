"use strict";
/* CiteSage web UI - vanilla JS, no build step. All state lives server-side;
   this file only renders /api responses and wires up the forms. */

const $ = (id) => document.getElementById(id);

function esc(text) {
  const div = document.createElement("div");
  div.textContent = text == null ? "" : String(text);
  return div.innerHTML;
}

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (e) { /* keep statusText */ }
    throw new Error(detail);
  }
  return res.json();
}

const post = (path, body) => api(path, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

/* ---------------- status + library ---------------- */

let currentStatus = null;

async function refreshStatus() {
  const pill = $("status-pill");
  try {
    const s = await api("/api/status");
    currentStatus = s;
    pill.className = "pill " + (s.index_ready ? "pill-ok" : "pill-warn");
    pill.textContent = s.index_ready
      ? `index ready · ${s.papers} papers · ${s.chunks} chunks`
      : `no index yet · ${s.papers} papers fetched`;
    $("key-banner").classList.toggle("hidden", s.llm_configured);
  } catch (err) {
    pill.className = "pill pill-warn";
    pill.textContent = "server unreachable";
  }
}

async function refreshPapers() {
  const list = $("paper-list");
  try {
    const papers = await api("/api/papers");
    $("paper-count").textContent = papers.length ? `(${papers.length})` : "";
    $("library-empty").classList.toggle("hidden", papers.length > 0);
    list.innerHTML = papers.map((p) => `
      <li>
        <a href="${esc(p.url)}" target="_blank" rel="noopener">${esc(p.title)}</a>
        <div class="paper-meta">
          ${esc((p.authors || []).slice(0, 3).join(", "))}${(p.authors || []).length > 3 ? " et al." : ""}
          ${p.year ? ` · ${p.year}` : ""} · arXiv:${esc(p.arxiv_id)}
        </div>
      </li>`).join("");
  } catch (err) {
    list.innerHTML = `<li class="muted">Could not load the library: ${esc(err.message)}</li>`;
  }
}

/* ---------------- fetch job ---------------- */

let jobTimer = null;

$("fetch-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const topic = $("fetch-topic").value.trim();
  if (!topic) return;
  $("job-error").classList.add("hidden");
  $("fetch-btn").disabled = true;
  try {
    const { job_id } = await post("/api/fetch", {
      topic,
      max_papers: parseInt($("fetch-count").value, 10) || 5,
    });
    $("job-progress").classList.remove("hidden");
    $("job-step").textContent = "queued";
    $("job-detail").textContent = "";
    clearInterval(jobTimer);
    jobTimer = setInterval(() => pollJob(job_id), 1000);
    pollJob(job_id);
  } catch (err) {
    $("fetch-btn").disabled = false;
    showJobError(err.message === "a fetch job is already running"
      ? "A fetch job is already running — wait for it to finish."
      : `Could not start fetch: ${err.message}`);
  }
});

function showJobError(message) {
  const box = $("job-error");
  box.textContent = message;
  box.classList.remove("hidden");
  $("job-progress").classList.add("hidden");
}

async function pollJob(jobId) {
  try {
    const job = await api(`/api/jobs/${jobId}`);
    $("job-step").textContent = `step: ${job.step}`;
    $("job-detail").textContent = job.detail || "";
    if (job.status === "done") {
      clearInterval(jobTimer);
      $("job-progress").classList.add("hidden");
      $("fetch-btn").disabled = false;
      refreshStatus();
      refreshPapers();
    } else if (job.status === "failed") {
      clearInterval(jobTimer);
      $("fetch-btn").disabled = false;
      showJobError(`Fetch failed at step "${job.step}": ${job.error || "unknown error"}`);
    }
  } catch (err) {
    clearInterval(jobTimer);
    $("fetch-btn").disabled = false;
    showJobError(`Lost track of the fetch job: ${err.message}`);
  }
}

/* ---------------- tabs ---------------- */

function showTab(which) {
  $("tab-ask").classList.toggle("active", which === "ask");
  $("tab-search").classList.toggle("active", which === "search");
  $("view-ask").classList.toggle("hidden", which !== "ask");
  $("view-search").classList.toggle("hidden", which !== "search");
}
$("tab-ask").addEventListener("click", () => showTab("ask"));
$("tab-search").addEventListener("click", () => showTab("search"));

/* ---------------- ask ---------------- */

$("ask-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = $("ask-input").value.trim();
  if (!question) return;
  const notice = $("ask-notice");
  notice.classList.add("hidden");
  $("answer").classList.add("hidden");
  $("answer-meta").classList.add("hidden");
  $("sources").innerHTML = "";
  $("ask-status").classList.remove("hidden");
  $("ask-btn").disabled = true;
  try {
    const data = await post("/api/ask", { question });
    renderAsk(data);
  } catch (err) {
    notice.textContent = `Ask failed: ${err.message}`;
    notice.className = "notice notice-error";
    notice.classList.remove("hidden");
  } finally {
    $("ask-status").classList.add("hidden");
    $("ask-btn").disabled = false;
  }
});

function renderAsk(data) {
  const notice = $("ask-notice");
  if (data.error === "no_llm_key") {
    notice.innerHTML = "Ask needs an LLM key. Add <code>CITESAGE_GROQ_API_KEY</code> or " +
      "<code>CITESAGE_GEMINI_API_KEY</code> to <code>.env</code> and restart the server. " +
      "Search works without one.";
    notice.className = "notice notice-warn";
    notice.classList.remove("hidden");
    return;
  }
  if (data.error === "no_index" || data.index_ready === false) {
    notice.textContent = "No index yet — fetch some papers first, then ask again.";
    notice.className = "notice notice-warn";
    notice.classList.remove("hidden");
    return;
  }
  if (data.error) {
    notice.textContent = `The LLM call failed (${data.warning || data.error}). Check your key, quota, or connection.`;
    notice.className = "notice notice-error";
    notice.classList.remove("hidden");
    return;
  }
  if (data.refused) {
    // An honest refusal is a feature, not a failure: the papers simply
    // don't cover the question.
    notice.textContent = `No supporting evidence in the indexed papers. ${data.answer_text || ""}`;
    notice.className = "notice notice-refusal";
    notice.classList.remove("hidden");
    return;
  }

  // Turn [n] markers into clickable chips that jump to the source card.
  const answerEl = $("answer");
  answerEl.innerHTML = esc(data.answer_text).replace(
    /\[(\d{1,2})\]/g,
    '<button type="button" class="chip" data-n="$1" title="jump to source $1">[$1]</button>'
  );
  answerEl.classList.remove("hidden");
  answerEl.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => jumpToSource(chip.dataset.n));
  });

  const meta = $("answer-meta");
  meta.textContent = `answered by ${data.answered_by || "unknown"}` +
    (data.warning ? ` · warning: ${data.warning}` : "");
  meta.classList.remove("hidden");

  $("sources").innerHTML = "<h3>Sources</h3>" + (data.sources || []).map((s) => `
    <div class="source-card" id="source-${s.n}">
      <div class="source-head"><span class="chip chip-static">[${s.n}]</span> ${esc(s.paper)}</div>
      <div class="paper-meta">${esc(s.citation)}</div>
    </div>`).join("");
}

function jumpToSource(n) {
  const card = $(`source-${n}`);
  if (!card) return;
  card.scrollIntoView({ behavior: "smooth", block: "center" });
  card.classList.remove("highlight");
  void card.offsetWidth; // restart the highlight animation
  card.classList.add("highlight");
}

/* ---------------- search ---------------- */

$("search-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = $("search-input").value.trim();
  if (!query) return;
  const notice = $("search-notice");
  notice.classList.add("hidden");
  $("results").innerHTML = "";
  $("search-status").classList.remove("hidden");
  $("search-btn").disabled = true;
  try {
    const data = await post("/api/search", { query });
    if (!data.index_ready) {
      notice.textContent = "No index yet — fetch some papers first, then search.";
      notice.className = "notice notice-warn";
      notice.classList.remove("hidden");
      return;
    }
    if (!data.results.length) {
      notice.textContent = "No results for that query.";
      notice.className = "notice";
      notice.classList.remove("hidden");
      return;
    }
    $("results").innerHTML = data.results.map((r, i) => `
      <div class="result-card">
        <div class="result-head">
          <span class="rank">#${i + 1}</span>
          <span class="score">score ${Number(r.score).toFixed(4)}</span>
          <span class="paper-meta">dense #${r.dense_rank ?? "–"} · sparse #${r.sparse_rank ?? "–"}</span>
        </div>
        <div class="citation">${esc(r.citation)}</div>
        <p class="snippet">${esc(r.snippet)}</p>
      </div>`).join("");
  } catch (err) {
    notice.textContent = `Search failed: ${err.message}`;
    notice.className = "notice notice-error";
    notice.classList.remove("hidden");
  } finally {
    $("search-status").classList.add("hidden");
    $("search-btn").disabled = false;
  }
});

/* ---------------- boot ---------------- */

refreshStatus();
refreshPapers();
