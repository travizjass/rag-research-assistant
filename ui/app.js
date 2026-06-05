// Multi-Agent RAG Research Assistant — live "agent thinking" UI.
// Streams Server-Sent Events from POST /query/stream and renders each node.

const queryEl = document.getElementById("query");
const topkEl = document.getElementById("topk");
const askBtn = document.getElementById("ask-btn");
const timelineEl = document.getElementById("timeline");
const answerPanel = document.getElementById("answer-panel");
const answerText = document.getElementById("answer-text");
const answerSources = document.getElementById("answer-sources");
const badgesEl = document.getElementById("badges");

// Per-node visual config (color + emoji icon).
const NODE_STYLE = {
  query_analyzer: { color: "#a371f7", icon: "🧭" },
  retriever: { color: "#4f8cff", icon: "📚" },
  grader: { color: "#d29922", icon: "⚖️" },
  generator: { color: "#3fb950", icon: "✍️" },
  hallucination_checker: { color: "#f778ba", icon: "🛡️" },
};

// Escape user/model text before injecting into HTML.
function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// Remove the "Agent is thinking…" pulse if present.
function clearThinking() {
  const t = document.getElementById("thinking");
  if (t) t.remove();
}

// Append the pulsing thinking indicator at the bottom of the timeline.
function showThinking(label) {
  clearThinking();
  const div = document.createElement("div");
  div.id = "thinking";
  div.className = "thinking";
  div.innerHTML =
    `<span class="dot"></span><span class="dot"></span><span class="dot"></span>` +
    `<span>${esc(label || "Agent is thinking…")}</span>`;
  timelineEl.appendChild(div);
  div.scrollIntoView({ behavior: "smooth", block: "end" });
}

// Render the inner HTML for a single node's detail payload.
function renderDetail(node, d) {
  if (node === "query_analyzer") {
    let html = `<div><span class="kv">Rewritten search query:</span> <code>${esc(d.search_query)}</code></div>`;
    if (d.query_type) html += `<div><span class="kv">Type:</span> ${esc(d.query_type)}</div>`;
    if (d.sub_queries && d.sub_queries.length) {
      html += `<div><span class="kv">Sub-questions:</span><ul>` +
        d.sub_queries.map((q) => `<li>${esc(q)}</li>`).join("") + `</ul></div>`;
    }
    return html;
  }
  if (node === "retriever") {
    return `<div><span class="kv">Retrieved</span> <b>${d.retrieved}</b> ` +
      `<span class="kv">chunk(s) · attempt ${d.attempt}</span></div>` + docsHtml(d.documents);
  }
  if (node === "grader") {
    const score = d.grading_score != null ? d.grading_score : "—";
    return `<div><span class="kv">Kept</span> <b>${d.kept}</b> ` +
      `<span class="kv">relevant chunk(s) · score</span> <b>${score}</b></div>` +
      docsHtml(d.documents);
  }
  if (node === "generator") {
    let html = `<div><span class="kv">Drafted an answer (attempt ${d.attempt}):</span></div>`;
    html += `<div class="doc"><div class="doc-snippet">${esc(d.answer)}</div></div>`;
    if (d.sources && d.sources.length) {
      html += `<div class="answer-sources">` +
        d.sources.map((s) => `<span class="source-tag">${esc(s)}</span>`).join("") + `</div>`;
    }
    return html;
  }
  if (node === "hallucination_checker") {
    const g = d.hallucination_check === "grounded";
    return `<div><span class="kv">Verdict:</span> ` +
      `<span class="${g ? "tag-grounded" : "tag-ungrounded"}">` +
      `${g ? "✓ grounded" : "✗ ungrounded"}</span></div>`;
  }
  return "";
}

// Render a list of retrieved/graded chunks.
function docsHtml(docs) {
  if (!docs || !docs.length) return "";
  return `<div class="docs">` + docs.map((doc) => {
    const where = doc.page != null ? `${doc.source} · p${doc.page}` : doc.source;
    return `<div class="doc"><div class="doc-meta">` +
      `<span>${esc(where)}</span><span>score ${doc.score}</span></div>` +
      `<div class="doc-snippet">${esc(doc.snippet)}</div></div>`;
  }).join("") + `</div>`;
}

// Append one completed step card to the timeline.
function addStep(ev) {
  clearThinking();
  const style = NODE_STYLE[ev.node] || { color: "#4f8cff", icon: "•" };
  const card = document.createElement("div");
  card.className = "step";
  card.style.setProperty("--node-color", style.color);
  card.innerHTML =
    `<div class="step-head">` +
    `<span class="step-icon">${style.icon}</span>` +
    `<span class="step-label">${esc(ev.label)}</span>` +
    `</div>` +
    `<div class="step-body">${renderDetail(ev.node, ev.detail || {})}</div>`;
  timelineEl.appendChild(card);
  card.scrollIntoView({ behavior: "smooth", block: "end" });
}

// Render the final consolidated answer panel.
function showAnswer(ev) {
  clearThinking();
  answerText.textContent = ev.answer || "(no answer)";
  answerSources.innerHTML = (ev.sources || [])
    .map((s) => `<span class="source-tag">${esc(s)}</span>`)
    .join("");

  const grounded = ev.hallucination_check === "grounded";
  badgesEl.innerHTML =
    `<span class="badge ${grounded ? "grounded" : "ungrounded"}">${esc(ev.hallucination_check)}</span>` +
    `<span class="badge neutral">relevance ${ev.grading_score}</span>` +
    `<span class="badge neutral">retries ${ev.retry_count}</span>`;
  answerPanel.classList.remove("hidden");
  answerPanel.scrollIntoView({ behavior: "smooth", block: "start" });
}

function showError(msg) {
  clearThinking();
  const div = document.createElement("div");
  div.className = "error-box";
  div.textContent = "Error: " + msg;
  timelineEl.appendChild(div);
}

// Dispatch a single parsed SSE event.
function handleEvent(data) {
  switch (data.type) {
    case "start":
      showThinking("Analyzing your question…");
      break;
    case "step":
      addStep(data);
      showThinking("Working…");
      break;
    case "done":
      showAnswer(data);
      break;
    case "error":
      showError(data.message);
      break;
  }
}

// Main: POST the query and stream the SSE response.
async function ask() {
  const query = queryEl.value.trim();
  if (!query) return;
  const topK = parseInt(topkEl.value, 10) || 5;

  // Reset UI state.
  askBtn.disabled = true;
  answerPanel.classList.add("hidden");
  timelineEl.innerHTML = "";
  showThinking("Connecting…");

  try {
    const resp = await fetch("/query/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: topK }),
    });
    if (!resp.ok || !resp.body) {
      throw new Error(`HTTP ${resp.status}`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    // Read the stream and parse SSE frames separated by a blank line.
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        const line = frame.split("\n").find((l) => l.startsWith("data:"));
        if (line) handleEvent(JSON.parse(line.slice(5).trim()));
      }
    }
  } catch (err) {
    showError(err.message || String(err));
  } finally {
    askBtn.disabled = false;
  }
}

// --- Wire up events ---
askBtn.addEventListener("click", ask);
queryEl.addEventListener("keydown", (e) => {
  // Cmd/Ctrl + Enter submits.
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") ask();
});
document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    queryEl.value = chip.dataset.q;
    ask();
  });
});
