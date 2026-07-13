// Automated Reward Repairer — frontend
// Connects to /events (SSE) and renders messages + questions live.

(() => {
  const feed = document.getElementById("feed");
  const empty = document.getElementById("emptyState");
  const workspaceEl = document.getElementById("workspace");
  const statusEl = document.getElementById("status");
  const sidebar = document.getElementById("sidebar");
  const sidebarKey = document.getElementById("sidebarKey");
  const thinkingEl = document.getElementById("thinking");
  let pairCount = 0;  // only the first pair card is expanded by default

  function setEmpty(hide) { empty.style.display = hide ? "none" : ""; }
  function setState(s) {
    if (!s) return;
    if (s.workspace) workspaceEl.textContent = s.workspace;
    if (s.status) {
      statusEl.textContent = s.status;
      statusEl.dataset.status = s.status;
    }
    updateThinking();
  }

  function updateThinking() {
    // Show the "thinking" spinner whenever the pipeline is actively running
    // and there is no pending question/CTA waiting on the user. Keep it
    // pinned to the end of the feed so new messages push it down.
    const running = statusEl.dataset.status === "running";
    const hasPending = !!feed.querySelector(".msg-q[data-qid], .msg-cta[data-qid]");
    const show = running && !hasPending;
    thinkingEl.hidden = !show;
    if (show) {
      // Move to the end of the feed so it always sits below the latest message.
      feed.appendChild(thinkingEl);
    }
  }

  // ----- minimal markdown renderer (headers, code, lists, tables, bold/italic) ----
  function escapeHTML(s) {
    return s.replace(/[&<>"']/g, c => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    }[c]));
  }

  // ----- minimal Python syntax highlighter -----
  // Regex-driven, ordered by priority. Strings/comments first so keywords
  // inside them aren't styled. Output: HTML with <span class="tok-*">.
  const PY_KEYWORDS = new Set([
    "False","None","True","and","as","assert","async","await","break","class",
    "continue","def","del","elif","else","except","finally","for","from","global",
    "if","import","in","is","lambda","nonlocal","not","or","pass","raise","return",
    "try","while","with","yield","match","case"
  ]);
  const PY_BUILTINS = new Set([
    "abs","all","any","bool","bytes","callable","chr","complex","dict","dir",
    "divmod","enumerate","eval","exec","filter","float","format","frozenset",
    "getattr","globals","hasattr","hash","help","hex","id","input","int",
    "isinstance","issubclass","iter","len","list","locals","map","max","min",
    "next","object","oct","open","ord","pow","print","property","range","repr",
    "reversed","round","set","setattr","slice","sorted","staticmethod","str",
    "sum","super","tuple","type","vars","zip"
  ]);

  function highlightPython(src) {
    const out = [];
    let i = 0;
    const n = src.length;

    function emit(text, cls) {
      const esc = escapeHTML(text);
      if (cls) out.push(`<span class="tok-${cls}">${esc}</span>`);
      else out.push(esc);
    }

    while (i < n) {
      const c = src[i];
      const next2 = src.slice(i, i + 3);

      // Triple-quoted strings (with optional prefix r/b/u/f)
      const tripleMatch = src.slice(i).match(/^[rbuRBUfF]{0,2}("""|''')/);
      if (tripleMatch) {
        const quote = tripleMatch[1];
        const startLen = tripleMatch[0].length;
        const end = src.indexOf(quote, i + startLen);
        const stop = end < 0 ? n : end + quote.length;
        emit(src.slice(i, stop), "string");
        i = stop;
        continue;
      }

      // Single-line strings (handle escapes; allow optional prefix)
      const singleMatch = src.slice(i).match(/^[rbuRBUfF]{0,2}(["'])/);
      if (singleMatch) {
        const quote = singleMatch[1];
        const startLen = singleMatch[0].length;
        let j = i + startLen;
        while (j < n) {
          if (src[j] === "\\" && j + 1 < n) { j += 2; continue; }
          if (src[j] === quote || src[j] === "\n") break;
          j++;
        }
        if (j < n && src[j] === quote) j++;
        emit(src.slice(i, j), "string");
        i = j;
        continue;
      }

      // Comments
      if (c === "#") {
        const eol = src.indexOf("\n", i);
        const stop = eol < 0 ? n : eol;
        emit(src.slice(i, stop), "comment");
        i = stop;
        continue;
      }

      // Numbers
      const numMatch = src.slice(i).match(/^(?:0[xX][0-9a-fA-F_]+|0[oO][0-7_]+|0[bB][01_]+|\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][-+]?\d[\d_]*)?)/);
      if (numMatch) {
        emit(numMatch[0], "number");
        i += numMatch[0].length;
        continue;
      }

      // Identifiers / keywords / builtins / function names / class names
      const idMatch = src.slice(i).match(/^[A-Za-z_][A-Za-z0-9_]*/);
      if (idMatch) {
        const word = idMatch[0];
        const after = src.slice(i + word.length);
        let cls = null;
        if (PY_KEYWORDS.has(word)) cls = "keyword";
        else if (PY_BUILTINS.has(word)) cls = "builtin";
        else if (/^[A-Z]/.test(word)) cls = "class-name";
        else if (after.startsWith("(")) cls = "function";
        emit(word, cls);
        i += word.length;
        continue;
      }

      // Decorators: @name
      if (c === "@") {
        const m = src.slice(i).match(/^@[A-Za-z_][A-Za-z0-9_.]*/);
        if (m) { emit(m[0], "decorator"); i += m[0].length; continue; }
      }

      // Operators / punctuation / whitespace — emit raw
      emit(c, null);
      i++;
    }
    return out.join("");
  }
  function inline(s) {
    s = escapeHTML(s);
    s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
    s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>");
    s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
    return s;
  }
  function renderMarkdown(src) {
    const lines = src.split("\n");
    const out = [];
    let inCode = null;
    let inTable = null;       // {cols: n, header: [...]}
    let inList = null;        // 'ul' or 'ol'
    let codeBuf = [];

    function flushList() {
      if (inList) { out.push(`</${inList}>`); inList = null; }
    }
    function flushTable() {
      if (inTable) { out.push("</tbody></table>"); inTable = null; }
    }

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];

      if (inCode !== null) {
        if (/^```/.test(line)) {
          out.push(`<pre class="code-block"><code class="lang-${inCode}">${escapeHTML(codeBuf.join("\n"))}</code></pre>`);
          codeBuf = []; inCode = null;
        } else {
          codeBuf.push(line);
        }
        continue;
      }
      const cb = line.match(/^```(\S*)\s*$/);
      if (cb) { flushList(); flushTable(); inCode = cb[1] || ""; continue; }

      // Tables: a row of pipes + header sep on next line
      if (/^\s*\|.*\|\s*$/.test(line) && i + 1 < lines.length && /^\s*\|[\s\-:|]+\|\s*$/.test(lines[i + 1])) {
        flushList();
        flushTable();
        const cells = line.trim().slice(1, -1).split("|").map(s => s.trim());
        out.push('<table><thead><tr>' +
          cells.map(c => `<th>${inline(c)}</th>`).join("") +
          "</tr></thead><tbody>");
        inTable = { cols: cells.length };
        i++; // skip the separator row
        continue;
      }
      if (inTable && /^\s*\|.*\|\s*$/.test(line)) {
        const cells = line.trim().slice(1, -1).split("|").map(s => s.trim());
        out.push("<tr>" + cells.map(c => `<td>${inline(c)}</td>`).join("") + "</tr>");
        continue;
      } else if (inTable) {
        flushTable();
      }

      const h = line.match(/^(#{1,3})\s+(.*)$/);
      if (h) {
        flushList();
        out.push(`<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`);
        continue;
      }
      const ul = line.match(/^\s*[-*+]\s+(.*)$/);
      if (ul) {
        if (inList !== "ul") { flushList(); out.push("<ul>"); inList = "ul"; }
        out.push(`<li>${inline(ul[1])}</li>`);
        continue;
      }
      const ol = line.match(/^\s*\d+\.\s+(.*)$/);
      if (ol) {
        if (inList !== "ol") { flushList(); out.push("<ol>"); inList = "ol"; }
        out.push(`<li>${inline(ol[1])}</li>`);
        continue;
      }
      const bq = line.match(/^\s*>\s?(.*)$/);
      if (bq) {
        flushList();
        out.push(`<blockquote>${inline(bq[1])}</blockquote>`);
        continue;
      }

      if (line.trim() === "") {
        flushList();
        out.push("");
        continue;
      }
      out.push(`<p>${inline(line)}</p>`);
    }
    flushList();
    flushTable();
    return out.join("\n");
  }

  function tpl(id) {
    return document.getElementById("tpl-" + id).content.firstElementChild.cloneNode(true);
  }

  // ----- per-type renderers --------------------------------------------------

  function renderText(m) {
    const el = tpl("text");
    el.querySelector(".msg-body").textContent = m.content || "";
    return el;
  }

  function renderMarkdownMsg(m) {
    const el = tpl("markdown");
    el.querySelector(".msg-body").innerHTML = renderMarkdown(m.content || "");
    return el;
  }

  function renderStatus(m) {
    const el = tpl("status");
    el.querySelector(".msg-body").textContent = m.content || "";
    return el;
  }

  function renderDivider() {
    return tpl("divider");
  }

  function showFeatureKey(m) {
    // A feature-key post marks the start of a fresh batch of comparisons, so
    // the next pair becomes the one expanded by default.
    pairCount = 0;
    // Feature keys aren't placed in the chronological feed — they live in the
    // right sidebar, which is only revealed once a key has been posted.
    sidebarKey.innerHTML = "";
    const card = tpl("feature-key");
    const tbody = card.querySelector("tbody");
    (m.features || []).forEach(f => {
      const tr = document.createElement("tr");
      const tdName = document.createElement("td");
      tdName.className = "k-name";
      tdName.textContent = f.name || "";
      const tdDesc = document.createElement("td");
      tdDesc.className = "k-desc";
      tdDesc.textContent = f.description || "";
      tr.appendChild(tdName);
      tr.appendChild(tdDesc);
      tbody.appendChild(tr);
    });
    sidebarKey.appendChild(card);
    document.body.classList.add("has-key");
    sidebar.setAttribute("aria-hidden", "false");
    return null;
  }

  function hideFeatureKey() {
    sidebarKey.innerHTML = "";
    document.body.classList.remove("has-key");
    sidebar.setAttribute("aria-hidden", "true");
  }

  function renderCode(m) {
    const el = tpl("code");
    el.querySelector(".lang").textContent = m.language || "code";
    el.querySelector(".caption").textContent = m.caption || "";
    const code = el.querySelector("code");
    const src = m.content || "";
    if ((m.language || "").toLowerCase() === "python") {
      code.innerHTML = highlightPython(src);
    } else {
      code.textContent = src;
    }
    return el;
  }

  function renderImage(m) {
    const el = tpl("image");
    el.querySelector(".caption").textContent = m.caption || "";
    const img = el.querySelector("img");
    img.src = m.src || "";
    img.alt = m.caption || "";
    return el;
  }

  function renderVideo(m) {
    const el = tpl("video");
    el.querySelector(".caption").textContent = m.caption || "";
    const v = el.querySelector("video");
    v.src = m.src || "";
    return el;
  }

  function _makeVizFigure(v) {
    const fig = document.createElement("figure");
    let media;
    if ((v.src || "").toLowerCase().endsWith(".mp4")) {
      media = document.createElement("video");
      media.controls = true;
      media.preload = "metadata";
    } else {
      media = document.createElement("img");
    }
    media.src = v.src || "";
    fig.appendChild(media);
    if (v.caption) {
      const cap = document.createElement("figcaption");
      cap.textContent = v.caption;
      fig.appendChild(cap);
    }
    if (v.probability !== undefined && v.probability !== null) {
      const prob = document.createElement("div");
      prob.className = "viz-prob";
      const pct = Math.round(Number(v.probability) * 100);
      prob.textContent = `${pct}% chance`;
      fig.appendChild(prob);
    }
    if (v.preferred) {
      const pref = document.createElement("div");
      pref.className = "viz-preferred";
      pref.textContent = "Designed reward function prefers this trajectory";
      fig.appendChild(pref);
    }
    return fig;
  }

  function renderComparison(m) {
    if ((m.layout || "").toLowerCase() === "lottery") return renderLottery(m);
    const el = tpl("comparison");
    el.querySelector(".label").textContent = m.label || "Comparison";
    // Videos on top, then "why they disagree", then one line per characteristic.
    const vizDiv = el.querySelector(".cmp-viz");
    (m.viz || []).forEach(v => vizDiv.appendChild(_makeVizFigure(v)));
    if (m.why) {
      el.querySelector(".cmp-why").innerHTML =
        '<strong>Why they disagree:</strong> ' + inline(m.why);
    } else {
      el.querySelector(".cmp-why").remove();
    }
    const chars = el.querySelector(".cmp-chars");
    const lines = (m.content || "").split("\n").map(s => s.trim()).filter(Boolean);
    // First line is the lead-in header ("Your preferred trajectory has a");
    // the remaining lines are the bulleted "higher/lower/same …" items.
    if (lines.length) {
      const head = document.createElement("div");
      head.className = "cmp-chars-head";
      head.innerHTML = inline(lines[0]);
      chars.parentNode.insertBefore(head, chars);
      lines.slice(1).forEach(line => {
        const li = document.createElement("li");
        li.innerHTML = inline(line);   // renders **higher**/**lower**/**same** as bold
        chars.appendChild(li);
      });
    }
    // Only the first pair is expanded; later pairs start collapsed.
    pairCount++;
    if (pairCount === 1) el.querySelector(".cmp-details").open = true;
    return el;
  }

  function renderLottery(m) {
    const el = tpl("lottery");
    el.querySelector(".label").textContent = m.label || "Lottery";
    const abRow = el.querySelector(".lot-ab");
    const cRow = el.querySelector(".lot-c");
    // Items with role "A"/"B" go on the top row, "C" on the bottom row.
    // If roles aren't set, fall back to positional: first two on top, third below.
    const items = m.viz || [];
    const ab = items.filter(v => /^[AB]$/i.test(String(v.role || "")));
    const c = items.filter(v => /^C$/i.test(String(v.role || "")));
    let topItems, bottomItems;
    if (ab.length || c.length) {
      topItems = ab;
      bottomItems = c;
    } else {
      topItems = items.slice(0, 2);
      bottomItems = items.slice(2, 3);
    }
    // A and B are complementary (B = 1 - A), but each percentage is rounded
    // independently below, which can sum to 101% (e.g. p=0.995 → 100% + 1%).
    // Derive B's displayed percent from A's so they always sum to 100.
    const aItem = topItems.find(v => /^A$/i.test(String(v.role || "")));
    const bItem = topItems.find(v => /^B$/i.test(String(v.role || "")));
    if (aItem && bItem &&
        aItem.probability != null && bItem.probability != null) {
      const pctA = Math.round(Number(aItem.probability) * 100);
      aItem.probability = pctA / 100;
      bItem.probability = (100 - pctA) / 100;
    }
    topItems.forEach(v => abRow.appendChild(_makeVizFigure(v)));
    bottomItems.forEach(v => cRow.appendChild(_makeVizFigure(v)));
    if (!topItems.length) abRow.remove();
    if (!bottomItems.length) cRow.remove();
    if (m.why) {
      el.querySelector(".cmp-why").innerHTML =
        '<strong>Why they disagree:</strong> ' + inline(m.why);
    } else {
      el.querySelector(".cmp-why").remove();
    }
    const block = el.querySelector(".cmp-block");
    (m.content || "").split("\n").map(s => s.trim()).filter(Boolean).forEach(line => {
      const p = document.createElement("p");
      p.innerHTML = inline(line);
      block.appendChild(p);
    });
    // Lotteries always start collapsed.
    return el;
  }

  function renderQuestion(m) {
    // Hide already-answered questions from the feed entirely.
    if (m.answered) return null;

    const el = tpl("question");
    el.dataset.qid = m.id;
    el.querySelector(".q-prompt").innerHTML = inline(m.prompt || "");
    const body = el.querySelector(".q-body");
    const qstatus = el.querySelector(".qstatus");

    qstatus.textContent = "awaiting your answer";
    statusEl.textContent = "awaiting"; statusEl.dataset.status = "awaiting";

    if (m.input === "form" && Array.isArray(m.fields) && m.fields.length) {
      const form = document.createElement("div");
      form.className = "q-form";
      const inputs = {};
      m.fields.forEach(field => {
        const wrap = document.createElement("div");
        wrap.className = "q-field";
        const lab = document.createElement("label");
        lab.className = "q-field-label";
        lab.textContent = field.label || field.name || "";
        wrap.appendChild(lab);
        if (field.description) {
          const desc = document.createElement("div");
          desc.className = "q-field-desc";
          desc.textContent = field.description;
          wrap.appendChild(desc);
        }
        const ta = document.createElement("textarea");
        ta.className = "q-field-input";
        ta.placeholder = field.placeholder || "";
        if (field.prefill) ta.value = field.prefill;
        wrap.appendChild(ta);
        inputs[field.name] = ta;
        form.appendChild(wrap);
      });
      body.appendChild(form);

      const actions = document.createElement("div");
      actions.className = "q-actions";
      const submit = document.createElement("button");
      submit.className = "btn";
      submit.textContent = "Submit";
      submit.addEventListener("click", () => {
        const value = {};
        let anyFilled = false;
        Object.entries(inputs).forEach(([name, ta]) => {
          value[name] = ta.value.trim();
          if (value[name]) anyFilled = true;
        });
        if (!anyFilled) return;
        submitAnswer(m.id, { value });
      });
      actions.appendChild(submit);
      body.appendChild(actions);
    } else if (m.input === "choice" || (m.options && m.options.length)) {
      const list = document.createElement("div");
      list.className = "q-options";
      let selected = null;
      const allowMulti = !!m.multi;
      const selectedSet = new Set();

      (m.options || []).forEach((opt, i) => {
        const optEl = document.createElement("button");
        optEl.className = "q-option";
        optEl.type = "button";
        optEl.innerHTML = `<span class="opt-label">${inline(opt.label || "")}</span>` +
          (opt.description ? `<span class="opt-desc">${inline(opt.description)}</span>` : "");
        optEl.dataset.value = opt.value || opt.label;
        optEl.addEventListener("click", () => {
          if (allowMulti) {
            if (selectedSet.has(optEl)) {
              selectedSet.delete(optEl); optEl.classList.remove("selected");
            } else {
              selectedSet.add(optEl); optEl.classList.add("selected");
            }
          } else {
            if (selected) selected.classList.remove("selected");
            optEl.classList.add("selected");
            selected = optEl;
          }
        });
        list.appendChild(optEl);
      });
      body.appendChild(list);

      const actions = document.createElement("div");
      actions.className = "q-actions";
      const submit = document.createElement("button");
      submit.className = "btn";
      submit.textContent = "Submit";
      submit.addEventListener("click", () => {
        let value;
        if (allowMulti) {
          value = [...selectedSet].map(o => o.dataset.value);
          if (!value.length) return;
        } else {
          if (!selected) return;
          value = selected.dataset.value;
        }
        submitAnswer(m.id, { value });
      });
      actions.appendChild(submit);
      body.appendChild(actions);
    } else if (m.input === "longtext" || m.input === "markdown") {
      const ta = document.createElement("textarea");
      ta.className = "q-textarea";
      ta.placeholder = m.placeholder || "Type here…";
      if (m.prefill) ta.value = m.prefill;
      body.appendChild(ta);

      const actions = document.createElement("div");
      actions.className = "q-actions";
      const submit = document.createElement("button");
      submit.className = "btn";
      submit.textContent = "Submit";
      submit.addEventListener("click", () => {
        const value = ta.value.trim();
        if (!value) return;
        submitAnswer(m.id, { value });
      });
      actions.appendChild(submit);
      body.appendChild(actions);
    } else {
      // single-line text
      const inp = document.createElement("input");
      inp.className = "q-input";
      inp.placeholder = m.placeholder || "";
      if (m.prefill) inp.value = m.prefill;
      body.appendChild(inp);

      const actions = document.createElement("div");
      actions.className = "q-actions";
      const submit = document.createElement("button");
      submit.className = "btn";
      submit.textContent = "Submit";
      submit.addEventListener("click", () => {
        const value = inp.value.trim();
        if (!value) return;
        submitAnswer(m.id, { value });
      });
      actions.appendChild(submit);
      body.appendChild(actions);
      inp.addEventListener("keydown", e => {
        if (e.key === "Enter") submit.click();
      });
    }
    return el;
  }

  function removeAnsweredCard(qid) {
    const el = feed.querySelector(`[data-qid="${qid}"]`);
    if (el) el.remove();
    statusEl.textContent = "running"; statusEl.dataset.status = "running";
    updateThinking();
  }

  function submitAnswer(qid, payload) {
    const el = feed.querySelector(`[data-qid="${qid}"]`);
    if (el) el.querySelectorAll("button, textarea, input").forEach(x => x.disabled = true);
    fetch(`/api/answer/${qid}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    }).then(r => r.json()).then(() => {
      // The "answered" SSE event will remove the card from the feed.
    }).catch(err => {
      console.error("Submit failed:", err);
      if (el) el.querySelectorAll("button, textarea, input").forEach(x => x.disabled = false);
    });
  }

  function renderCTA(m) {
    if (m.answered) return null;
    const el = tpl("cta");
    el.dataset.qid = m.id;
    if (m.prompt) el.querySelector(".cta-prompt").textContent = m.prompt;
    else el.querySelector(".cta-prompt").remove();
    const btn = el.querySelector(".btn-cta");
    btn.textContent = m.label || "Continue";
    btn.addEventListener("click", () => {
      btn.disabled = true;
      submitAnswer(m.id, { value: m.value ?? "continue" });
    });
    statusEl.textContent = "awaiting"; statusEl.dataset.status = "awaiting";
    return el;
  }

  // ----- master dispatcher --------------------------------------------------

  const RENDERERS = {
    text: renderText,
    markdown: renderMarkdownMsg,
    status: renderStatus,
    divider: renderDivider,
    code: renderCode,
    image: renderImage,
    video: renderVideo,
    comparison: renderComparison,
    question: renderQuestion,
    cta: renderCTA,
  };

  function render(m) {
    if (m.type === "feature_key") {
      showFeatureKey(m);
      return;
    }
    const r = RENDERERS[m.type] || renderText;
    const node = r(m);
    if (!node) {
      updateThinking();
      return;
    }
    if (m.id) node.dataset.mid = m.id;
    feed.appendChild(node);
    setEmpty(true);
    updateThinking();
    // smooth scroll if near bottom
    const nearBottom = window.innerHeight + window.scrollY > document.body.offsetHeight - 200;
    if (nearBottom) window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
  }

  function clearFeed() {
    pairCount = 0;
    [...feed.querySelectorAll(".msg, .divider")].forEach(n => n.remove());
    hideFeatureKey();
    setEmpty(false);
    updateThinking();
  }


  // ----- SSE connection -----------------------------------------------------

  function connect() {
    const sse = new EventSource("/events");
    sse.onmessage = ev => {
      if (!ev.data) return;
      let payload;
      try { payload = JSON.parse(ev.data); } catch { return; }
      const e = payload.event;
      if (e === "init") {
        setState(payload.state);
        clearFeed();
        (payload.messages || []).forEach(render);
      } else if (e === "message") {
        render(payload.data);
      } else if (e === "state") {
        setState(payload.data);
      } else if (e === "answered") {
        removeAnsweredCard(payload.qid);
      } else if (e === "clear") {
        clearFeed();
      }
    };
    sse.onerror = () => {
      sse.close();
      setTimeout(connect, 1500);
    };
  }
  connect();
})();
