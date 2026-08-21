const REGIMES = ["CHOP_HIGH_VOL","CHOP_LOW_VOL","TREND_LOW_VOL","TREND_HIGH_VOL","VOLUME_EXPANSION"];
const REGIME_SHORT = {CHOP_HIGH_VOL:"CHOP-HV",CHOP_LOW_VOL:"CHOP-LV",TREND_LOW_VOL:"TREND-LV",TREND_HIGH_VOL:"TREND-HV",VOLUME_EXPANSION:"VOL-EXP"};
const CONF_COLOR = {HIGH:"#3FB68B", MEDIUM:"#D4A24E", LOW:"#E2574C"};
const POLL_MS = 15000;
let lastSignals = {};

if ("Notification" in window && Notification.permission === "default") { Notification.requestPermission(); }

function fmtPrice(p) {
  if (p === undefined || p === null) return "—";
  return p >= 100 ? p.toLocaleString(undefined,{maximumFractionDigits:2}) : p.toLocaleString(undefined,{maximumFractionDigits:6});
}

function regimeStrip(activeRegime) {
  return REGIMES.map(r => `<div class="regime-seg ${r===activeRegime?"active":""}" title="${REGIME_SHORT[r]}"></div>`).join("");
}

function notify(symbol, signal, price) {
  if ("Notification" in window && Notification.permission === "granted") {
    new Notification(`${signal} — ${symbol}`, { body: `Price: ${fmtPrice(price)}`, tag: symbol });
  }
}

function tierRow(tier, t) {
  if (!t) return "";
  const cappedTag = t.capped ? " ⚠" : "";
  return `<div class="tier-mini"><span class="tier-mini-label">$${tier}</span><span>$${t.position_usd}</span><span class="tier-mini-risk">risk $${t.risk_usd}${cappedTag}</span></div>`;
}

function renderCard(s) {
  if (s.error) {
    return `<div class="card"><div class="card-head"><span class="symbol">${s.symbol}</span></div><div class="error-card">⚠ ${s.error}</div></div>`;
  }
  const signal = s.signal || "NONE";
  if (signal !== "NONE" && lastSignals[s.symbol] !== signal) { notify(s.symbol, signal, s.price); }
  lastSignals[s.symbol] = signal;

  const confColor = CONF_COLOR[s.confidence] || "#4A5568";
  const confBadge = s.confidence
    ? `<span class="conf-badge" style="color:${confColor};border-color:${confColor}" title="${s.confidence_reason || ''}">${s.confidence}</span>`
    : "";

  const sizing = (signal !== "NONE" && s.sizing)
    ? `<div class="tier-mini-wrap">${tierRow("100", s.sizing["100"])}${tierRow("1000", s.sizing["1000"])}</div>`
    : "";

  const levels = signal === "NONE"
    ? `<div class="no-signal-note">${s.reason || "no active setup"}</div>`
    : `<div class="levels">
        <div><div class="level-label">entry</div><div>${fmtPrice(s.entry_ref)}</div></div>
        <div><div class="level-label">stop</div><div class="stop-val">${fmtPrice(s.stop)}</div></div>
        <div><div class="level-label">target</div><div class="target-val">${fmtPrice(s.target)}</div></div>
      </div>
      ${sizing}`;

  const whyTitle = s.why ? s.why.replace(/"/g, "&quot;") : "";
  const strategyTag = signal !== "NONE"
    ? `<span class="strategy-tag" title="${whyTitle}">${s.strategy} · RR${s.rr} · ATRx${s.atr_mult}</span>`
    : `<span class="strategy-tag">${REGIME_SHORT[s.regime]||s.regime}</span>`;

  return `<div class="card">
      <div class="card-head"><span class="symbol">${s.symbol}</span><span class="price">${fmtPrice(s.price)}</span></div>
      <div class="regime-strip" title="${REGIME_SHORT[s.regime]||s.regime||''}">${regimeStrip(s.regime)}</div>
      <div class="signal-row"><span class="signal-badge ${signal}">${signal}</span>${strategyTag}${confBadge}</div>
      ${levels}
      <div class="meta-row">
        <span>${s.candle_time ? new Date(s.candle_time).toLocaleTimeString() : "—"}</span>
        <span title="Train profit factor / sample size">PF ${s.train_pf ?? "—"} · N${s.train_trades ?? "—"}</span>
      </div>
    </div>`;
}

async function refresh() {
  const dot = document.getElementById("liveDot");
  const lastUpdateEl = document.getElementById("lastUpdate");
  try {
    const res = await fetch(`status.json?_=${Date.now()}`, { cache: "no-store" });
    if (!res.ok) throw new Error("status.json not found yet");
    const data = await res.json();
    document.getElementById("grid").innerHTML = data.symbols.map(renderCard).join("");
    const newest = data.symbols.map(s => s.updated_at).filter(Boolean).sort().pop();
    const ageSec = newest ? (Date.now() - new Date(newest).getTime())/1000 : 999;
    dot.className = "live-dot";
    if (ageSec > 300) dot.classList.add("dead");
    else if (ageSec > 90) dot.classList.add("stale");
    lastUpdateEl.textContent = newest ? `last update: ${new Date(newest).toLocaleTimeString()}` : "waiting for engine…";
  } catch (e) {
    document.getElementById("liveDot").className = "live-dot dead";
    document.getElementById("lastUpdate").textContent = "engine not running";
  }
}

refresh();
setInterval(refresh, POLL_MS);
