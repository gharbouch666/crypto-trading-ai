const POLL_MS = 20000;

function fmtPrice(p) {
  if (p === undefined || p === null) return "—";
  return p >= 100 ? p.toLocaleString(undefined,{maximumFractionDigits:2}) : p.toLocaleString(undefined,{maximumFractionDigits:6});
}

function fmtUsd(v) {
  if (v === null || v === undefined || isNaN(v)) return "—";
  const sign = v >= 0 ? "+" : "";
  return `${sign}$${v.toFixed(2)}`;
}

function dollarPnl(row, tier) {
  if (row.result_R === null || row.result_R === undefined) return null;
  const risk = row[`risk_usd_${tier}`];
  if (risk === null || risk === undefined) return null;
  return risk * row.result_R;
}

function renderRow(r) {
  const statusColor = r.status === "WIN" ? "#3FB68B" : r.status === "LOSS" ? "#E2574C" : "#7A8296";
  const rDisplay = (r.result_R !== null && r.result_R !== undefined) ? `${r.result_R > 0 ? "+" : ""}${r.result_R}R` : "—";
  const pnl100 = dollarPnl(r, "100");
  const pnl1000 = dollarPnl(r, "1000");

  return `<tr>
    <td>${new Date(r.logged_at).toLocaleString()}</td>
    <td>${r.symbol}</td>
    <td class="sig-${r.signal}">${r.signal}</td>
    <td>${r.strategy}</td>
    <td>${r.confidence}</td>
    <td>${fmtPrice(r.entry_ref)}</td>
    <td>${fmtPrice(r.stop)}</td>
    <td>${fmtPrice(r.target)}</td>
    <td style="color:${statusColor}">${r.status}</td>
    <td style="color:${statusColor}">${rDisplay}</td>
    <td style="color:${pnl100>=0?'#3FB68B':'#E2574C'}">${fmtUsd(pnl100)}</td>
    <td style="color:${pnl1000>=0?'#3FB68B':'#E2574C'}">${fmtUsd(pnl1000)}</td>
  </tr>`;
}

async function refresh() {
  try {
    const res = await fetch(`signal_log.json?_=${Date.now()}`, { cache: "no-store" });
    if (!res.ok) return;
    const data = await res.json();
    const rows = data.signals || [];

    const tbody = document.getElementById("logBody");
    if (rows.length === 0) {
      tbody.innerHTML = `<tr><td colspan="12" class="no-signal-note">no signals logged yet</td></tr>`;
    } else {
      tbody.innerHTML = rows.map(renderRow).join("");
    }

    const resolved = rows.filter(r => r.status === "WIN" || r.status === "LOSS");
    document.getElementById("logSummary").textContent = resolved.length
      ? `${rows.length} total · ${resolved.length} resolved`
      : `${rows.length} total · none resolved yet`;

    for (const tier of ["100", "1000"]) {
      const pnls = resolved.map(r => dollarPnl(r, tier)).filter(v => v !== null);
      const total = pnls.reduce((a, b) => a + b, 0);
      const wins = resolved.filter(r => r.status === "WIN").length;
      document.getElementById(`pnl${tier}`).textContent = pnls.length ? fmtUsd(total) : "—";
      document.getElementById(`pnl${tier}`).style.color = total >= 0 ? "#3FB68B" : "#E2574C";
      document.getElementById(`sub${tier}`).textContent = resolved.length
        ? `${resolved.length} trades · ${wins}/${resolved.length} won · from $${tier} starting capital`
        : "no resolved trades yet";
    }

    document.getElementById("lastUpdate").textContent = `refreshed ${new Date().toLocaleTimeString()}`;
    document.getElementById("liveDot").className = "live-dot";
  } catch (e) {
    document.getElementById("liveDot").className = "live-dot dead";
  }
}

refresh();
setInterval(refresh, POLL_MS);
