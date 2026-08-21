"""
LIVE SIGNAL MONITOR -- read-only, NO ORDERS EVER PLACED.

v4: sizing computed for TWO capital tiers simultaneously ($100 and
$1000) so you can directly compare them, per signal AND over time in
the trade log. Both tiers are logged to signal_log.csv so trades.html
can show cumulative $ performance under each.
"""

from pathlib import Path
import sys
import json
import time
import winsound
from datetime import datetime, timezone

import ccxt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ai_strategy_selection as base

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "data" / "live" / "configs.json"
HISTORY_PATH = ROOT / "data" / "live" / "config_history.jsonl"
STATUS_PATH = ROOT / "web" / "status.json"
LOG_CSV_PATH = ROOT / "data" / "live" / "signal_log.csv"
LOG_JSON_PATH = ROOT / "web" / "signal_log.json"

POLL_SECONDS = 60
CANDLE_LIMIT = 300
CONFIG_MAX_AGE_DAYS = 7

CAPITAL_TIERS = {"100": 100.0, "1000": 1000.0}
RISK_PCT = 0.01

LOW_CONFIDENCE_SYMBOLS = {"SOL/USDT"}

WHY = {
    "TREND": "EMA20 above EMA50 and price above EMA20 -- confirmed uptrend.",
    "MOMENTUM": "Price broke above previous candle's high -- breakout.",
    "MEAN_REVERSION": "Price >1 ATR below EMA20 -- snap-back expected.",
}

LOG_COLUMNS = [
    "id", "logged_at", "symbol", "signal", "regime", "strategy", "rr", "atr_mult",
    "confidence", "entry_ref", "stop", "target",
    "risk_usd_100", "position_usd_100", "capped_100",
    "risk_usd_1000", "position_usd_1000", "capped_1000",
    "status", "resolved_at", "exit_price", "result_R",
]


def build_configs():
    print("Rebuilding live configs from full fresh history (~1-2 min per symbol)...")
    configs = {}
    per_symbol_log = {}
    for symbol in base.SYMBOLS:
        df = base.load_symbol(symbol)
        selected = base.select_strategy(df)
        symbol_cfg = {}
        for regime, cfg in selected.items():
            symbol_cfg[regime] = None if cfg is None else {
                "strategy": cfg["strategy"], "rr": cfg["rr"], "atr": cfg["atr"],
                "train_pf": round(cfg["pf"], 3), "train_expectancy": round(cfg["expectancy"], 4),
                "train_trades": cfg["trades"],
            }
        configs[symbol] = symbol_cfg
        per_symbol_log[symbol] = symbol_cfg
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(configs, indent=2))
    with open(HISTORY_PATH, "a") as f:
        f.write(json.dumps({"refit_at": datetime.now(timezone.utc).isoformat(), "configs": per_symbol_log}) + "\n")
    print(f"Saved: {CONFIG_PATH}")
    return configs


def load_configs(force=False):
    if not force and CONFIG_PATH.exists():
        age_days = (time.time() - CONFIG_PATH.stat().st_mtime) / 86400
        if age_days < CONFIG_MAX_AGE_DAYS:
            print(f"Loading cached configs ({age_days:.1f}d old): {CONFIG_PATH}")
            return json.loads(CONFIG_PATH.read_text())
        print(f"Configs {age_days:.1f}d old (max {CONFIG_MAX_AGE_DAYS}) -- refitting.")
    return build_configs()


def fetch_recent(exchange, symbol):
    raw = exchange.fetch_ohlcv(symbol, timeframe="15m", limit=CANDLE_LIMIT)
    return pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])


def confidence_rating(symbol, cfg):
    if symbol in LOW_CONFIDENCE_SYMBOLS:
        return "LOW", "weak separation from random entries in validation testing"
    pf, n = cfg["train_pf"], cfg["train_trades"]
    if pf >= 1.6 and n >= 150:
        return "HIGH", "strong train PF with solid sample size"
    if pf >= 1.3 and n >= 100:
        return "MEDIUM", "acceptable train PF and sample size"
    return "LOW", "thin sample size or weak train PF"


def size_position_multi(entry, stop):
    stop_distance = abs(entry - stop)
    if stop_distance <= 0 or entry <= 0:
        return None
    stop_pct = stop_distance / entry
    out = {}
    for tier_name, capital in CAPITAL_TIERS.items():
        risk_usd = capital * RISK_PCT
        pos_uncapped = risk_usd / stop_pct
        capped = pos_uncapped > capital
        pos_usd = min(pos_uncapped, capital)
        actual_risk = pos_usd * stop_pct
        out[tier_name] = {
            "risk_usd": round(actual_risk, 2),
            "position_usd": round(pos_usd, 2),
            "position_pct_capital": round(pos_usd / capital * 100, 1),
            "capped": capped,
        }
    return out


def compute_status(symbol, df, config):
    prepared = base.prepare(df)
    if len(prepared) < 3:
        return {"symbol": symbol, "error": "not enough candles after indicator warm-up"}, prepared

    closed = prepared.iloc[-2]
    current_price = float(prepared.iloc[-1]["open"])
    regime = closed["regime"]
    regime_config = config.get(regime)

    result = {
        "symbol": symbol, "price": current_price, "regime": regime,
        "candle_time": datetime.fromtimestamp(int(closed["timestamp"]) / 1000, tz=timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    if regime_config is None:
        result.update({"signal": "NONE", "reason": "no validated strategy for this regime"})
        return result, prepared

    signal = base.generate_signal(closed, regime_config["strategy"])
    conf_label, conf_reason = confidence_rating(symbol, regime_config)

    result.update({
        "strategy": regime_config["strategy"], "rr": regime_config["rr"], "atr_mult": regime_config["atr"],
        "train_pf": regime_config["train_pf"], "train_expectancy": regime_config["train_expectancy"],
        "train_trades": regime_config["train_trades"], "confidence": conf_label,
        "confidence_reason": conf_reason, "why": WHY.get(regime_config["strategy"], ""),
    })

    if signal == 0:
        result.update({"signal": "NONE", "reason": "no entry condition met"})
        return result, prepared

    atr = float(closed["atr"])
    risk = atr * regime_config["atr"]
    rr = regime_config["rr"]

    if signal == 1:
        stop = current_price - risk
        target = current_price + risk * rr
        result["signal"] = "LONG"
    else:
        stop = current_price + risk
        target = current_price - risk * rr
        result["signal"] = "SHORT"

    sizing = size_position_multi(current_price, stop)
    result.update({"entry_ref": round(current_price, 6), "stop": round(stop, 6), "target": round(target, 6)})
    if sizing:
        result["sizing"] = sizing

    return result, prepared


def load_log():
    if LOG_CSV_PATH.exists():
        return pd.read_csv(LOG_CSV_PATH)
    return pd.DataFrame(columns=LOG_COLUMNS)


def save_log(log_df):
    LOG_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(LOG_CSV_PATH, index=False)
    LOG_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    safe_df = log_df.tail(100).astype(object).where(pd.notnull(log_df.tail(100)), None)
    recent = safe_df.to_dict(orient="records")
    LOG_JSON_PATH.write_text(json.dumps({"signals": recent[::-1]}, indent=2))


def log_new_signal(log_df, status):
    trade_id = f"{status['symbol']}_{status['candle_time']}"
    if (log_df["id"] == trade_id).any():
        return log_df

    sizing = status.get("sizing", {})
    t100 = sizing.get("100", {})
    t1000 = sizing.get("1000", {})

    row = {
        "id": trade_id, "logged_at": datetime.now(timezone.utc).isoformat(),
        "symbol": status["symbol"], "signal": status["signal"], "regime": status["regime"],
        "strategy": status["strategy"], "rr": status["rr"], "atr_mult": status["atr_mult"],
        "confidence": status["confidence"], "entry_ref": status["entry_ref"],
        "stop": status["stop"], "target": status["target"],
        "risk_usd_100": t100.get("risk_usd"), "position_usd_100": t100.get("position_usd"),
        "capped_100": t100.get("capped"),
        "risk_usd_1000": t1000.get("risk_usd"), "position_usd_1000": t1000.get("position_usd"),
        "capped_1000": t1000.get("capped"),
        "status": "OPEN", "resolved_at": None, "exit_price": None, "result_R": None,
    }
    return pd.concat([log_df, pd.DataFrame([row])], ignore_index=True)


def resolve_open_trades(log_df, symbol, prepared_df):
    open_mask = (log_df["status"] == "OPEN") & (log_df["symbol"] == symbol)
    if not open_mask.any():
        return log_df

    for idx in log_df[open_mask].index:
        row = log_df.loc[idx]
        entry_time = pd.to_datetime(row["id"].split("_", 1)[1])
        candle_times = pd.to_datetime(prepared_df["timestamp"], unit="ms", utc=True)
        after = prepared_df[candle_times > entry_time]
        if after.empty:
            continue

        is_long = row["signal"] == "LONG"
        stop, target = float(row["stop"]), float(row["target"])

        for _, candle in after.iterrows():
            h, l = float(candle["high"]), float(candle["low"])
            if is_long:
                hit_sl, hit_tp = l <= stop, h >= target
            else:
                hit_sl, hit_tp = h >= stop, l <= target

            if hit_sl:
                log_df.loc[idx, ["status", "resolved_at", "exit_price", "result_R"]] = \
                    ["LOSS", datetime.now(timezone.utc).isoformat(), stop, round(-1.0 - base.COST_R, 4)]
                break
            if hit_tp:
                log_df.loc[idx, ["status", "resolved_at", "exit_price", "result_R"]] = \
                    ["WIN", datetime.now(timezone.utc).isoformat(), target, round(row["rr"] - base.COST_R, 4)]
                break

    return log_df


def beep_for(signal):
    try:
        if signal == "LONG":
            winsound.Beep(880, 180); winsound.Beep(1046, 180)
        elif signal == "SHORT":
            winsound.Beep(523, 180); winsound.Beep(392, 180)
    except Exception:
        pass


def main():
    configs = load_configs()
    exchange = ccxt.binance()
    last_signal = {}
    last_config_check = time.time()
    log_df = load_log()

    print("\nLive monitor running (read-only, no orders). Ctrl+C to stop.")
    print(f"Comparing capital tiers: {list(CAPITAL_TIERS.values())} @ {RISK_PCT*100:.0f}% risk/trade")
    print(f"Signal log: {LOG_CSV_PATH}\n")
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)

    while True:
        if time.time() - last_config_check > 3600:
            configs = load_configs()
            last_config_check = time.time()

        statuses = []
        for symbol in base.SYMBOLS:
            try:
                df = fetch_recent(exchange, symbol)
                status, prepared = compute_status(symbol, df, configs.get(symbol, {}))
                log_df = resolve_open_trades(log_df, symbol, prepared)
            except Exception as e:
                status = {"symbol": symbol, "error": str(e)}

            sig = status.get("signal", "ERR")
            prev = last_signal.get(symbol, "NONE")
            if sig in ("LONG", "SHORT") and sig != prev:
                print(f"  *** NEW {sig} SIGNAL: {symbol} ***")
                beep_for(sig)
                log_df = log_new_signal(log_df, status)
            last_signal[symbol] = sig

            statuses.append(status)
            print(f"  {symbol:<10} {status.get('regime', '-'):<18} {sig:<6} conf={status.get('confidence','-')}")

        STATUS_PATH.write_text(json.dumps({"symbols": statuses}, indent=2))
        save_log(log_df)

        resolved = log_df["status"].isin(["WIN", "LOSS"]).sum()
        openc = (log_df["status"] == "OPEN").sum()
        if len(log_df) and resolved:
            wins = (log_df["status"] == "WIN").sum()
            print(f"  Log: {openc} open, {resolved} resolved ({wins/resolved*100:.0f}% win rate)")

        print(f"[{datetime.now().strftime('%H:%M:%S')}] updated. Next check in {POLL_SECONDS}s.\n")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()

