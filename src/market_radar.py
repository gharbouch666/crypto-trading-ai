import ccxt
import pandas as pd
import numpy as np

exchange = ccxt.binance({"enableRateLimit": True})

MIN_24H_VOLUME = 10_000_000
MIN_RV = 1.20
MAX_1H_MOVE = 5.0
RR = 2.0

EXCLUDED = {
    "BTC/USDT", "ETH/USDT",
    "USDC/USDT", "FDUSD/USDT",
    "USDT/USDT", "USD1/USDT",
    "BFUSD/USDT", "RLUSD/USDT",
    "EURI/USDT", "EUR/USDT"
}


def get_candles(symbol, timeframe, limit):
    data = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    return pd.DataFrame(
        data,
        columns=["time", "open", "high", "low", "close", "volume"]
    )


def btc_regime():
    df = get_candles("BTC/USDT", "1h", 100)
    close = df["close"]

    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]

    price = close.iloc[-1]
    change24 = (price / close.iloc[-25] - 1) * 100

    if price > ema20 > ema50 and change24 > 0:
        return "RISK_ON", price, change24

    if price < ema20 < ema50 and change24 < 0:
        return "RISK_OFF", price, change24

    return "NEUTRAL", price, change24


def analyze(symbol):

    df5 = get_candles(symbol, "5m", 100)
    df1 = get_candles(symbol, "1h", 100)

    if len(df5) < 60 or len(df1) < 50:
        return None

    close = df5["close"]
    high = df5["high"]
    low = df5["low"]
    volume = df5["volume"]

    price = close.iloc[-1]

    # -------------------------
    # MOMENTUM
    # -------------------------

    move15m = (price / close.iloc[-4] - 1) * 100
    move1h = (price / close.iloc[-13] - 1) * 100
    move4h = (price / df1["close"].iloc[-5] - 1) * 100

    # -------------------------
    # TREND
    # -------------------------

    ema20 = df1["close"].ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = df1["close"].ewm(span=50, adjust=False).mean().iloc[-1]

    trend_up = price > ema20 > ema50
    trend_down = price < ema20 < ema50

    # -------------------------
    # RELATIVE VOLUME
    # -------------------------

    avg_volume = volume.iloc[-21:-1].mean()
    rv = volume.iloc[-1] / avg_volume if avg_volume > 0 else 0

    # -------------------------
    # RECENT RANGE
    # -------------------------

    previous_high = high.iloc[-21:-1].max()
    previous_low = low.iloc[-21:-1].min()

    breakout_up = price > previous_high
    breakout_down = price < previous_low

    # -------------------------
    # PULLBACK
    # -------------------------

    recent_high = high.iloc[-13:-1].max()
    recent_low = low.iloc[-13:-1].min()

    distance_from_high = (recent_high - price) / recent_high * 100
    distance_from_low = (price - recent_low) / recent_low * 100

    bullish_pullback = (
        trend_up
        and move4h > 0
        and move1h > 0
        and distance_from_high <= 1.5
        and not breakout_up
    )

    bearish_pullback = (
        trend_down
        and move4h < 0
        and move1h < 0
        and distance_from_low <= 1.5
        and not breakout_down
    )

    # -------------------------
    # AVOID CHASING
    # -------------------------

    long_ok = (
        move1h > 0
        and move1h <= MAX_1H_MOVE
        and move4h > 0
    )

    short_ok = (
        move1h < 0
        and move1h >= -MAX_1H_MOVE
        and move4h < 0
    )

    # -------------------------
    # ENTRY CONDITIONS
    # -------------------------

    long_setup = (
        trend_up
        and bullish_pullback
        and long_ok
        and rv >= MIN_RV
    )

    short_setup = (
        trend_down
        and bearish_pullback
        and short_ok
        and rv >= MIN_RV
    )

    # -------------------------
    # RISK MODEL
    # -------------------------

    if long_setup:

        entry = price
        stop = recent_low
        risk = entry - stop

        if risk <= 0:
            return None

        tp = entry + (risk * RR)

        return {
            "symbol": symbol,
            "signal": "LONG",
            "entry": entry,
            "sl": stop,
            "tp": tp,
            "rr": RR,
            "rv": rv,
            "1h": move1h,
            "4h": move4h
        }

    if short_setup:

        entry = price
        stop = recent_high
        risk = stop - entry

        if risk <= 0:
            return None

        tp = entry - (risk * RR)

        return {
            "symbol": symbol,
            "signal": "SHORT",
            "entry": entry,
            "sl": stop,
            "tp": tp,
            "rr": RR,
            "rv": rv,
            "1h": move1h,
            "4h": move4h
        }

    return None


def main():

    regime, btc_price, btc_change = btc_regime()

    print("\n==============================")
    print("CRYPTO TRADE RADAR")
    print("==============================")
    print(f"BTC:       {btc_price:,.2f}")
    print(f"BTC 24H:   {btc_change:+.2f}%")
    print(f"REGIME:    {regime}")
    print("==============================\n")

    if regime == "NEUTRAL":
        print("NO TRADE - BTC REGIME NEUTRAL")
        return

    markets = exchange.load_markets()
    tickers = exchange.fetch_tickers()

    candidates = []

    for symbol, market in markets.items():

        if symbol in EXCLUDED:
            continue

        if not symbol.endswith("/USDT"):
            continue

        if not market.get("spot") or not market.get("active"):
            continue

        ticker = tickers.get(symbol)

        if not ticker:
            continue

        volume24h = ticker.get("quoteVolume") or 0

        if volume24h < MIN_24H_VOLUME:
            continue

        try:

            result = analyze(symbol)

            if result:
                result["volume24h"] = volume24h
                candidates.append(result)

        except Exception:
            continue

    if regime == "RISK_ON":
        candidates = [
            x for x in candidates
            if x["signal"] == "LONG"
        ]

    elif regime == "RISK_OFF":
        candidates = [
            x for x in candidates
            if x["signal"] == "SHORT"
        ]

    print("ACTIONABLE SETUPS\n")

    if not candidates:
        print("NO TRADE")
        return

    for i, x in enumerate(candidates[:10], 1):

        print(
            f"{i:2}. {x['symbol']:14} "
            f"{x['signal']:5} "
            f"ENTRY:{x['entry']:.8g} "
            f"SL:{x['sl']:.8g} "
            f"TP:{x['tp']:.8g} "
            f"RR:{x['rr']:.1f} "
            f"RV:{x['rv']:.2f}x "
            f"1H:{x['1h']:+.2f}% "
            f"4H:{x['4h']:+.2f}%"
        )


if __name__ == "__main__":
    main()
