import ccxt, pandas as pd, numpy as np, time, os
from itertools import product

SYMBOLS = ["BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT","BNB/USDT"]
TIMEFRAME = "15m"
DAYS = 180
FEE = 0.001
SLIPPAGE = 0.0005
LIMIT = 1000

exchange = ccxt.binance({"enableRateLimit": True})

def download(symbol):
    since = exchange.milliseconds() - DAYS*24*60*60*1000
    rows = []
    while since < exchange.milliseconds():
        try:
            data = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since, limit=LIMIT)
            if not data:
                break
            rows.extend(data)
            since = data[-1][0] + 1
            if len(data) < LIMIT:
                break
        except Exception as e:
            print("retry:", symbol, type(e).__name__)
            time.sleep(2)

    df = pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    return df

def indicators(df):
    d = df.copy()

    d["ema20"] = d.close.ewm(span=20, adjust=False).mean()
    d["ema50"] = d.close.ewm(span=50, adjust=False).mean()
    d["ema200"] = d.close.ewm(span=200, adjust=False).mean()

    tr1 = d.high-d.low
    tr2 = abs(d.high-d.close.shift())
    tr3 = abs(d.low-d.close.shift())
    tr = pd.concat([tr1,tr2,tr3],axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()

    delta = d.close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0,np.nan)
    d["rsi"] = 100-(100/(1+rs))

    d["vol_ma"] = d.volume.rolling(20).mean()
    d["vol_ratio"] = d.volume / d.vol_ma

    up = d.high.diff()
    down = -d.low.diff()
    plus = np.where((up > down) & (up > 0), up, 0)
    minus = np.where((down > up) & (down > 0), down, 0)

    atr14 = d["atr"].replace(0,np.nan)
    plus_di = 100 * pd.Series(plus).rolling(14).mean() / atr14
    minus_di = 100 * pd.Series(minus).rolling(14).mean() / atr14

    dx = 100 * abs(plus_di-minus_di) / (plus_di+minus_di).replace(0,np.nan)
    d["adx"] = dx.rolling(14).mean()

    d["hh20"] = d.high.shift(1).rolling(20).max()
    d["ll20"] = d.low.shift(1).rolling(20).min()

    return d

def backtest(df, rr, atr_mult, adx_min, vol_min, mode):
    trades = []
    equity = 0
    peak = 0
    maxdd = 0

    for i in range(220, len(df)-2):
        r = df.iloc[i]

        if pd.isna(r.atr) or pd.isna(r.adx) or pd.isna(r.rsi):
            continue

        trend_long = r.close > r.ema50 > r.ema200
        trend_short = r.close < r.ema50 < r.ema200

        volume_ok = r.vol_ratio >= vol_min
        adx_ok = r.adx >= adx_min

        breakout_long = r.close > r.hh20
        breakout_short = r.close < r.ll20

        pullback_long = r.low <= r.ema20 and r.close > r.ema20
        pullback_short = r.high >= r.ema20 and r.close < r.ema20

        long_signal = trend_long and adx_ok and volume_ok and (
            breakout_long or pullback_long
        ) and r.rsi < 70

        short_signal = trend_short and adx_ok and volume_ok and (
            breakout_short or pullback_short
        ) and r.rsi > 30

        if mode == "LONG" and not long_signal:
            continue
        if mode == "SHORT" and not short_signal:
            continue
        if mode == "BOTH" and not (long_signal or short_signal):
            continue

        direction = 1 if long_signal else -1

        entry = df.iloc[i+1].open

        stop_distance = r.atr * atr_mult

        if direction == 1:
            stop = entry - stop_distance
            target = entry + stop_distance * rr
        else:
            stop = entry + stop_distance
            target = entry - stop_distance * rr

        result = None

        for j in range(i+2, min(i+150,len(df))):
            c = df.iloc[j]

            if direction == 1:
                hit_stop = c.low <= stop
                hit_target = c.high >= target
            else:
                hit_stop = c.high >= stop
                hit_target = c.low <= target

            if hit_stop and hit_target:
                result = -1
                break

            if hit_stop:
                result = -1
                break

            if hit_target:
                result = rr
                break

        if result is None:
            continue

        cost_r = (FEE*2 + SLIPPAGE*2) / (stop_distance/entry)
        result -= cost_r

        trades.append(result)

        equity += result
        peak = max(peak,equity)
        maxdd = min(maxdd,equity-peak)

    if not trades:
        return None

    wins = [x for x in trades if x > 0]
    losses = [x for x in trades if x <= 0]

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))

    pf = gross_profit/gross_loss if gross_loss else 999
    expectancy = np.mean(trades)
    winrate = len(wins)/len(trades)*100

    return {
        "trades":len(trades),
        "winrate":winrate,
        "pf":pf,
        "expectancy":expectancy,
        "net":sum(trades),
        "drawdown":maxdd
    }

def main():
    os.makedirs("data",exist_ok=True)

    all_results=[]

    configurations = list(product(
        [1.5,2.0,2.5,3.0],
        [1.0,1.5,2.0],
        [15,20,25],
        [1.0,1.25,1.5],
        ["LONG","SHORT","BOTH"]
    ))

    print("\n========================================")
    print("ADVANCED CRYPTO STRATEGY LAB")
    print("========================================")
    print("Testing configurations:",len(configurations))
    print("Period:",DAYS,"days")
    print("Timeframe:",TIMEFRAME)
    print("Fees:",FEE,"| Slippage:",SLIPPAGE)
    print("========================================")

    for symbol in SYMBOLS:
        print("\nDownloading:",symbol)

        df = download(symbol)
        print("Candles:",len(df))

        df = indicators(df)

        for n,(rr,atr,adx,vol,mode) in enumerate(configurations,1):
            result = backtest(df,rr,atr,adx,vol,mode)

            if result and result["trades"] >= 30:
                all_results.append({
                    "symbol":symbol,
                    "rr":rr,
                    "atr":atr,
                    "adx":adx,
                    "vol_min":vol,
                    "mode":mode,
                    **result
                })

        print("Completed:",symbol)

    results = pd.DataFrame(all_results)

    if results.empty:
        print("\nNO VALID RESULTS")
        return

    results = results.sort_values(
        ["expectancy","pf"],
        ascending=False
    )

    results.to_csv("data/strategy_results.csv",index=False)

    print("\n========================================")
    print("TOP 20 CONFIGURATIONS")
    print("========================================")

    print(
        results.head(20).to_string(
            index=False,
            formatters={
                "winrate":"{:.2f}".format,
                "pf":"{:.2f}".format,
                "expectancy":"{:.3f}".format,
                "net":"{:.2f}".format,
                "drawdown":"{:.2f}".format
            }
        )
    )

    profitable = results[
        (results.expectancy > 0) &
        (results.pf > 1.05) &
        (results.trades >= 50)
    ]

    print("\n========================================")
    print("ROBUST CANDIDATES")
    print("========================================")

    if profitable.empty:
        print("NONE")
        print("\nNo configuration has proven profitable yet.")
    else:
        print(
            profitable.head(20).to_string(
                index=False
            )
        )

    print("\nSaved: data/strategy_results.csv")
    print("========================================")
    print("LAB COMPLETE")
    print("========================================")

if __name__ == "__main__":
    main()
