"""Dip-buy swing bot for btc-race-24h-20261118-1444 (agent-fun/bars-replay/v1).

Strategy, written from scratch: buy CBBTC when its close has fallen DIP_BPS
below its trailing EMA_REF_LEN-bar EMA, sell at TP_BPS above the entry close
or SL_BPS below it. One position at a time, full-size buys, no shorts, no
leverage, integer (bps) math only, no clock, no network, no randomness:
fully deterministic.

Fills land at the next bar's open, so every decision is made on closes and
every exit is judged on closes against the entry close. Entries stop in the
last END_GUARD bars so a position always has time to reach its exit.

Tuned by exact replay sweep on the dev window (see replay_btc.py): the dev
window holds one tradeable V-dip; TP 225 was chosen over 250 because the
dev recovery stayed above +2.50% for only 11 bars but above +2.25% for 66.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))

import makefun_bars

SYMBOL = "CBBTC"
BPS = 10_000

# Tuned on the dev window by replay sweep; see replay_btc.py.
PARAMS = {
    "window": 120,        # entry warm-up: bars of history before the first buy
    "dip_bps": 150,       # buy when close <= EMA * (1 - dip)
    "tp_bps": 225,        # take profit above entry close
    "sl_bps": 150,        # stop loss below entry close
    "trail_bps": 0,       # trailing stop below peak close since entry (0 = off)
    "trend": "none",      # none | above_ema | below_ema
    "ema_len": 240,       # trend filter EMA length
    "confirm": 0,         # 1 = require close >= previous close (bounce) to buy
    "size_bps": 10000,    # fraction of cash to deploy per buy
    "cooldown": 0,        # bars to wait after an exit before re-entering
    "end_guard": 120,     # stop entering in the last N bars of the window
    "total_bars": 1440,   # window length (dev and race windows are both 1440)
    "max_trades": 99,     # cap on round trips
    "max_slip_bps": 500,  # max slippage allowed on any order
    "time_exit": 0,       # exit at market after N bars in a trade (0 = off)
    "tp1_bps": 0,         # partial take-profit level (0 = off)
    "tp1_frac_bps": 5000, # fraction of the position sold at tp1
    "add_bps": 0,         # ladder: add when close falls this far below entry (0 = off)
    "add_frac_bps": 5000, # fraction of cash deployed on the ladder add
    "max_adds": 1,        # ladder adds per round trip
    "ref": "ema",         # dip reference: trailing high of closes | ema
    "ema_ref_len": 240,   # EMA length when ref == ema
}

# Mutable run state, reset per run (a fresh process resets it; the sweep
# harness calls reset_state() between runs).
ST = {}


def reset_state():
    ST.clear()
    ST.update({
        "closes": [],        # closes seen, ints
        "ema": None,         # trend EMA, int (price units)
        "entry_close": None, # close at the bar the buy was decided on
        "entry_seq": -1,     # seq the buy was decided on
        "peak_close": None,  # highest close since entry
        "tp1_done": False,   # partial take-profit already taken
        "adds": 0,           # ladder adds taken this round trip
        "cooldown_until": -1,
        "round_trips": 0,
    })


reset_state()


def _ema_update(prev, close, length):
    # EMA with integer math: alpha = 2 / (length + 1), rounded deterministically.
    if prev is None:
        return close
    return prev + (2 * (close - prev)) // (length + 1)


def on_batch(state, batch):
    bars = batch.get("bars") or []
    if not bars:
        return []
    bar = bars[0]
    if bar.get("symbol") != SYMBOL:
        return []
    close = int(bar["c"])
    seq = state.seq
    closes = ST["closes"]
    closes.append(close)
    ST["ema"] = _ema_update(ST["ema"], close, PARAMS["ema_len"])
    ST["ema_ref"] = _ema_update(ST.get("ema_ref"), close, PARAMS["ema_ref_len"])

    held = state.positions.get(SYMBOL)
    if held is not None and held.qty > 0:
        entry = ST["entry_close"]
        if entry is None:
            entry = close
            ST["entry_close"] = entry
        if close > (ST["peak_close"] or 0):
            ST["peak_close"] = close
        # Partial take-profit first: bank part of the move, let the rest ride.
        tp1 = PARAMS["tp1_bps"]
        if tp1 and not ST["tp1_done"] and close * BPS >= entry * (BPS + tp1):
            ST["tp1_done"] = True
            qty = held.qty * PARAMS["tp1_frac_bps"] // BPS
            if qty > 0 and qty < held.qty:
                return [_sell(state, qty, "tp1", full=False)]
        # Full exits, all on closes vs the entry close.
        if close * BPS >= entry * (BPS + PARAMS["tp_bps"]):
            return [_sell(state, held.qty, "tp")]
        if close * BPS <= entry * (BPS - PARAMS["sl_bps"]):
            return [_sell(state, held.qty, "sl")]
        trail = PARAMS["trail_bps"]
        if trail and ST["peak_close"] is not None:
            if close * BPS <= ST["peak_close"] * (BPS - trail):
                return [_sell(state, held.qty, "trail")]
        if PARAMS["time_exit"] and seq - ST["entry_seq"] >= PARAMS["time_exit"]:
            return [_sell(state, held.qty, "time")]
        # Ladder: price fell further below entry while holding; add if cash allows.
        add = PARAMS["add_bps"]
        if add and ST["adds"] < PARAMS["max_adds"]:
            if close * BPS <= entry * (BPS - add):
                notional = state.cash * PARAMS["add_frac_bps"] // BPS
                if notional >= 1_000_000:
                    ST["adds"] += 1
                    return [{
                        "side": "buy", "symbol": SYMBOL,
                        "notional_quote_units": notional,
                        "max_slippage_bps": PARAMS["max_slip_bps"],
                    }]
        return []

    # Flat: maybe enter.
    if ST["round_trips"] >= PARAMS["max_trades"]:
        return []
    if seq >= PARAMS["total_bars"] - PARAMS["end_guard"]:
        return []
    if seq < ST["cooldown_until"]:
        return []
    window = PARAMS["window"]
    if len(closes) < window + 1:
        return []
    if PARAMS["ref"] == "ema":
        ref = ST["ema_ref"]
    else:
        ref = max(closes[-window:])
    if close * BPS > ref * (BPS - PARAMS["dip_bps"]):
        return []
    if PARAMS["confirm"] and len(closes) >= 2 and close < closes[-2]:
        return []
    trend = PARAMS["trend"]
    if trend == "above_ema" and ST["ema"] is not None and close < ST["ema"]:
        return []
    if trend == "below_ema" and ST["ema"] is not None and close > ST["ema"]:
        return []
    cash = state.cash
    notional = cash * PARAMS["size_bps"] // BPS
    if notional <= 0:
        return []
    ST["entry_close"] = close
    ST["entry_seq"] = seq
    ST["peak_close"] = close
    ST["tp1_done"] = False
    ST["adds"] = 0
    return [{
        "side": "buy",
        "symbol": SYMBOL,
        "notional_quote_units": notional,
        "max_slippage_bps": PARAMS["max_slip_bps"],
    }]


def _sell(state, qty, _why, full=True):
    if full:
        ST["entry_close"] = None
        ST["peak_close"] = None
        ST["round_trips"] += 1
        ST["cooldown_until"] = state.seq + PARAMS["cooldown"] + 1
    return {
        "side": "sell",
        "symbol": SYMBOL,
        "qty": int(qty),
        "max_slippage_bps": PARAMS["max_slip_bps"],
    }


def on_done(state, done):
    return None


if __name__ == "__main__":
    makefun_bars.run(on_batch, on_done=on_done)
