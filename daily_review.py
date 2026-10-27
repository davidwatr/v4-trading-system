#!/usr/bin/env python3
"""Daily self-review for the V4 paper trading system.

Collects, from live sources only:
  1. Performance snapshot  - latest GH Actions executor run + live Alpaca account
  2. Risk metrics          - drawdown vs peak (sharpe/win-rate/trade-duration gaps flagged)
  3. Data quality          - symbols loaded, local cache staleness (drift checks not in pipeline)
  4. Execution quality     - open/rejected/filled/canceled orders via Alpaca API
  5. Action items          - dynamic (unfilled orders, dd breach) + standing TODOs

Usage: python daily_review.py --snapshot
Keys: reads APCA_* from .env (local run) or GH secrets (workflow run).
"""
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "davidwatr/v4-trading-system"
SCHEDULE_TARGET_UTC = "14:30Z"


def load_env():
    env_path = os.path.join(HERE, ".env")
    if os.path.exists(env_path):
        for line in open(env_path):
            k, _, v = line.strip().partition("=")
            if k and v:
                os.environ.setdefault(k, v)
    # workflow already injects secrets as env vars


def gh(args, timeout=90):
    try:
        r = subprocess.run(
            ["gh"] + args + ["--repo", REPO],
            capture_output=True, text=True, timeout=timeout,
        )
        return r.stdout if r.returncode == 0 else ""
    except Exception:
        return ""


def latest_completed_run():
    out = gh(["run", "list", "--limit", "15", "--json",
              "databaseId,createdAt,conclusion,status,event,displayTitle"])
    if not out:
        return None, []
    runs = json.loads(out)
    done = [r for r in runs if r["status"] == "completed"]
    return (done[0] if done else None), runs


def parse_run_log(run_id):
    log = gh(["run", "view", str(run_id), "--log"], timeout=180)
    keys = {
        "Portfolio Value": r"Portfolio Value: \$([\d,\.]+)",
        "Cash": r"Cash: \$([\d,\.]+)",
        "Regime": r"Regime: (\S+)",
        "Positions": r"Positions: (\d+)",
        "Symbols loaded": r"Loaded (\d+) symbols",
        "Orders submitted": r"Submitted (\d+) buy orders",
        "Confirmed fills": r"CONFIRMED FILL",
        "Pending": r"PENDING \(unfilled yet\)",
        "Failed buys": r"FAILED BUY",
        "Circuit breaker": r"CIRCUIT BREAKER",
        "Errors": r"(?:##\[error\]|ERROR|CRITICAL|Traceback)",
        "Stale closes": r"CLOSED STALE position",
        "Stale cancels": r"Cancelled (\d+) stale orders",
    }
    found = {name: 0 for name in keys}
    values = {}
    for line in log.splitlines():
        for name, pat in keys.items():
            m = re.search(pat, line)
            if m:
                found[name] += 1
                if name in ("Portfolio Value", "Cash", "Regime", "Positions", "Symbols loaded"):
                    values[name] = m.group(1)
    # keep value of LAST occurrence for portfolio/regime lines
    for name in ("Portfolio Value", "Cash", "Regime", "Positions",
                 "Symbols loaded", "Orders submitted"):
        pat = keys[name]
        m = None
        for line in log.splitlines():
            hit = re.search(pat, line)
            if hit:
                m = hit
        if m:
            values[name] = m.group(1)
    return found, values


def alpaca_client():
    from alpaca.trading.client import TradingClient
    return TradingClient(
        os.environ.get("APCA_API_KEY_ID"),
        os.environ.get("APCA_API_SECRET_KEY"),
        paper=os.environ.get("APCA_PAPER", "true").lower() == "true",
    )


def read_peak():
    try:
        return float(json.load(open(os.path.join(HERE, "v4_peak_portfolio.json")))["peak"])
    except Exception:
        return None


def main():
    if "--snapshot" not in sys.argv:
        print("usage: python daily_review.py --snapshot")
        sys.exit(1)
    load_env()
    now = datetime.now(timezone.utc)

    print(f"V4 DAILY SELF-REVIEW — {now:%Y-%m-%d %H:%M}Z")
    print("=" * 60)

    # ── 1. Performance snapshot ─────────────────────────────
    run, runs = latest_completed_run()
    if run:
        sched = [r for r in runs if r["event"] == "schedule"]
        slips = []
        for s in sched[:5]:
            t = datetime.fromisoformat(s["createdAt"].replace("Z", "+00:00"))
            hhmm = (t.hour * 60 + t.minute) // 60.0
            if t.hour < 14 or (t.hour == 14 and t.minute < 30):
                slips.append(f"{s['createdAt'][:16]}Z ({t.hour + 24 - 14.5:.1f}h late)")
            else:
                slips.append(f"{s['createdAt'][:16]}Z (+{(t.hour - 14) + (t.minute - 30) / 60:.1f}h)")
        print(f"[GH] latest completed run: {run['conclusion']} {run['event']} "
              f"{run['createdAt'][:19]}Z (id {run['databaseId']})")
        if slips:
            print(f"[GH] schedule vs {SCHEDULE_TARGET_UTC} target: {'; '.join(slips)}")
        found, vals = parse_run_log(run["databaseId"])
        for k in ("Portfolio Value", "Cash", "Regime", "Positions",
                  "Symbols loaded", "Orders submitted"):
            if k in vals:
                print(f"[GH] {k}: {vals[k]}")
        print(f"[GH] orders={found['Orders submitted']} fills={found['Confirmed fills']} "
              f"pending={found['Pending']} failed={found['Failed buys']} "
              f"errors={found['Errors']} stale_closed={found['Stale closes']}")

    try:
        c = alpaca_client()
        a = c.get_account()
        pv = float(a.portfolio_value)
        print(f"[ALPACA] equity=${pv:,.2f} cash=${float(a.cash):,.2f} "
              f"BP=${float(a.buying_power):,.2f} positions={len(c.get_all_positions())}")
    except Exception as e:
        print(f"[ALPACA] query failed: {e}")
        pv = None

    # ── 2. Risk metrics ─────────────────────────────────────
    peak = read_peak()
    if peak and pv:
        dd = pv / peak - 1
        print(f"[RISK] peak=${peak:,.2f} drawdown={dd:.2%} "
              f"({'OK' if dd > -0.15 else 'BREACH vs 15% cap'})")
    else:
        print("[RISK] peak file unreadable — drawdown unknown")
    print("[RISK] sharpe / win-rate / avg-trade-duration: NOT COMPUTED "
          "by pipeline (executor logs PV/cash/positions only) — metric gap")
    print("[RISK] NOTE: breaker LIVE — executor loads/ratchets v4_peak_portfolio.json "
          "(peak persists across runs; trips at -15% vs all-time peak)")

    # ── 3. Data quality ─────────────────────────────────
    print("[DATA] bars: fresh Yahoo fetch per GH run (no persisted bar store)")
    cache = os.path.join(HERE, "data_cache")
    if os.path.isdir(cache):
        files = [os.path.join(cache, f) for f in os.listdir(cache)]
        if files:
            oldest = min(os.path.getmtime(f) for f in files)
            print(f"[DATA] local data_cache: {len(files)} files, oldest "
                  f"{(now - datetime.fromtimestamp(oldest, timezone.utc)).days}d old "
                  f"(stale — used only by local tooling, not the executor)")
    print("[DATA] missing-bar / stale-quote / feature-drift checks: absent from pipeline")

    # ── 4. Execution quality ────────────────────────────────
    try:
        from alpaca.trading.requests import GetOrdersRequest
        open_orders = c.get_orders(GetOrdersRequest(status="open"))
        acc = sum(1 for o in open_orders if o.status and "accepted" in str(o.status).lower())
        print(f"[EXEC] open orders: {len(open_orders)} ({acc} ACCEPTED, "
              f"{len(open_orders) - acc} other):")
        for o in open_orders:
            print(f"  - {o.symbol} {str(o.side).split('.')[-1]} {int(float(o.qty))} "
                  f"{str(o.type).split('.')[-1]} status={str(o.status).split('.')[-1]} "
                  f"submitted={str(o.submitted_at)[:19]}Z")
        closed = c.get_orders(GetOrdersRequest(status="closed", limit=100))
        fills = [o for o in closed if getattr(o, "filled_at", None)]
        rejects = [o for o in closed if o.status and "rejected" in str(o.status).lower()]
        cancels = [o for o in closed if o.status and "canceled" in str(o.status).lower()]
        print(f"[EXEC] recent closed: {len(fills)} filled, {len(rejects)} rejected, "
              f"{len(cancels)} canceled")
        if fills:
            for o in fills[:10]:
                print(f"  - FILLED {o.symbol} {int(float(o.qty))} at "
                      f"{str(o.filled_at)[:19]}Z")
        print("[EXEC] slippage: unmeasurable until fills print (market orders)")
    except Exception as e:
        print(f"[EXEC] order query failed: {e}")

    # ── 5. Action items ─────────────────────────────────────
    print("-" * 60)
    print("ACTION ITEMS (today):")
    n_open = len(open_orders) if open_orders else 0
    if n_open:
        print(f" 1. {n_open} pending market orders — confirm fills at 13:30Z open "
              f"or cancel-and-redeploy in today's 14:30Z GH run "
              f"(cancel: python -c \"import os; [os.environ.setdefault(k,v) for k,_,v in (l.strip().partition('=') for l in open('.env')) if v]; from alpaca.trading.client import TradingClient; TradingClient(os.environ['APCA_API_KEY_ID'],os.environ['APCA_API_SECRET_KEY'],paper=True).cancel_all_orders(); print('canceled')\")")
    else:
        print(" 1. no open orders — verify today's GH run executed")
    print(" 2. ROTATE paper keys (public since pre-purge exposure): Alpaca dashboard -> "
          "new keys -> GH secrets APCA_API_KEY_ID/APCA_API_SECRET_KEY + local .env")
    print(" 3. Verify local dispatcher cron fired (hermes job e2f4f1832109, weekdays "
          "16:25 Israel = 14:25Z) — GH schedule is backup only; check 'hermes cron list --json'")
    print(" STANDING: weekend guard LIVE (no new buys Sat/Sun), model competition archived "
          "(dead), DST: shift dispatcher cron 16:25<->17:25 local on Israel clock changes")


if __name__ == "__main__":
    main()
