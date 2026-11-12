#!/usr/bin/env python3
"""V4 Executor Dispatch — deterministic no-agent dispatcher (hermes cron job e2f4f1832109).

Replaces the LLM-agent dispatcher on 2026-11-12 after it failed every weekday
Oct 30-Nov 10 (hallucinated UTC window times -> wrong SKIPs; garbled Nov 2;
Nov 11 died on OpenRouter HTTP 402 prompt-token cap). No LLM calls in this job.

Logic:
  1. UTC window guard 14:20-14:40Z (job fires 16:25 Israel = 14:25Z Mon-Fri)
  2. same-day duplicate check vs GH runs (backup cron / manual dispatch)
  3. gh workflow run 359758176 --ref main
  4. verify a workflow_dispatch run appeared today
Env overrides (testing only, never set by cron): FORCE_WINDOW=1, DRY_RUN=1
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = "davidwatr/v4-trading-system"
WF = "359758176"
REF = "main"


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def gh_runs(limit):
    r = sh(f"gh run list --repo {REPO} --workflow {WF} --limit {limit} "
           "--json databaseId,createdAt,conclusion,status,event")
    if r.returncode != 0:
        return None, (r.stderr or r.stdout).strip()[:300]
    try:
        return json.loads(r.stdout), None
    except Exception as e:
        return None, f"parse error: {e}"


def main():
    now = datetime.now(timezone.utc)
    hhmm = now.strftime("%H:%M")
    today = now.strftime("%Y-%m-%d")
    force = os.environ.get("FORCE_WINDOW") == "1"
    dry = os.environ.get("DRY_RUN") == "1"

    # 1) UTC window guard
    if not force and not ("14:20" <= hhmm <= "14:40"):
        print(f"V4 dispatch SKIP: fired at {hhmm}Z, outside 14:20-14:40Z window. "
              "Nothing else to do.")
        return 0

    # 2) same-day duplicate check
    runs, err = gh_runs(10)
    if runs is None:
        print(f"V4 dispatch FAILED: gh run list error: {err}")
        return 1
    dup = next((x for x in runs
                if x.get("createdAt", "")[:10] == today
                and (x.get("conclusion") == "success"
                     or x.get("status") in ("queued", "in_progress"))), None)
    if dup:
        print(f"V4 dispatch SKIP: GH run {dup['databaseId']} already "
              f"{dup['status']}/{dup.get('conclusion') or '-'} today ({today}) — "
              "backup/manual fired. Nothing else to do.")
        return 0

    # 3) dispatch
    cmd = f"gh workflow run {WF} --repo {REPO} --ref {REF}"
    if dry:
        print(f"DRY-RUN: no same-day run found; would run: {cmd}")
        return 0
    r = sh(cmd)
    if r.returncode != 0:
        print(f"V4 dispatch FAILED: gh workflow run error: "
              f"{(r.stderr or r.stdout).strip()[:300]}")
        return 1

    # 4) verify queued
    time.sleep(20)
    runs, err = gh_runs(5)
    new = None
    if runs is not None:
        new = next((x for x in runs
                    if x.get("event") == "workflow_dispatch"
                    and x.get("createdAt", "")[:10] == today), None)
    if new:
        print(f"V4 dispatch OK: run {new['databaseId']} queued today ({today}, "
              f"event=workflow_dispatch, status={new['status']}). Executor will "
              "fetch fresh bars, check breaker + weekend guard, and act.")
    else:
        print(f"V4 dispatch WARNING: gh accepted dispatch but no workflow_dispatch "
              f"run for {today} appeared yet. Check GH Actions manually.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
