#!/usr/bin/env python3
"""Daily Briefing Builder for Hermes.

Scans email inbox + cron schedule + todo list and produces a morning briefing.
Delivered to Telegram at wake-up time.

Usage:
    python daily_briefing.py [--hours 24] [--output json|text|markdown]
"""
import sys
import os
import json
import sqlite3
import subprocess
import re
from datetime import datetime, timedelta
from pathlib import Path

BRIEFING_VERSION = "1.0.0"
DEFAULT_HOURS = 24

def get_himalaya_bin():
    """Find himalaya binary."""
    candidates = [
        os.path.expanduser("~/.local/bin/himalaya.exe"),
        os.path.expanduser("~/.local/bin/himalaya"),
        "himalaya",
        "himalaya.exe",
    ]
    for c in candidates:
        if os.path.isfile(c) or subprocess.run(["where", c], capture_output=True).returncode == 0:
            return c
    return None

def scan_email(himalaya_bin, hours=24):
    """Scan inbox for unread/important emails."""
    emails = []
    try:
        result = subprocess.run(
            [himalaya_bin, "envelope", "list", "--output", "json"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            for msg in data:
                emails.append({
                    "id": msg.get("id", ""),
                    "from": msg.get("from", ""),
                    "subject": msg.get("subject", "(no subject)"),
                    "date": msg.get("date", ""),
                    "flags": msg.get("flags", []),
                })
    except Exception as e:
        emails.append({"error": f"Email scan failed: {e}"})
    return emails

def get_cron_jobs():
    """Get upcoming cron jobs from Hermes internal storage."""
    jobs = []
    try:
        cron_dir = Path.home() / ".hermes" / "cron"
        db_path = cron_dir / "jobs.db"
        if not db_path.exists():
            db_path = cron_dir / "cron.db"
        if db_path.exists():
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT name, schedule, next_run_at, enabled FROM jobs WHERE enabled=1")
            for row in cursor.fetchall():
                jobs.append({
                    "name": row[0],
                    "schedule": row[1],
                    "next": row[2],
                    "enabled": bool(row[3]),
                })
            conn.close()
    except Exception as e:
        jobs.append({"error": f"Cron scan failed: {e}"})
    return jobs

def get_todos_from_memory():
    """Extract todos from Hermes memory (MOTD + persistent facts)."""
    todos = []
    try:
        mem_dir = Path.home() / "AppData" / "Local" / "hermes" / "profiles" / "default" / "memories"
        if not mem_dir.exists():
            mem_dir = Path.home() / ".hermes" / "memories"
        for f in mem_dir.glob("*.md"):
            content = f.read_text(encoding="utf-8", errors="ignore")
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("- ") or line.startswith("§"):
                    todos.append(line.lstrip("- ").lstrip("§").strip())
    except Exception as e:
        todos.append(f"[Memory scan error: {e}]")
    return todos[:20]

def classify_email_priority(msg):
    """Classify email by priority keywords."""
    subject = msg.get("subject", "").lower()
    sender = str(msg.get("from", "")).lower()
    high_keywords = ["urgent", "deadline", "action required", "important", "customs", "seizure", "payment", "order"]
    med_keywords = ["follow up", "update", "reminder", "schedule", "meeting"]
    
    for kw in high_keywords:
        if kw in subject or kw in sender:
            return "HIGH"
    for kw in med_keywords:
        if kw in subject or kw in sender:
            return "MEDIUM"
    return "LOW"

def build_briefing(format="markdown", hours=24):
    """Build the daily briefing."""
    now = datetime.now()
    
    # 1. Date/Time header
    date_str = now.strftime("%A, %B %d, %Y")
    time_str = now.strftime("%H:%M")
    
    # 2. Scan emails
    himalaya_bin = get_himalaya_bin()
    emails = scan_email(himalaya_bin, hours) if himalaya_bin else [{"error": "Himalaya not configured — no email scan"}]
    
    high_emails = [e for e in emails if classify_email_priority(e) == "HIGH"]
    med_emails = [e for e in emails if classify_email_priority(e) == "MEDIUM"]
    
    # 3. Cron jobs today
    cron_jobs = get_cron_jobs()
    upcoming = cron_jobs[:8]
    
    # 4. Todos from memory
    todos = get_todos_from_memory()
    active_todos = [t for t in todos if len(t) > 3 and not t.startswith("[")]
    
    # 5. Build output
    if format == "json":
        return json.dumps({
            "date": date_str,
            "time": time_str,
            "emails_total": len(emails),
            "emails_high": len(high_emails),
            "emails_med": len(med_emails),
            "high_emails": high_emails,
            "med_emails": med_emails,
            "cron_jobs_today": upcoming,
            "todos": active_todos[:10],
            "version": BRIEFING_VERSION,
        }, ensure_ascii=False, indent=2)
    
    # Markdown/Text format
    lines = [
        f"☀️ Good morning — {date_str}",
        f"🕐 {time_str}",
        "",
        f"📬 **Email Inbox** ({len(emails)} messages)",
    ]
    
    if high_emails:
        lines.append(f"\n🔴 **HIGH PRIORITY** ({len(high_emails)})")
        for e in high_emails[:5]:
            lines.append(f"  • {e.get('subject', '(no subject)')}")
            lines.append(f"    From: {e.get('from', 'unknown')}")
    
    if med_emails:
        lines.append(f"\n🟡 **MEDIUM** ({len(med_emails)})")
        for e in med_emails[:3]:
            lines.append(f"  • {e.get('subject', '(no subject)')}")
    
    if not high_emails and not med_emails:
        lines.append("  ✅ No urgent emails")
    
    lines.append(f"\n📅 **Today's Schedule** ({len(upcoming)} jobs)")
    for j in upcoming[:5]:
        lines.append(f"  • {j.get('name', 'unnamed')} — {j.get('schedule', '?')}")
    
    lines.append(f"\n✅ **Active Todos** ({len(active_todos)})")
    for t in active_todos[:8]:
        lines.append(f"  • {t}")
    
    lines.append(f"\n💡 **Today's Focus**")
    if high_emails:
        lines.append("  1. Handle HIGH priority emails first")
    if any("customs" in t.lower() or "seizure" in t.lower() for t in active_todos):
        lines.append("  2. Customs package deadline approaching — check status")
    if any("order" in t.lower() or "india" in t.lower() for t in active_todos):
        lines.append("  3. IndiaMART order — confirm Aromasin→Anastrozole swap")
    if not high_emails and not any("customs" in t.lower() for t in active_todos):
        lines.append("  1. No critical blockers — focus on cycle planning")
    
    lines.append(f"\n_Generated {time_str} by Hermes Daily Briefing v{BRIEFING_VERSION}_")
    
    return "\n".join(lines)

if __name__ == "__main__":
    fmt = sys.argv[1] if len(sys.argv) > 1 else "markdown"
    hrs = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_HOURS
    print(build_briefing(format=fmt, hours=hrs))
