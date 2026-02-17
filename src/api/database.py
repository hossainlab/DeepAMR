"""SQLite database layer for DeepAMR.

Provides user auth, prediction history, activity logging, and dashboard stats.
Uses Python stdlib sqlite3 — zero extra dependencies.
"""

import hashlib
import json
import os
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DB_PATH = Path(__file__).parent.parent.parent / "deepamr.db"

# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ---------------------------------------------------------------------------
# Schema / init
# ---------------------------------------------------------------------------

def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id          TEXT PRIMARY KEY,
        email       TEXT UNIQUE NOT NULL,
        name        TEXT NOT NULL,
        password_hash TEXT NOT NULL,
        salt        TEXT NOT NULL,
        role        TEXT NOT NULL DEFAULT 'user',
        organization TEXT,
        created_at  TEXT NOT NULL,
        last_login  TEXT
    );

    CREATE TABLE IF NOT EXISTS sessions (
        token       TEXT PRIMARY KEY,
        user_id     TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        created_at  TEXT NOT NULL,
        expires_at  TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS predictions (
        id           TEXT PRIMARY KEY,
        sample_id    TEXT NOT NULL,
        user_id      TEXT REFERENCES users(id) ON DELETE SET NULL,
        organism     TEXT NOT NULL,
        status       TEXT NOT NULL DEFAULT 'pending',
        risk_level   TEXT,
        file_name    TEXT,
        file_size    INTEGER,
        results_json TEXT,
        created_at   TEXT NOT NULL,
        completed_at TEXT
    );

    CREATE TABLE IF NOT EXISTS activity_log (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id   TEXT,
        user_name TEXT,
        action    TEXT NOT NULL,
        details   TEXT,
        timestamp TEXT NOT NULL
    );
    """)

    # Seed admin user if table is empty
    row = cur.execute("SELECT COUNT(*) FROM users").fetchone()
    if row[0] == 0:
        salt = os.urandom(16).hex()
        pw_hash = hash_password("admin123", salt)
        cur.execute(
            "INSERT INTO users (id, email, name, password_hash, salt, role, organization, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                "admin@deepamr.org",
                "Admin",
                pw_hash,
                salt,
                "admin",
                "DeepAMR",
                datetime.utcnow().isoformat(),
            ),
        )

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Password helpers
# ---------------------------------------------------------------------------

def hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode()).hexdigest()


def verify_password(password: str, salt: str, pw_hash: str) -> bool:
    return hash_password(password, salt) == pw_hash


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def create_user(email: str, name: str, password: str, role: str = "user", organization: str | None = None) -> Dict:
    conn = get_db()
    user_id = str(uuid.uuid4())
    salt = os.urandom(16).hex()
    pw_hash = hash_password(password, salt)
    now = datetime.utcnow().isoformat()

    try:
        conn.execute(
            "INSERT INTO users (id, email, name, password_hash, salt, role, organization, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, email, name, pw_hash, salt, role, organization, now),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise ValueError("Email already registered")
    user = dict(conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone())
    conn.close()
    return _sanitize_user(user)


def get_user_by_email(email: str) -> Optional[Dict]:
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_id(user_id: str) -> Optional[Dict]:
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_users() -> List[Dict]:
    conn = get_db()
    rows = conn.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
    conn.close()
    return [_sanitize_user(dict(r)) for r in rows]


def delete_user(user_id: str) -> bool:
    conn = get_db()
    cur = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    return cur.rowcount > 0


def update_last_login(user_id: str):
    conn = get_db()
    conn.execute("UPDATE users SET last_login = ? WHERE id = ?", (datetime.utcnow().isoformat(), user_id))
    conn.commit()
    conn.close()


def _sanitize_user(user: Dict) -> Dict:
    """Remove password fields from user dict for API responses."""
    return {
        "id": user["id"],
        "email": user["email"],
        "name": user["name"],
        "role": user["role"],
        "organization": user.get("organization"),
        "createdAt": user["created_at"],
        "lastLogin": user.get("last_login"),
    }


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

def create_session(user_id: str) -> str:
    conn = get_db()
    token = str(uuid.uuid4())
    now = datetime.utcnow()
    expires = now + timedelta(days=7)
    conn.execute(
        "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
        (token, user_id, now.isoformat(), expires.isoformat()),
    )
    conn.commit()
    conn.close()
    return token


def get_session(token: str) -> Optional[Dict]:
    conn = get_db()
    row = conn.execute("SELECT * FROM sessions WHERE token = ?", (token,)).fetchone()
    conn.close()
    if not row:
        return None
    session = dict(row)
    if datetime.fromisoformat(session["expires_at"]) < datetime.utcnow():
        delete_session(token)
        return None
    return session


def delete_session(token: str):
    conn = get_db()
    conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

def save_prediction(
    sample_id: str,
    user_id: Optional[str],
    organism: str,
    status: str,
    risk_level: Optional[str],
    file_name: Optional[str],
    file_size: Optional[int],
    results_json: Optional[str],
) -> Dict:
    conn = get_db()
    pred_id = f"pred-{uuid.uuid4().hex[:8]}"
    now = datetime.utcnow().isoformat()
    completed = now if status == "completed" else None
    conn.execute(
        "INSERT INTO predictions (id, sample_id, user_id, organism, status, risk_level, file_name, file_size, results_json, created_at, completed_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (pred_id, sample_id, user_id, organism, status, risk_level, file_name, file_size, results_json, now, completed),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM predictions WHERE id = ?", (pred_id,)).fetchone()
    conn.close()
    return _format_prediction(dict(row))


def get_prediction(pred_id: str) -> Optional[Dict]:
    conn = get_db()
    row = conn.execute("SELECT * FROM predictions WHERE id = ?", (pred_id,)).fetchone()
    conn.close()
    return _format_prediction(dict(row)) if row else None


def list_predictions(
    user_id: Optional[str] = None,
    organism: Optional[str] = None,
    status: Optional[str] = None,
    risk: Optional[str] = None,
    search: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> List[Dict]:
    conn = get_db()
    query = "SELECT * FROM predictions WHERE 1=1"
    params: List[Any] = []

    if user_id:
        query += " AND user_id = ?"
        params.append(user_id)
    if organism:
        query += " AND organism = ?"
        params.append(organism)
    if status:
        query += " AND status = ?"
        params.append(status)
    if risk:
        query += " AND risk_level = ?"
        params.append(risk)
    if search:
        query += " AND (sample_id LIKE ? OR organism LIKE ? OR file_name LIKE ?)"
        like = f"%{search}%"
        params.extend([like, like, like])
    if date_from:
        query += " AND created_at >= ?"
        params.append(date_from)
    if date_to:
        query += " AND created_at <= ?"
        params.append(date_to)

    query += " ORDER BY created_at DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [_format_prediction(dict(r)) for r in rows]


def get_recent_predictions(limit: int = 5) -> List[Dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [_format_prediction(dict(r)) for r in rows]


def _format_prediction(p: Dict) -> Dict:
    """Convert DB row to frontend-friendly format."""
    results_data = None
    if p.get("results_json"):
        try:
            results_data = json.loads(p["results_json"])
        except json.JSONDecodeError:
            pass

    return {
        "id": p["id"],
        "sampleId": p["sample_id"],
        "organism": p["organism"],
        "status": p["status"],
        "createdAt": p["created_at"],
        "completedAt": p.get("completed_at"),
        "uploadedBy": p.get("user_id", ""),
        "fileName": p.get("file_name", ""),
        "fileSize": p.get("file_size", 0),
        "overallRisk": (p.get("risk_level") or "low").lower(),
        "results": results_data.get("results") if results_data else None,
        "detectedGenes": results_data.get("detectedGenes") if results_data else None,
        "summary": results_data.get("summary") if results_data else None,
    }


# ---------------------------------------------------------------------------
# Activity log
# ---------------------------------------------------------------------------

def log_activity(user_id: Optional[str], user_name: str, action: str, details: Optional[str] = None):
    conn = get_db()
    conn.execute(
        "INSERT INTO activity_log (user_id, user_name, action, details, timestamp) VALUES (?, ?, ?, ?, ?)",
        (user_id, user_name, action, details, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_recent_activity(limit: int = 20) -> List[Dict]:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM activity_log ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [
        {
            "userId": r["user_id"],
            "userName": r["user_name"],
            "action": r["action"],
            "details": r["details"],
            "timestamp": r["timestamp"],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Dashboard stats
# ---------------------------------------------------------------------------

def get_dashboard_stats() -> Dict:
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    resistant = conn.execute("SELECT COUNT(*) FROM predictions WHERE risk_level = 'high'").fetchone()[0]
    susceptible = conn.execute("SELECT COUNT(*) FROM predictions WHERE risk_level IN ('low', 'minimal')").fetchone()[0]
    pending = conn.execute("SELECT COUNT(*) FROM predictions WHERE status IN ('pending', 'processing')").fetchone()[0]

    week_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
    two_weeks_ago = (datetime.utcnow() - timedelta(days=14)).isoformat()
    this_week = conn.execute("SELECT COUNT(*) FROM predictions WHERE created_at >= ?", (week_ago,)).fetchone()[0]
    last_week = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE created_at >= ? AND created_at < ?",
        (two_weeks_ago, week_ago),
    ).fetchone()[0]
    weekly_change = this_week - last_week if last_week else 0

    this_week_r = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE risk_level = 'high' AND created_at >= ?", (week_ago,)
    ).fetchone()[0]
    last_week_r = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE risk_level = 'high' AND created_at >= ? AND created_at < ?",
        (two_weeks_ago, week_ago),
    ).fetchone()[0]
    weekly_r_change = this_week_r - last_week_r if last_week_r else 0

    conn.close()
    return {
        "totalPredictions": total,
        "resistantCount": resistant,
        "susceptibleCount": susceptible,
        "pendingCount": pending,
        "weeklyChange": {
            "predictions": weekly_change,
            "resistant": weekly_r_change,
        },
    }


def get_resistance_overview() -> List[Dict]:
    conn = get_db()
    resistant = conn.execute("SELECT COUNT(*) FROM predictions WHERE risk_level = 'high'").fetchone()[0]
    moderate = conn.execute("SELECT COUNT(*) FROM predictions WHERE risk_level = 'moderate'").fetchone()[0]
    susceptible = conn.execute("SELECT COUNT(*) FROM predictions WHERE risk_level IN ('low', 'minimal')").fetchone()[0]
    conn.close()
    return [
        {"name": "Resistant", "value": resistant, "color": "#ef4444"},
        {"name": "Intermediate", "value": moderate, "color": "#eab308"},
        {"name": "Susceptible", "value": susceptible, "color": "#22c55e"},
    ]


def get_trends() -> List[Dict]:
    conn = get_db()
    trends = []
    for i in range(6, -1, -1):
        day = datetime.utcnow() - timedelta(days=i)
        day_start = day.strftime("%Y-%m-%dT00:00:00")
        day_end = day.strftime("%Y-%m-%dT23:59:59")
        label = day.strftime("%b %d")

        r = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE risk_level = 'high' AND created_at BETWEEN ? AND ?",
            (day_start, day_end),
        ).fetchone()[0]
        s = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE risk_level IN ('low', 'minimal') AND created_at BETWEEN ? AND ?",
            (day_start, day_end),
        ).fetchone()[0]
        m = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE risk_level = 'moderate' AND created_at BETWEEN ? AND ?",
            (day_start, day_end),
        ).fetchone()[0]

        trends.append({"date": label, "resistant": r, "susceptible": s, "intermediate": m})

    conn.close()
    return trends


# ---------------------------------------------------------------------------
# Admin stats
# ---------------------------------------------------------------------------

def get_admin_stats() -> Dict:
    conn = get_db()
    total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    week_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
    active_users = conn.execute(
        "SELECT COUNT(*) FROM users WHERE last_login >= ?", (week_ago,)
    ).fetchone()[0]
    total_predictions = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    today_start = datetime.utcnow().strftime("%Y-%m-%dT00:00:00")
    predictions_today = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE created_at >= ?", (today_start,)
    ).fetchone()[0]

    # Estimate storage from file sizes
    storage_row = conn.execute("SELECT COALESCE(SUM(file_size), 0) FROM predictions").fetchone()
    storage_bytes = storage_row[0]
    storage_gb = round(storage_bytes / (1024**3), 2)

    conn.close()
    return {
        "totalUsers": total_users,
        "activeUsers": active_users,
        "totalPredictions": total_predictions,
        "predictionsToday": predictions_today,
        "storageUsed": storage_gb,
        "storageLimit": 10,
    }
