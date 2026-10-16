"""SQLite Database Layer for StayWise Retention Management.

Provides local persistent storage for retention tasks, notes, customer interactions,
and workflow status tracking.
"""

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Any

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "retention.db"


def get_db_connection() -> sqlite3.Connection:
    """Create a thread-safe connection with dictionary-like row access."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize database tables with indexes."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Retention tasks table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS retention_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT NOT NULL,
                task_type TEXT NOT NULL,
                priority TEXT NOT NULL,
                status TEXT NOT NULL,
                assigned_to TEXT NOT NULL,
                notes TEXT,
                due_date TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT
            )
            """
        )

        # Retention notes log table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS retention_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT NOT NULL,
                author TEXT NOT NULL,
                note TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

        # Indexes for fast lookup
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tasks_cust ON retention_tasks(customer_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON retention_tasks(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_notes_cust ON retention_notes(customer_id)")
        
        # Seed default seed tasks if table is empty to provide realistic enterprise demo data
        cursor.execute("SELECT COUNT(*) FROM retention_tasks")
        if cursor.fetchone()[0] == 0:
            _seed_initial_tasks(cursor)
            
        conn.commit()


def _seed_initial_tasks(cursor: sqlite3.Cursor) -> None:
    """Seed initial sample tasks for demonstration."""
    now = datetime.now(timezone.utc).isoformat()
    demo_tasks = [
        (
            "9237-HQITU",
            "Contract Upgrade Offer",
            "High",
            "Open",
            "Sarah Chen (CSM)",
            "Customer is on Month-to-month with Fiber Optic and no tech support. High churn probability. Reach out with 15% 1-year bundle incentive.",
            "2026-10-25",
            now,
            now,
            None,
        ),
        (
            "9305-CDSKC",
            "Auto-Pay Migration Bonus",
            "High",
            "In Progress",
            "Alex Rivera (Retention Specialist)",
            "Electronic check payer with $99.65/mo charges. Contacted via phone; offered $5 recurring auto-pay credit.",
            "2026-10-22",
            now,
            now,
            None,
        ),
        (
            "1452-KIOVK",
            "Value Add-on Onboarding",
            "Medium",
            "Completed",
            "Michael Scott (Account Executive)",
            "Enrolled subscriber in complimentary 6-month Tech Support pilot to reduce friction.",
            "2026-10-18",
            now,
            now,
            now,
        ),
    ]
    cursor.executemany(
        """
        INSERT INTO retention_tasks (
            customer_id, task_type, priority, status, assigned_to, notes, due_date, created_at, updated_at, completed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        demo_tasks,
    )

    demo_notes = [
        (
            "9237-HQITU",
            "Sarah Chen (CSM)",
            "Identified as high hazard risk (< 2 months tenure, Fiber optic without tech support). Triage scheduled.",
            now,
        ),
        (
            "9305-CDSKC",
            "Alex Rivera (Retention Specialist)",
            "Left voicemail regarding payment method discount and bill review.",
            now,
        ),
    ]
    cursor.executemany(
        """
        INSERT INTO retention_notes (customer_id, author, note, created_at)
        VALUES (?, ?, ?, ?)
        """,
        demo_notes,
    )


def list_tasks(
    status: str | None = None,
    priority: str | None = None,
    customer_id: str | None = None,
) -> list[dict[str, Any]]:
    """Query retention tasks with optional filters."""
    query = "SELECT * FROM retention_tasks WHERE 1=1"
    params: list[Any] = []

    if status and status != "All":
        query += " AND status = ?"
        params.append(status)
    if priority and priority != "All":
        query += " AND priority = ?"
        params.append(priority)
    if customer_id:
        query += " AND customer_id = ?"
        params.append(customer_id)

    query += " ORDER BY id DESC"

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_task(task_id: int) -> dict[str, Any] | None:
    """Retrieve a single task by ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM retention_tasks WHERE id = ?", (task_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def create_task(
    customer_id: str,
    task_type: str,
    priority: str,
    assigned_to: str,
    notes: str = "",
    due_date: str = "",
) -> dict[str, Any]:
    """Create a new retention task."""
    now = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO retention_tasks (
                customer_id, task_type, priority, status, assigned_to, notes, due_date, created_at, updated_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (customer_id, task_type, priority, "Open", assigned_to, notes, due_date, now, now, None),
        )
        task_id = cursor.lastrowid
        conn.commit()
    return get_task(task_id)  # type: ignore


def update_task(
    task_id: int,
    status: str | None = None,
    priority: str | None = None,
    assigned_to: str | None = None,
    notes: str | None = None,
    due_date: str | None = None,
) -> dict[str, Any] | None:
    """Update task attributes."""
    existing = get_task(task_id)
    if not existing:
        return None

    now = datetime.now(timezone.utc).isoformat()
    new_status = status if status is not None else existing["status"]
    new_priority = priority if priority is not None else existing["priority"]
    new_assigned = assigned_to if assigned_to is not None else existing["assigned_to"]
    new_notes = notes if notes is not None else existing["notes"]
    new_due = due_date if due_date is not None else existing["due_date"]
    
    completed_at = existing["completed_at"]
    if new_status == "Completed" and existing["status"] != "Completed":
        completed_at = now
    elif new_status != "Completed":
        completed_at = None

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE retention_tasks
            SET status = ?, priority = ?, assigned_to = ?, notes = ?, due_date = ?, updated_at = ?, completed_at = ?
            WHERE id = ?
            """,
            (new_status, new_priority, new_assigned, new_notes, new_due, now, completed_at, task_id),
        )
        conn.commit()

    return get_task(task_id)


def delete_task(task_id: int) -> bool:
    """Delete a task by ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM retention_tasks WHERE id = ?", (task_id,))
        conn.commit()
        return cursor.rowcount > 0


def list_notes(customer_id: str) -> list[dict[str, Any]]:
    """List all notes logged for a specific customer."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM retention_notes WHERE customer_id = ? ORDER BY id DESC",
            (customer_id,),
        )
        return [dict(row) for row in cursor.fetchall()]


def add_note(customer_id: str, author: str, note: str) -> dict[str, Any]:
    """Add a customer activity note."""
    now = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO retention_notes (customer_id, author, note, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (customer_id, author, note, now),
        )
        note_id = cursor.lastrowid
        conn.commit()
        cursor.execute("SELECT * FROM retention_notes WHERE id = ?", (note_id,))
        return dict(cursor.fetchone())
