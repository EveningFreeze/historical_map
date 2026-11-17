import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TaskStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self):
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    processed_path TEXT,
                    result_path TEXT,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    notes TEXT DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    score REAL NOT NULL,
                    rec_confidence REAL,
                    polygon_json TEXT NOT NULL,
                    source_tile_json TEXT,
                    corrected_text TEXT,
                    FOREIGN KEY(task_id) REFERENCES tasks(id)
                );
                """
            )

    def create_task(self, task_id: str, source_path: Path, mode: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO tasks(id, created_at, source_path, mode, status) VALUES (?, ?, ?, ?, ?)",
                (task_id, utc_now(), str(source_path), mode, "running"),
            )

    def finish_task(self, task_id: str, processed_path: Path, result_path: Path, status: str = "completed") -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE tasks SET processed_path=?, result_path=?, status=? WHERE id=?",
                (str(processed_path), str(result_path), status, task_id),
            )

    def save_detections(self, task_id: str, detections: Iterable[dict]) -> None:
        rows = []
        for ordinal, item in enumerate(detections, start=1):
            rows.append(
                (
                    task_id,
                    ordinal,
                    str(item.get("text", "")),
                    float(item.get("score", 0.0)),
                    float(item.get("rec_confidence", 0.0)),
                    json.dumps(item.get("polygon", []), ensure_ascii=False),
                    json.dumps(item.get("source_tile"), ensure_ascii=False),
                    str(item.get("text_correction", {}).get("text", "")),
                )
            )
        with self._connect() as connection:
            connection.executemany(
                """INSERT INTO detections(
                    task_id, ordinal, text, score, rec_confidence,
                    polygon_json, source_tile_json, corrected_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )

    def recent_tasks(self, limit: int = 30):
        with self._connect() as connection:
            return connection.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
