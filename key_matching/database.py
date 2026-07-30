from __future__ import annotations

import json
import datetime
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

from .types import KeyFeatures


class DuplicateKeyError(ValueError):
    pass


class PgRow(dict):
    def __getitem__(self, key):
        val = super().__getitem__(key) if not isinstance(key, int) else list(self.values())[key]
        if isinstance(val, (datetime.datetime, datetime.date)):
            return val.strftime("%Y-%m-%d %H:%M:%S")
        return val

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __iter__(self):
        for val in self.values():
            if isinstance(val, (datetime.datetime, datetime.date)):
                yield val.strftime("%Y-%m-%d %H:%M:%S")
            else:
                yield val


class PgCursorWrapper:
    def __init__(self, cursor):
        self.cursor = cursor

    def fetchone(self):
        row = self.cursor.fetchone()
        if row is None:
            return None
        return PgRow(row)

    def fetchall(self):
        rows = self.cursor.fetchall()
        return [PgRow(r) for r in rows]

    @property
    def rowcount(self):
        return self.cursor.rowcount


class FeatureStore:
    """Transactional PostgreSQL persistence safe for FastAPI's worker threads."""

    def __init__(self, dsn: str | Path, timeout: float = 30) -> None:
        self._lock = RLock()
        self._dsn = str(dsn)
        self._timeout = timeout
        self.connection = psycopg2.connect(self._dsn, connect_timeout=int(self._timeout))
        self._migrate()
        self._index = None
        self._index_ids: list[tuple[str, str]] = []

    def _reconnect(self) -> None:
        try:
            self.connection.close()
        except Exception:
            pass
        self.connection = psycopg2.connect(self._dsn, connect_timeout=int(self._timeout))

    def _execute(self, sql: str, params: tuple = ()) -> PgCursorWrapper:
        sql = sql.replace("?", "%s")
        if "INSERT OR IGNORE INTO key_registrations" in sql:
            sql = sql.replace(
                "INSERT OR IGNORE INTO key_registrations",
                "INSERT INTO key_registrations"
            )
            if "ON CONFLICT" not in sql:
                sql += " ON CONFLICT (key_id) DO NOTHING"
        
        try:
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            cursor.execute(sql, params)
            return PgCursorWrapper(cursor)
        except (psycopg2.InterfaceError, psycopg2.OperationalError):
            # Try to reconnect and retry execution once
            try:
                self._reconnect()
                cursor = self.connection.cursor(cursor_factory=RealDictCursor)
                cursor.execute(sql, params)
                return PgCursorWrapper(cursor)
            except Exception:
                raise

    def _migrate(self) -> None:
        with self.connection:
            self._execute(
                "CREATE TABLE IF NOT EXISTS key_features ("
                "key_id VARCHAR(255) NOT NULL, side VARCHAR(50) NOT NULL, features TEXT NOT NULL, "
                "front_image TEXT, back_image TEXT, normalized_image TEXT, mask_image TEXT, "
                "metadata TEXT NOT NULL DEFAULT '{}', PRIMARY KEY(key_id, side))"
            )
            self._execute(
                "CREATE TABLE IF NOT EXISTS key_registrations ("
                "key_id VARCHAR(255) PRIMARY KEY, metadata TEXT NOT NULL, "
                "created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                "updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            # Preserve catalogs created before first-class registration records.
            self._execute(
                "INSERT OR IGNORE INTO key_registrations(key_id, metadata) "
                "SELECT key_id, COALESCE(MAX(metadata), '{}') "
                "FROM key_features GROUP BY key_id"
            )

    def ping(self) -> bool:
        with self._lock:
            return self._execute("SELECT 1").fetchone()[0] == 1

    def exists(self, key_id: str) -> bool:
        with self._lock:
            row = self._execute(
                "SELECT 1 FROM key_registrations WHERE key_id=?", (key_id,)
            ).fetchone()
            return row is not None

    def register(
        self,
        key_id: str,
        metadata: dict[str, Any],
        sides: list[tuple[str, KeyFeatures, dict[str, str | None]]],
    ) -> None:
        encoded_metadata = json.dumps(metadata, separators=(",", ":"), sort_keys=True)
        with self._lock:
            try:
                with self.connection:
                    self._execute(
                        "INSERT INTO key_registrations(key_id, metadata) VALUES (?, ?)",
                        (key_id, encoded_metadata),
                    )
                    for side, features, paths in sides:
                        self._execute(
                            "INSERT INTO key_features("
                            "key_id, side, features, front_image, back_image, "
                            "normalized_image, mask_image, metadata"
                            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                key_id,
                                side,
                                json.dumps(features.to_dict(), separators=(",", ":")),
                                paths.get("front_image"),
                                paths.get("back_image"),
                                paths.get("normalized_image"),
                                paths.get("mask_image"),
                                encoded_metadata,
                            ),
                        )
            except psycopg2.IntegrityError as exc:
                raise DuplicateKeyError(key_id) from exc
            self._index = None

    def all(self) -> list[tuple[str, str, KeyFeatures]]:
        with self._lock:
            rows = self._execute(
                "SELECT key_id, side, features FROM key_features"
            ).fetchall()
        return [
            (row["key_id"], row["side"], KeyFeatures.from_dict(json.loads(row["features"])))
            for row in rows
        ]

    def get_registration(self, key_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._execute(
                "SELECT key_id, metadata, created_at, updated_at "
                "FROM key_registrations WHERE key_id=?",
                (key_id,),
            ).fetchone()
            if row is None:
                return None
            sides = [
                item["side"] for item in self._execute(
                    "SELECT side FROM key_features WHERE key_id=? ORDER BY side",
                    (key_id,),
                ).fetchall()
            ]
        return {
            "key_id": row["key_id"],
            "metadata": json.loads(row["metadata"]),
            "sides": sides,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def update_registration(self, key_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        with self._lock:
            row = self._execute(
                "SELECT metadata FROM key_registrations WHERE key_id=?", (key_id,)
            ).fetchone()
            if row is None:
                return None
            
            metadata = json.loads(row["metadata"])
            metadata.update(updates)
            encoded_metadata = json.dumps(metadata, separators=(",", ":"), sort_keys=True)
            
            with self.connection:
                self._execute(
                    "UPDATE key_registrations SET metadata=?, updated_at=CURRENT_TIMESTAMP "
                    "WHERE key_id=?",
                    (encoded_metadata, key_id)
                )
                self._execute(
                    "UPDATE key_features SET metadata=? WHERE key_id=?",
                    (encoded_metadata, key_id)
                )
        return self.get_registration(key_id)

    def update_cross_refs(self, key_id: str, references: list) -> dict[str, Any] | None:
        """Update only the cross_refs field within the key's metadata."""
        with self._lock:
            row = self._execute(
                "SELECT metadata FROM key_registrations WHERE key_id=?", (key_id,)
            ).fetchone()
            if row is None:
                return None
            metadata = json.loads(row["metadata"])
            metadata["cross_refs"] = references
            encoded_metadata = json.dumps(metadata, separators=(",", ":"), sort_keys=True)
            with self.connection:
                self._execute(
                    "UPDATE key_registrations SET metadata=?, updated_at=CURRENT_TIMESTAMP "
                    "WHERE key_id=?",
                    (encoded_metadata, key_id)
                )
                self._execute(
                    "UPDATE key_features SET metadata=? WHERE key_id=?",
                    (encoded_metadata, key_id)
                )
        return self.get_registration(key_id)

    def update_side(
        self,
        key_id: str,
        side: str,
        features: KeyFeatures,
        paths: dict[str, str | None]
    ) -> list[str]:
        encoded_features = json.dumps(features.to_dict(), separators=(",", ":"))
        with self._lock:
            row = self._execute(
                "SELECT front_image, back_image, normalized_image, mask_image "
                "FROM key_features WHERE key_id=? AND side=?",
                (key_id, side)
            ).fetchone()
            
            old_paths = [value for value in row if value] if row else []
            
            reg_row = self._execute(
                "SELECT metadata FROM key_registrations WHERE key_id=?", (key_id,)
            ).fetchone()
            
            metadata = reg_row["metadata"] if reg_row else "{}"
            
            with self.connection:
                if row:
                    self._execute(
                        "UPDATE key_features SET features=?, front_image=?, back_image=?, "
                        "normalized_image=?, mask_image=?, metadata=? "
                        "WHERE key_id=? AND side=?",
                        (
                            encoded_features,
                            paths.get("front_image"),
                            paths.get("back_image"),
                            paths.get("normalized_image"),
                            paths.get("mask_image"),
                            metadata,
                            key_id,
                            side
                        )
                    )
                else:
                    self._execute(
                        "INSERT INTO key_features("
                        "key_id, side, features, front_image, back_image, "
                        "normalized_image, mask_image, metadata"
                        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            key_id,
                            side,
                            encoded_features,
                            paths.get("front_image"),
                            paths.get("back_image"),
                            paths.get("normalized_image"),
                            paths.get("mask_image"),
                            metadata,
                        )
                    )
            
            self._index = None
            return old_paths

    def list_registrations(self, limit: int, offset: int) -> list[dict[str, Any]]:
        with self._lock:
            ids = [
                row["key_id"] for row in self._execute(
                    "SELECT key_id FROM key_registrations "
                    "ORDER BY created_at DESC, key_id LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
            ]
        return [record for key_id in ids if (record := self.get_registration(key_id))]

    def delete(self, key_id: str) -> list[str]:
        with self._lock:
            rows = self._execute(
                "SELECT front_image, back_image, normalized_image, mask_image "
                "FROM key_features WHERE key_id=?",
                (key_id,),
            ).fetchall()
            paths = [value for row in rows for value in row if value]
            with self.connection:
                self._execute("DELETE FROM key_features WHERE key_id=?", (key_id,))
                cursor = self._execute(
                    "DELETE FROM key_registrations WHERE key_id=?", (key_id,)
                )
            if cursor.rowcount:
                self._index = None
            return paths if cursor.rowcount else []

    def build_embedding_index(self) -> bool:
        rows = [(key_id, side, feat) for key_id, side, feat in self.all() if feat.embedding.size]
        if not rows:
            return False
        try:
            import faiss
        except ImportError:
            return False
        vectors = np.stack([row[2].embedding for row in rows]).astype(np.float32)
        faiss.normalize_L2(vectors)
        with self._lock:
            self._index = faiss.IndexFlatIP(vectors.shape[1])
            self._index.add(vectors)
            self._index_ids = [(row[0], row[1]) for row in rows]
        return True

    def embedding_search(
        self, embedding: np.ndarray, limit: int = 10
    ) -> list[tuple[str, str, float]]:
        if self._index is None and not self.build_embedding_index():
            return []
        vector = embedding.reshape(1, -1).astype(np.float32)
        import faiss
        faiss.normalize_L2(vector)
        with self._lock:
            scores, indices = self._index.search(
                vector, min(limit, len(self._index_ids))
            )
            return [
                (self._index_ids[i][0], self._index_ids[i][1], float(score))
                for score, i in zip(scores[0], indices[0])
                if i >= 0
            ]

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    def __enter__(self) -> "FeatureStore":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
