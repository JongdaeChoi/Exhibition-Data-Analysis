from __future__ import annotations

import datetime as dt
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.config import SupabaseStorageConfig


class PersistentStorageError(RuntimeError):
    """Raised when the server-side default test file cannot be persisted."""


@dataclass(frozen=True)
class StoredTestFile:
    filename: str
    raw: bytes
    rows: int
    columns: int
    saved_at: str


class SupabaseDefaultFileStore:
    """Private Supabase Storage adapter used only by the Streamlit server."""

    def __init__(self, config: SupabaseStorageConfig, client: Any | None = None):
        self.config = config
        self._client = client

    def _bucket(self):
        if self._client is None:
            try:
                from supabase import create_client
            except ImportError as exc:
                raise PersistentStorageError(
                    "Supabase client is not installed. Reinstall streamlit_app/requirements.txt."
                ) from exc
            self._client = create_client(
                self.config.url,
                self.config.service_role_key,
            )
        return self._client.storage.from_(self.config.bucket)

    @property
    def manifest_path(self) -> str:
        return f"{self.config.prefix}/manifest.json"

    def _data_path(self, filename: str) -> str:
        suffix = Path(filename).suffix.lower()
        return f"{self.config.prefix}/default_test_file{suffix}"

    @staticmethod
    def _download_bytes(bucket, path: str) -> bytes:
        value = bucket.download(path)
        if isinstance(value, bytes):
            return value
        if hasattr(value, "read"):
            return value.read()
        return bytes(value)

    def load(self) -> StoredTestFile | None:
        bucket = self._bucket()
        try:
            manifest_raw = self._download_bytes(bucket, self.manifest_path)
        except Exception as exc:
            text = str(exc).casefold()
            if any(token in text for token in ("not found", "404", "does not exist")):
                return None
            raise PersistentStorageError(
                "저장된 기본 테스트 파일 정보를 불러오지 못했습니다."
            ) from exc
        try:
            manifest = json.loads(manifest_raw.decode("utf-8"))
            raw = self._download_bytes(bucket, str(manifest["object_path"]))
            return StoredTestFile(
                filename=Path(str(manifest["filename"])).name,
                raw=raw,
                rows=int(manifest["rows"]),
                columns=int(manifest["columns"]),
                saved_at=str(manifest["saved_at"]),
            )
        except PersistentStorageError:
            raise
        except Exception as exc:
            raise PersistentStorageError(
                "저장된 기본 테스트 파일이 손상되었거나 읽을 수 없습니다."
            ) from exc

    def save(self, *, filename: str, raw: bytes, rows: int, columns: int) -> StoredTestFile:
        if not raw:
            raise PersistentStorageError("빈 파일은 기본 테스트 파일로 저장할 수 없습니다.")
        safe_filename = Path(filename).name
        object_path = self._data_path(safe_filename)
        saved_at = dt.datetime.now(dt.timezone.utc).isoformat()
        manifest = {
            "schema_version": 1,
            "filename": safe_filename,
            "object_path": object_path,
            "rows": int(rows),
            "columns": int(columns),
            "saved_at": saved_at,
        }
        bucket = self._bucket()
        try:
            current = self.load()
            if current is not None:
                current_path = self._data_path(current.filename)
                if current_path != object_path:
                    bucket.remove([current_path])
            content_type = mimetypes.guess_type(safe_filename)[0] or "application/octet-stream"
            bucket.upload(
                path=object_path,
                file=raw,
                file_options={"content-type": content_type, "upsert": "true"},
            )
            bucket.upload(
                path=self.manifest_path,
                file=json.dumps(manifest, ensure_ascii=False).encode("utf-8"),
                file_options={"content-type": "application/json", "upsert": "true"},
            )
        except PersistentStorageError:
            raise
        except Exception as exc:
            raise PersistentStorageError(
                "기본 테스트 파일을 Supabase Storage에 저장하지 못했습니다."
            ) from exc
        return StoredTestFile(
            filename=safe_filename,
            raw=bytes(raw),
            rows=int(rows),
            columns=int(columns),
            saved_at=saved_at,
        )

    def delete(self) -> None:
        bucket = self._bucket()
        try:
            current = self.load()
            paths = [self.manifest_path]
            if current is not None:
                paths.insert(0, self._data_path(current.filename))
            bucket.remove(paths)
        except PersistentStorageError:
            raise
        except Exception as exc:
            raise PersistentStorageError(
                "기본 테스트 파일을 Supabase Storage에서 삭제하지 못했습니다."
            ) from exc
