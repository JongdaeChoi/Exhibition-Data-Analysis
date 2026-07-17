from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd
import requests


CSV_ENCODINGS = ("utf-8-sig", "cp949", "utf-8")
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
MAX_DRIVE_FILE_BYTES = 50 * 1024 * 1024


class DataLoadError(ValueError):
    """Raised when an uploaded dataset cannot be loaded safely."""


@dataclass(frozen=True)
class LoadedDataset:
    filename: str
    df: pd.DataFrame
    df_clean: pd.DataFrame


def _read_csv(raw: bytes) -> pd.DataFrame:
    last_error: UnicodeDecodeError | None = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise DataLoadError("CSV 파일의 문자 인코딩을 확인할 수 없습니다.") from last_error


def load_table(raw: bytes, filename: str) -> LoadedDataset:
    """Load CSV/Excel bytes and create independent source/working frames."""
    if not raw:
        raise DataLoadError("선택한 파일이 비어 있습니다.")

    safe_filename = Path(filename or "uploaded_data").name
    extension = Path(safe_filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise DataLoadError("CSV, XLSX 또는 XLS 파일만 적재할 수 있습니다.")

    try:
        if extension == ".csv":
            loaded = _read_csv(raw)
        else:
            loaded = pd.read_excel(io.BytesIO(raw))
    except DataLoadError:
        raise
    except Exception as exc:
        raise DataLoadError(f"파일을 읽는 중 오류가 발생했습니다: {exc}") from exc

    # `loaded` is local to this function, so it can become the session-owned
    # source frame directly. Only one deep copy is required for the workspace.
    df = loaded
    df_clean = df.copy(deep=True)
    return LoadedDataset(filename=safe_filename, df=df, df_clean=df_clean)


def extract_google_drive_file_id(url: str) -> str:
    """Extract a file id only from supported Google Drive sharing URLs."""
    parsed = urlparse((url or "").strip())
    if parsed.scheme != "https" or parsed.hostname not in {
        "drive.google.com",
        "docs.google.com",
    }:
        raise DataLoadError("Google Drive의 HTTPS 공유 링크를 입력하세요.")

    match = re.search(r"/d/([a-zA-Z0-9_-]+)", parsed.path)
    if match:
        return match.group(1)
    file_id = parse_qs(parsed.query).get("id", [""])[0]
    if re.fullmatch(r"[a-zA-Z0-9_-]+", file_id):
        return file_id
    raise DataLoadError("공유 링크에서 Google Drive 파일 ID를 찾을 수 없습니다.")


def download_google_drive_file(url: str, timeout: int = 30) -> tuple[bytes, str]:
    """Download a publicly shared Drive file with a bounded response size."""
    file_id = extract_google_drive_file_id(url)
    endpoint = "https://drive.usercontent.google.com/download"
    try:
        with requests.get(
            endpoint,
            params={"id": file_id, "export": "download", "confirm": "t"},
            timeout=timeout,
            stream=True,
            allow_redirects=True,
        ) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if "text/html" in content_type:
                raise DataLoadError(
                    "파일을 내려받지 못했습니다. Drive 공유 권한을 '링크가 있는 모든 사용자'로 설정하세요."
                )
            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_DRIVE_FILE_BYTES:
                    raise DataLoadError("Drive 파일은 최대 50MB까지 적재할 수 있습니다.")
                chunks.append(chunk)

            disposition = response.headers.get("content-disposition", "")
            filename_match = re.search(r'filename\*?=(?:UTF-8\'\')?["\']?([^"\';]+)', disposition)
            filename = unquote(filename_match.group(1)) if filename_match else "google_drive_file"
            return b"".join(chunks), filename
    except DataLoadError:
        raise
    except requests.RequestException as exc:
        raise DataLoadError(f"Google Drive 파일 다운로드에 실패했습니다: {exc}") from exc
