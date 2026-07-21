from __future__ import annotations

import json

from core.config import (
    SupabaseStorageConfig,
    configured_api_keys,
    default_ai_provider,
    supabase_storage_config,
)
from data.storage import SupabaseDefaultFileStore


def test_api_keys_prefer_secrets_and_openai_is_default() -> None:
    keys = configured_api_keys(
        secrets={"OPENAI_API_KEY": "secret-openai", "GEMINI_API_KEY": "secret-gemini"},
        environ={"OPENAI_API_KEY": "env-openai"},
    )
    assert keys == {"OpenAI": "secret-openai", "Gemini": "secret-gemini"}
    assert default_ai_provider(keys) == "OpenAI"
    assert default_ai_provider({"Gemini": "gemini"}) == "Gemini"
    assert default_ai_provider({}) == "OpenAI"


def test_storage_config_requires_all_server_credentials() -> None:
    assert supabase_storage_config(secrets={}, environ={}) is None
    config = supabase_storage_config(
        secrets={
            "SUPABASE_URL": "https://project.supabase.co/",
            "SUPABASE_SERVICE_ROLE_KEY": "server-only",
            "SUPABASE_STORAGE_BUCKET": "private-test-files",
        },
        environ={},
    )
    assert config is not None
    assert config.url == "https://project.supabase.co"
    assert config.bucket == "private-test-files"


class _FakeBucket:
    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def download(self, path: str) -> bytes:
        if path not in self.objects:
            raise RuntimeError("404 not found")
        return self.objects[path]

    def upload(self, *, path: str, file: bytes, file_options: dict) -> None:
        assert file_options.get("upsert") == "true"
        self.objects[path] = bytes(file)

    def remove(self, paths: list[str]) -> None:
        for path in paths:
            self.objects.pop(path, None)


class _FakeStorage:
    def __init__(self, bucket: _FakeBucket):
        self.bucket = bucket

    def from_(self, name: str) -> _FakeBucket:
        assert name == "private-test-files"
        return self.bucket


class _FakeClient:
    def __init__(self, bucket: _FakeBucket):
        self.storage = _FakeStorage(bucket)


def test_supabase_default_file_save_load_replace_delete() -> None:
    bucket = _FakeBucket()
    store = SupabaseDefaultFileStore(
        SupabaseStorageConfig(
            url="https://project.supabase.co",
            service_role_key="server-only",
            bucket="private-test-files",
        ),
        client=_FakeClient(bucket),
    )
    assert store.load() is None

    first = store.save(filename="first.csv", raw=b"a\n1\n", rows=1, columns=1)
    assert first.filename == "first.csv"
    assert store.load().raw == b"a\n1\n"

    second = store.save(filename="second.xlsx", raw=b"xlsx", rows=2, columns=3)
    manifest = json.loads(bucket.objects[store.manifest_path])
    assert second.filename == "second.xlsx"
    assert manifest["rows"] == 2
    assert "default_test_file.csv" not in bucket.objects
    assert store.load().columns == 3

    store.delete()
    assert store.load() is None

