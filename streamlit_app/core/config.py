from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

import streamlit as st


@dataclass(frozen=True)
class SupabaseStorageConfig:
    url: str
    service_role_key: str
    bucket: str
    prefix: str = "streamlit-default-test-file"


def _value(
    name: str,
    *,
    secrets: Mapping[str, object] | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    environment = os.environ if environ is None else environ
    secret_mapping = secrets
    if secret_mapping is None:
        try:
            secret_mapping = st.secrets
        except Exception:
            secret_mapping = {}
    try:
        secret_value = secret_mapping.get(name) if secret_mapping is not None else None
    except Exception:
        secret_value = None
    if secret_value is not None and str(secret_value).strip():
        return str(secret_value).strip()
    return str(environment.get(name, "")).strip()


def configured_api_keys(
    *,
    secrets: Mapping[str, object] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    values = {
        "OpenAI": _value("OPENAI_API_KEY", secrets=secrets, environ=environ),
        "Gemini": _value("GEMINI_API_KEY", secrets=secrets, environ=environ),
    }
    return {provider: value for provider, value in values.items() if value}


def default_ai_provider(api_keys: Mapping[str, str] | None = None) -> str:
    available = configured_api_keys() if api_keys is None else api_keys
    if available.get("OpenAI"):
        return "OpenAI"
    if available.get("Gemini"):
        return "Gemini"
    return "OpenAI"


def supabase_storage_config(
    *,
    secrets: Mapping[str, object] | None = None,
    environ: Mapping[str, str] | None = None,
) -> SupabaseStorageConfig | None:
    url = _value("SUPABASE_URL", secrets=secrets, environ=environ).rstrip("/")
    key = _value("SUPABASE_SERVICE_ROLE_KEY", secrets=secrets, environ=environ)
    bucket = _value("SUPABASE_STORAGE_BUCKET", secrets=secrets, environ=environ)
    prefix = _value("SUPABASE_STORAGE_PREFIX", secrets=secrets, environ=environ)
    if not (url and key and bucket):
        return None
    return SupabaseStorageConfig(
        url=url,
        service_role_key=key,
        bucket=bucket,
        prefix=(prefix or "streamlit-default-test-file").strip("/"),
    )
