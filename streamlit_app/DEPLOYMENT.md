# Streamlit deployment

## Local execution

From the repository root:

```bash
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` for local
development. The real secrets file is ignored by Git.

## Server secrets

Configure these in Streamlit Community Cloud **Settings → Secrets** or as
server environment variables:

```toml
OPENAI_API_KEY = "..."
GEMINI_API_KEY = "..."

SUPABASE_URL = "https://PROJECT.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = "..."
SUPABASE_STORAGE_BUCKET = "private-test-files"
SUPABASE_STORAGE_PREFIX = "streamlit-default-test-file"
```

- OpenAI is selected by default when both AI keys exist.
- If only the Gemini key exists, Gemini is selected automatically.
- AI keys and Supabase credentials are never entered in the UI or included in
  saved conversations.
- The service-role key must remain server-side. Never commit it or expose it in
  browser code.

## Supabase Storage

Create `private-test-files` as a **private** bucket in the developer-owned
Supabase project before deployment. The app server uses the service-role key to
load, replace, and delete one default test file plus its manifest. End users do
not sign in to Supabase and receive no storage credentials.

The app does not use Streamlit server-local disk as permanent storage. When the
Supabase settings are missing, persistent save/delete controls are disabled,
while upload, preprocessing, structured ChartSpec visualization, and downloads
continue to work.

## Streamlit Community Cloud

- Repository: this GitHub repository
- Branch: `main`
- Main file path: `streamlit_app/app.py`
- Python dependencies: `streamlit_app/requirements.txt`
- Secrets: add the values listed above in the app settings

## Colab launcher

`streamlit_app/Streamlit_Colab_Test.ipynb` remains a launcher for testing the
same independent app in Colab. It reads `OPENAI_API_KEY` and `GEMINI_API_KEY`
from Colab Secrets (with legacy `exhibition` as a Gemini fallback), passes only
the available values to the Streamlit subprocess, and never prints the values.
If the four optional `SUPABASE_*` secrets are registered in Colab, the launcher
also passes them so persistent default-file operations can be tested there.
