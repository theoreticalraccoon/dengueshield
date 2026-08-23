# Deploying DengueShield

The app is deploy-ready: **26.7 MB across 56 files**, well inside every free tier. The
1.2 GB of raw training data in `data/raw/` is **not** needed at runtime and is excluded
by `.gitignore` — the app only loads pre-trained models and pre-computed reports.

## What ships

| Path | Purpose | Size |
|---|---|---|
| `app.py` | the Streamlit application (3 screens) | 26 KB |
| `models/*.joblib` | screening, complication, continuation, emergence models | 24 MB |
| `reports/` | pre-computed metrics, forecasts, audit tables | 1.5 MB |
| `data/processed/` | district history for the trend charts | 1.9 MB |
| `requirements.txt` | runtime dependencies only | — |
| `.streamlit/config.toml` | theme + server settings | — |

`models/model2_lstm.pt` is excluded — it is a training-time benchmark the app never
loads.

---

## Option A — Streamlit Community Cloud (free, recommended)

**You must do steps 1–2 yourself; they require your GitHub and Streamlit accounts.**

1. **Push to GitHub**

   ```bash
   cd c:/Users/HP/DengueShield
   git init
   git add .
   git commit -m "DengueShield: dengue screening and outbreak early warning"
   git branch -M main
   git remote add origin https://github.com/<your-username>/dengueshield.git
   git push -u origin main
   ```

   `.gitignore` already excludes `data/raw/`, `.venv/` and the LSTM artifact, so the
   push is ~27 MB.

2. **Deploy** — go to <https://share.streamlit.io>, sign in with GitHub,
   **New app**, select your repo, set:

   - Branch: `main`
   - Main file path: `app.py`
   - Python version: `3.11`

   Click **Deploy**. First build takes ~3–5 minutes.

3. **Your link** will be:

   ```
   https://<your-username>-dengueshield-app-<hash>.streamlit.app
   ```

   Streamlit shows the exact URL once the build finishes. You can set a custom
   subdomain under **Settings → General → App URL**.

---

## Option B — Hugging Face Spaces (free)

1. Create a Space at <https://huggingface.co/new-space>, SDK **Streamlit**, hardware
   **CPU basic**.
2. Push:

   ```bash
   git remote add hf https://huggingface.co/spaces/<your-username>/dengueshield
   git push hf main
   ```

3. Link: `https://huggingface.co/spaces/<your-username>/dengueshield`

Spaces needs `app.py` at the repo root — it already is.

---

## Option C — Docker (any host)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY models/ models/
COPY reports/ reports/
COPY data/processed/ data/processed/
COPY .streamlit/ .streamlit/
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t dengueshield .
docker run -p 8501:8501 dengueshield
```

Deploy that image to Render, Railway, Fly.io, Cloud Run or any container host.

---

## Run locally right now

```bash
cd c:/Users/HP/DengueShield
.venv/Scripts/python.exe -m streamlit run app.py
```

Opens at <http://localhost:8501>. Streamlit also prints a Network URL
(`http://192.168.x.x:8501`) that other devices on your Wi-Fi can open — useful for a
demo without deploying anything.

---

## Refreshing the forecasts

The app reads pre-computed forecasts. To regenerate them against newer surveillance
data:

```bash
.venv/Scripts/python.exe finalize_srilanka.py                                  # continuation
.venv/Scripts/python.exe finalize_emergence.py                                    # emergence
```

Both rewrite `reports/srilanka_dual_risk.csv`, which the app picks up on next load.
This needs `data/raw/`, so run it locally and commit the refreshed CSVs — the deployed
app never downloads the raw datasets.

---

## Pre-flight checklist

- [x] All three screens render without exception (verified via `AppTest`)
- [x] Both assessment buttons produce predictions
- [x] `ruff check` passes with zero errors
- [x] Runtime deps pinned in `requirements.txt` (no torch/shap/xgboost)
- [x] Raw data excluded from the repo
- [x] No secrets, API keys or credentials anywhere in the codebase
- [x] Medical disclaimer on every prediction surface
