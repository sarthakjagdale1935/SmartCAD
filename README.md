# SmartCAD-AI FastAPI + React

This workspace turns `SmartCAD_AI.ipynb` into a small full-stack app:

- FastAPI backend with the notebook dataset, 12-rule engine, GBM classifier, and fusion verdict.
- React frontend for product selection, CAD parameters, sample cases, and validation results.

## Backend

```powershell
cd backend
uv sync
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The API exposes:

- `GET /health`
- `GET /metadata`
- `POST /validate`
- `POST /validate/batch`

## Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. The Vite dev server proxies `/api/*` calls to the backend.

