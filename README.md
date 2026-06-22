# SmartCAD-AI Backend

FastAPI service extracted from `SmartCAD_AI.ipynb`.

## Run

```powershell
uv sync
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## Endpoints

- `GET /health`
- `GET /metadata`
- `POST /validate`
- `POST /validate/batch`

