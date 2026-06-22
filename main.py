from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from .smartcad import MODEL_CARD, metadata, validate_design


ProductType = Literal["LIGHTING", "EV", "ADAS", "STRUCTURAL"]


class DesignValidationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    design_id: str = Field(default="CUSTOM-001", min_length=1, max_length=80)
    product_type: ProductType
    description: str | None = Field(default=None, max_length=240)
    features: dict[str, float]


class BatchValidationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    designs: list[DesignValidationRequest] = Field(min_length=1, max_length=50)


app = FastAPI(
    title="SmartCAD-AI API",
    description="FastAPI integration for the SmartCAD_AI notebook validation pipeline.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model": MODEL_CARD["name"], "dataset_size": MODEL_CARD["dataset_size"]}


@app.get("/metadata")
def get_metadata() -> dict[str, Any]:
    return metadata()


@app.post("/validate")
def validate(payload: DesignValidationRequest) -> dict[str, Any]:
    try:
        return validate_design(
            features=payload.features,
            product_type=payload.product_type,
            design_id=payload.design_id,
            description=payload.description,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/validate/batch")
def validate_batch(payload: BatchValidationRequest) -> dict[str, Any]:
    results = [validate(design) for design in payload.designs]
    return {"count": len(results), "results": results}

