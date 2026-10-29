from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from . import db
from .config import DAMAGE_WEIGHTS, PARTS_WEIGHTS, STATUSES
from .detector import Detector
from .pipeline import run_inspection
from .report import build_report, export_pdf


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="Car Damage Inspector", version="0.1.0", lifespan=lifespan)


@lru_cache(maxsize=1)
def get_detector() -> Detector:
    return Detector(PARTS_WEIGHTS, DAMAGE_WEIGHTS)


class StatusUpdate(BaseModel):
    status: str


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/inspections")
def create_inspection(files: list[UploadFile] = File(...), car_label: str = Form("unlabeled")):
    """Upload one or more photos of the same car. Returns the merged damage report."""
    images = []
    for f in files:
        data = f.file.read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, f"{f.filename} is not a readable image")
        images.append((f.filename or f"image_{len(images)}.jpg", img))
    try:
        detector = get_detector()
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    findings, annotated = run_inspection(detector, images)
    report = build_report(findings)
    inspection_id = db.save_inspection(car_label, report, annotated)
    return {"id": inspection_id, "car_label": car_label, "status": "pending", "report": report}


@app.get("/inspections")
def list_inspections():
    return db.list_inspections()


@app.get("/inspections/{inspection_id}")
def get_inspection(inspection_id: int):
    record = db.get_inspection(inspection_id)
    if record is None:
        raise HTTPException(404, "Inspection not found")
    record.pop("image_files")
    return record


@app.patch("/inspections/{inspection_id}/status")
def update_status(inspection_id: int, body: StatusUpdate):
    if body.status not in STATUSES:
        raise HTTPException(400, f"status must be one of {STATUSES}")
    if not db.set_status(inspection_id, body.status):
        raise HTTPException(404, "Inspection not found")
    return {"id": inspection_id, "status": body.status}


@app.get("/inspections/{inspection_id}/report.pdf")
def report_pdf(inspection_id: int):
    record = db.get_inspection(inspection_id)
    if record is None:
        raise HTTPException(404, "Inspection not found")
    annotated = {Path(p).name: Path(p).read_bytes() for p in record["image_files"]}
    pdf = export_pdf(record["car_label"], record["report"], annotated, record["created_at"])
    return Response(pdf, media_type="application/pdf")
