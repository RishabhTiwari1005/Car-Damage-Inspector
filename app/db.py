import re
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import DateTime, Float, ForeignKey, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from .config import DATA_DIR, DB_URL, STATUSES


class Base(DeclarativeBase):
    pass


class Inspection(Base):
    __tablename__ = "inspections"

    id: Mapped[int] = mapped_column(primary_key=True)
    car_label: Mapped[str] = mapped_column(String(120))
    status: Mapped[str] = mapped_column(String(20), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    overall_severity: Mapped[str] = mapped_column(String(20))
    total_low: Mapped[float] = mapped_column(Float)
    total_high: Mapped[float] = mapped_column(Float)
    currency: Mapped[str] = mapped_column(String(8))
    findings: Mapped[list["FindingRow"]] = relationship(
        back_populates="inspection", cascade="all, delete-orphan"
    )


class FindingRow(Base):
    __tablename__ = "findings"

    id: Mapped[int] = mapped_column(primary_key=True)
    inspection_id: Mapped[int] = mapped_column(ForeignKey("inspections.id"))
    part: Mapped[str] = mapped_column(String(80))
    damage: Mapped[str] = mapped_column(String(80))
    severity: Mapped[str] = mapped_column(String(20))
    confidence: Mapped[float] = mapped_column(Float)
    area_pct: Mapped[float] = mapped_column(Float)
    action: Mapped[str] = mapped_column(String(200))
    cost_low: Mapped[float] = mapped_column(Float)
    cost_high: Mapped[float] = mapped_column(Float)
    images: Mapped[str] = mapped_column(String(1000), default="")
    inspection: Mapped["Inspection"] = relationship(back_populates="findings")


_Session = None


def init_db(url: str | None = None):
    """Create tables. Call once at startup. Pass a url to use a different database (tests)."""
    global _Session
    if url is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url or DB_URL)
    Base.metadata.create_all(engine)
    _Session = sessionmaker(engine, expire_on_commit=False)


@contextmanager
def _session():
    if _Session is None:
        init_db()
    with _Session() as s:
        yield s


def _safe_name(i: int, name: str) -> str:
    return f"{i:02d}_{re.sub(r'[^A-Za-z0-9_-]', '_', Path(name).stem)}.jpg"


def save_inspection(car_label: str, report: dict, annotated: dict, data_dir: Path = DATA_DIR) -> int:
    with _session() as s:
        insp = Inspection(
            car_label=car_label,
            overall_severity=report["overall_severity"],
            total_low=report["total_low"],
            total_high=report["total_high"],
            currency=report["currency"],
        )
        for it in report["items"]:
            insp.findings.append(
                FindingRow(
                    part=it["part"], damage=it["damage"], severity=it["severity"],
                    confidence=it["confidence"], area_pct=it["area_pct"], action=it["action"],
                    cost_low=it["cost_low"], cost_high=it["cost_high"], images="|".join(it["images"]),
                )
            )
        s.add(insp)
        s.flush()
        folder = Path(data_dir) / "annotated" / str(insp.id)
        folder.mkdir(parents=True, exist_ok=True)
        for i, data in enumerate(annotated.values()):
            (folder / _safe_name(i, list(annotated)[i])).write_bytes(data)
        s.commit()
        return insp.id


def _summary(i: Inspection) -> dict:
    return {
        "id": i.id,
        "car_label": i.car_label,
        "status": i.status,
        "created_at": i.created_at.strftime("%Y-%m-%d %H:%M"),
        "overall_severity": i.overall_severity,
        "total_low": i.total_low,
        "total_high": i.total_high,
        "currency": i.currency,
    }


def list_inspections() -> list[dict]:
    with _session() as s:
        rows = s.scalars(select(Inspection).order_by(Inspection.id.desc())).all()
        return [_summary(r) for r in rows]


def get_inspection(inspection_id: int, data_dir: Path = DATA_DIR) -> dict | None:
    with _session() as s:
        insp = s.get(Inspection, inspection_id)
        if insp is None:
            return None
        items = [
            {
                "part": f.part, "damage": f.damage, "severity": f.severity,
                "confidence": f.confidence, "area_pct": f.area_pct, "action": f.action,
                "cost_low": f.cost_low, "cost_high": f.cost_high,
                "images": f.images.split("|") if f.images else [],
            }
            for f in insp.findings
        ]
        folder = Path(data_dir) / "annotated" / str(insp.id)
        files = sorted(folder.glob("*.jpg")) if folder.exists() else []
        record = _summary(insp)
        record["report"] = {
            "items": items,
            "total_low": insp.total_low,
            "total_high": insp.total_high,
            "currency": insp.currency,
            "overall_severity": insp.overall_severity,
            "disclaimer": "Costs are rough estimates from a rule table, not a quote. A technician should confirm every finding.",
        }
        record["image_files"] = [str(p) for p in files]
        return record


def set_status(inspection_id: int, status: str) -> bool:
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    with _session() as s:
        insp = s.get(Inspection, inspection_id)
        if insp is None:
            return False
        insp.status = status
        s.commit()
        return True
