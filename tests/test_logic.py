"""Tests for matching, severity, report and storage. No trained weights needed."""
import numpy as np

from app import db
from app.pipeline import run_inspection
from app.report import build_report, export_pdf
from app.schemas import Region

H, W = 300, 400


def box_region(label, conf, x1, y1, x2, y2):
    mask = np.zeros((H, W), bool)
    mask[y1:y2, x1:x2] = True
    return Region(label, conf, mask, (x1, y1, x2, y2))


class FakeDetector:
    def __init__(self, extra_damage=None):
        self.extra_damage = extra_damage or []

    def predict(self, image):
        parts = [
            box_region("front bumper", 0.9, 0, 100, 200, 200),
            box_region("hood", 0.9, 200, 100, 400, 200),
        ]
        damages = [
            box_region("dent", 0.8, 20, 120, 100, 180),  # large dent on the bumper
            box_region("scratch", 0.7, 250, 120, 260, 130),  # tiny scratch on the hood
        ] + self.extra_damage
        return parts, damages


def blank():
    return np.zeros((H, W, 3), np.uint8)


def test_damage_is_matched_to_the_right_part_with_severity():
    findings, annotated = run_inspection(FakeDetector(), [("a.jpg", blank())])
    by_damage = {f.damage: f for f in findings}
    assert by_damage["dent"].part == "front bumper"
    assert by_damage["dent"].severity == "severe"
    assert by_damage["scratch"].part == "hood"
    assert by_damage["scratch"].severity == "minor"
    assert "a.jpg" in annotated


def test_damage_outside_any_part_is_flagged_unknown():
    stray = box_region("crack", 0.6, 5, 5, 30, 30)
    findings, _ = run_inspection(FakeDetector([stray]), [("a.jpg", blank())])
    assert any(f.part == "unknown area" and f.damage == "crack" for f in findings)


def test_multi_angle_findings_are_merged():
    findings, annotated = run_inspection(FakeDetector(), [("front.jpg", blank()), ("side.jpg", blank())])
    assert len(findings) == 2
    assert all(len(f.images) == 2 for f in findings)
    assert len(annotated) == 2


def test_report_totals_and_pdf():
    findings, annotated = run_inspection(FakeDetector(), [("a.jpg", blank())])
    report = build_report(findings)
    assert report["overall_severity"] == "severe"
    assert 0 < report["total_low"] <= report["total_high"]
    assert export_pdf("TEST-1", report, annotated).startswith(b"%PDF")


def test_empty_report():
    report = build_report([])
    assert report["overall_severity"] == "none"
    assert report["total_high"] == 0
    assert export_pdf("EMPTY", report, {}).startswith(b"%PDF")


def test_database_roundtrip(tmp_path):
    db.init_db(f"sqlite:///{tmp_path / 'test.db'}")
    findings, annotated = run_inspection(FakeDetector(), [("a.jpg", blank())])
    report = build_report(findings)
    new_id = db.save_inspection("CAR-7", report, annotated, data_dir=tmp_path)

    record = db.get_inspection(new_id, data_dir=tmp_path)
    assert record["car_label"] == "CAR-7"
    assert record["status"] == "pending"
    assert len(record["report"]["items"]) == len(report["items"])
    assert len(record["image_files"]) == 1

    assert db.set_status(new_id, "repaired")
    assert db.list_inspections()[0]["status"] == "repaired"
