import numpy as np

from .config import SEVERITIES, SEVERITY_FLOOR, SEVERITY_THRESHOLDS, DAMAGE_WEIGHT
from .schemas import Finding

UNKNOWN_PART = "unknown area"


def rank(severity: str) -> int:
    return SEVERITIES.index(severity)


def classify_severity(damage: str, ratio: float) -> str:
    """ratio = damaged area / part area (0 to 1)."""
    score = ratio * DAMAGE_WEIGHT.get(damage, 1.0)
    low, high = SEVERITY_THRESHOLDS
    severity = "minor" if score < low else "moderate" if score < high else "severe"
    floor = SEVERITY_FLOOR.get(damage)
    if floor and rank(severity) < rank(floor):
        severity = floor
    return severity


def match_damage_to_parts(parts, damages, image_shape, image_name=""):
    """Assign each damage region to the part it overlaps most.

    Returns one Finding per damage, in the same order as `damages`.
    """
    h, w = image_shape
    findings = []
    for d in damages:
        best, best_inter = None, 0
        for p in parts:
            inter = int(np.logical_and(d.mask, p.mask).sum())
            if inter > best_inter:
                best, best_inter = p, inter
        if best is None:
            # No overlapping part: assume a typical part fills about 10% of the frame.
            part_name, part_area, damaged = UNKNOWN_PART, 0.1 * h * w, d.area
        else:
            part_name, part_area, damaged = best.label, best.area, best_inter
        ratio = min(1.0, damaged / max(part_area, 1))
        findings.append(
            Finding(
                part=part_name,
                damage=d.label,
                confidence=round(d.conf, 3),
                area_pct=round(ratio * 100, 1),
                severity=classify_severity(d.label, ratio),
                images=[image_name],
            )
        )
    return findings


def merge_findings(findings):
    """Combine findings from several photos into one entry per (part, damage type).

    Keeps the worst severity seen. Two separate dents on the same part count as one entry.
    """
    merged = {}
    for f in findings:
        key = (f.part, f.damage)
        cur = merged.get(key)
        if cur is None:
            merged[key] = Finding(**{**f.__dict__, "images": list(f.images)})
            continue
        for img in f.images:
            if img not in cur.images:
                cur.images.append(img)
        if (rank(f.severity), f.confidence) > (rank(cur.severity), cur.confidence):
            cur.severity, cur.confidence, cur.area_pct = f.severity, f.confidence, f.area_pct
    return sorted(merged.values(), key=lambda f: (-rank(f.severity), f.part))
