import cv2
import numpy as np

from .matching import match_damage_to_parts, merge_findings


def annotate(image, parts, damages, findings):
    """Draw damage masks in red, part outlines in blue, and a label per damage."""
    overlay = image.copy()
    for d in damages:
        overlay[d.mask] = (0, 0, 255)
    out = cv2.addWeighted(overlay, 0.45, image, 0.55, 0)
    for p in parts:
        contours, _ = cv2.findContours(p.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (255, 160, 0), 1)
    for d, f in zip(damages, findings):
        text = f"{f.damage} | {f.part} | {f.severity}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x = max(0, min(d.box[0], image.shape[1] - tw - 4))
        y = max(d.box[1], 16)
        cv2.rectangle(out, (x, y - th - 6), (x + tw + 4, y), (0, 0, 0), -1)
        cv2.putText(out, text, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def run_inspection(detector, images):
    """images: list of (name, BGR array). Returns (merged findings, {name: annotated jpg bytes})."""
    all_findings, annotated = [], {}
    for i, (name, img) in enumerate(images):
        if name in annotated:
            name = f"{i}_{name}"
        parts, damages = detector.predict(img)
        findings = match_damage_to_parts(parts, damages, img.shape[:2], name)
        all_findings.extend(findings)
        ok, buf = cv2.imencode(".jpg", annotate(img, parts, damages, findings))
        if ok:
            annotated[name] = buf.tobytes()
    return merge_findings(all_findings), annotated
