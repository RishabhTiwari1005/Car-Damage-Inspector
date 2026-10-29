"""Central settings.

Class names used as keys below must match the labels in your trained models
(lowercase, underscores are converted to spaces by the detector).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
PARTS_WEIGHTS = MODELS_DIR / "parts.pt"
DAMAGE_WEIGHTS = MODELS_DIR / "damage.pt"
DB_URL = f"sqlite:///{DATA_DIR / 'inspections.db'}"

CONF_THRESHOLD = 0.25
IMGSZ = 640
CURRENCY = "USD"

SEVERITIES = ["minor", "moderate", "severe"]
STATUSES = ["pending", "in repair", "repaired"]

# Damage area as a share of the part is multiplied by this before thresholds apply.
DAMAGE_WEIGHT = {
    "scratch": 0.7,
    "dent": 1.5,
    "crack": 1.5,
    "glass shatter": 2.0,
    "lamp broken": 2.0,
    "tire flat": 2.0,
}
# Some damage is never "minor", however small it looks.
SEVERITY_FLOOR = {
    "glass shatter": "severe",
    "lamp broken": "moderate",
    "tire flat": "moderate",
}
# score < first -> minor, score < second -> moderate, otherwise severe
SEVERITY_THRESHOLDS = (0.03, 0.12)

# Substring match on the part name, first hit wins.
PART_COST_MULTIPLIER = {
    "windshield": 1.8,
    "headlight": 1.5,
    "taillight": 1.3,
    "light": 1.3,
    "lamp": 1.3,
    "roof": 1.6,
    "hood": 1.4,
    "door": 1.3,
    "trunk": 1.2,
    "wheel": 1.2,
    "fender": 1.1,
    "bumper": 1.0,
    "mirror": 0.8,
}


def _flat(action, low, high):
    return {s: (action, low, high) for s in SEVERITIES}


# damage -> severity -> (repair action, cost low, cost high). Illustrative numbers only.
REPAIR_ACTIONS = {
    "scratch": {
        "minor": ("Polish and touch-up paint", 40, 120),
        "moderate": ("Sand, fill and repaint area", 150, 400),
        "severe": ("Repaint full panel", 300, 800),
    },
    "dent": {
        "minor": ("Paintless dent repair", 80, 200),
        "moderate": ("Body filler and repaint", 200, 600),
        "severe": ("Replace panel", 500, 1500),
    },
    "crack": {
        "minor": ("Seal or patch crack", 60, 180),
        "moderate": ("Plastic weld and repaint", 200, 550),
        "severe": ("Replace part", 400, 1200),
    },
    "glass shatter": _flat("Replace glass", 250, 900),
    "lamp broken": _flat("Replace lamp assembly", 150, 700),
    "tire flat": _flat("Repair or replace tire", 60, 300),
}
DEFAULT_ACTIONS = {
    "minor": ("Inspect and touch up", 50, 150),
    "moderate": ("Repair and repaint", 200, 600),
    "severe": ("Replace part", 500, 1500),
}
