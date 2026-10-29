from dataclasses import dataclass, field

import numpy as np


@dataclass
class Region:
    """One detected object (a car part or a damage) with its mask."""

    label: str
    conf: float
    mask: np.ndarray  # bool array, shape (H, W)
    box: tuple  # x1, y1, x2, y2

    @property
    def area(self) -> int:
        return int(self.mask.sum())


@dataclass
class Finding:
    """Damage of one type on one part."""

    part: str
    damage: str
    confidence: float
    area_pct: float  # damaged area as a percentage of the part
    severity: str
    images: list = field(default_factory=list)
