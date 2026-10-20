"""Load config.yaml and describe the raster grids the pipeline works on."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from affine import Affine
from rasterio.crs import CRS

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out = Path(cfg["paths"]["output"])
    if not out.is_absolute():
        cfg["paths"]["output"] = str(REPO_ROOT / out)
    return cfg


@dataclass(frozen=True)
class Grid:
    """A north-up raster grid: origin is the top-left corner, shape is (rows, cols)."""

    origin: tuple[float, float]
    res: float
    shape: tuple[int, int]
    crs: CRS

    @property
    def transform(self) -> Affine:
        return Affine(self.res, 0.0, self.origin[0], 0.0, -self.res, self.origin[1])

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        rows, cols = self.shape
        left, top = self.origin
        return left, top - rows * self.res, left + cols * self.res, top

    @classmethod
    def from_config(cls, cfg: dict, name: str) -> Grid:
        g = cfg["grids"][name]
        return cls(
            tuple(g["origin"]), float(g["res"]), tuple(g["shape"]), CRS.from_user_input(cfg["crs"])
        )
