"""Phase 0 check: core geospatial imports work and the right copies are loaded."""

import sys

import geopandas as gpd
import numpy
import osmnx
import pyogrio
import pyproj
import rasterio
import rasterstats
import rioxarray
import scipy
import shapely
import sklearn
import xarray
from osgeo import gdal

import stp
from stp.config import load_config

print("python      ", sys.version.split()[0], sys.executable)
print("stp from    ", stp.__file__)
for m in [numpy, rasterio, rioxarray, xarray, gpd, pyogrio, shapely, pyproj, scipy, sklearn,
          rasterstats, osmnx]:
    print(f"{m.__name__:12s} {m.__version__:10s} {m.__file__}")
print(f"{'gdal':12s} {gdal.__version__:10s} rasterio built on GDAL {rasterio.__gdal_version__}")

user_site = [p for p in sys.path if "AppData" in p and "Roaming" in p]
if user_site:
    raise SystemExit(f"FAIL: user-site directory on sys.path: {user_site}")

cfg = load_config()
g = gpd.read_file(cfg["paths"]["boundary"], engine="pyogrio")
print("read_file OK:", len(g), "feature(s),", g.crs)
print("ENV OK")
