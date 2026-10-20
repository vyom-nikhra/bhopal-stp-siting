"""Phase 6d: road-network access for each site (can a vacuum truck get there?).

Euclidean distance to the nearest road says little about access: the nearest road may
be a dead-end lane far, by road, from anything a truck uses. Here each site gets:
- last_mile_m: straight-line distance from the site edge to the nearest drivable node;
- network_to_major_m: shortest drivable distance from that node to the nearest major
  road (motorway / trunk / primary / secondary, incl. links), respecting one-way
  streets. Computed with one multi-source Dijkstra on the reversed graph, so a single
  run gives every node's distance *to* the nearest major road.

The drivable network comes from OpenStreetMap through osmnx (one Overpass request,
cached), because the QuickOSM road lines have no junction topology to route on.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
from shapely.geometry.base import BaseGeometry

MAJOR = {"motorway", "trunk", "primary", "secondary"}


def is_major(highway) -> bool:
    """OSM `highway` value (a string or a list after simplification) is a major road."""
    values = highway if isinstance(highway, list) else [highway]
    return any(str(v).removesuffix("_link") in MAJOR for v in values)


def load_drive_graph(
    area_lonlat: BaseGeometry, graphml: Path, cache_dir: Path, crs
) -> nx.MultiDiGraph:
    """Drivable OSM graph for `area_lonlat` (EPSG:4326), projected to `crs`.

    Downloaded once and saved as GraphML; later runs load the file and make no request.
    """
    if graphml.exists():
        g = ox.load_graphml(graphml)
    else:
        from stp.net import prefer_reachable

        prefer_reachable("overpass-api.de")
        ox.settings.use_cache = True
        ox.settings.cache_folder = str(cache_dir)
        g = ox.graph_from_polygon(area_lonlat, network_type="drive")
        graphml.parent.mkdir(parents=True, exist_ok=True)
        ox.save_graphml(g, graphml)
    return ox.project_graph(g, to_crs=crs)


def major_nodes(g: nx.MultiDiGraph) -> set:
    """Nodes touching at least one major-road edge."""
    nodes = set()
    for u, v, data in g.edges(data=True):
        if is_major(data.get("highway")):
            nodes.update((u, v))
    if not nodes:
        raise ValueError("Graph has no major-road edges")
    return nodes


def distance_to_nodes(g: nx.MultiDiGraph, targets: set) -> dict:
    """Shortest drivable distance (edge `length`, metres) from every node TO any target."""
    return nx.multi_source_dijkstra_path_length(g.reverse(copy=False), targets, weight="length")


def site_access(sites: gpd.GeoDataFrame, g: nx.MultiDiGraph, to_major: dict) -> gpd.GeoDataFrame:
    """Add last_mile_m, access_node, network_to_major_m and reaches_major per site."""
    nodes = ox.graph_to_gdfs(g, edges=False)[["geometry"]]
    if nodes.crs != sites.crs:
        raise ValueError(f"CRS mismatch: graph {nodes.crs} vs sites {sites.crs}")
    nodes = nodes.rename_axis("node_id").reset_index()  # make the node id a plain column
    joined = gpd.sjoin_nearest(sites[["geometry"]], nodes, how="left", distance_col="last_mile_m")
    joined = joined[~joined.index.duplicated(keep="first")]  # ties: keep one node per site
    out = sites.copy()
    out["access_node"] = joined["node_id"].reindex(out.index).to_numpy()
    out["last_mile_m"] = joined["last_mile_m"].reindex(out.index).to_numpy()
    out["network_to_major_m"] = [to_major.get(n, np.inf) for n in out["access_node"]]
    out["reaches_major"] = np.isfinite(out["network_to_major_m"])
    return out
