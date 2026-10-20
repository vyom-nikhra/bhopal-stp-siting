import geopandas as gpd
import networkx as nx
import numpy as np
import pytest
from shapely.geometry import box

from stp import network

CRS = "EPSG:32643"


def _graph():
    """Nodes on a line at x = 0, 100, 200, 300 (y = 0); node 3 sits on a primary road.

    1 <-> 2 is two-way residential; 2 -> 3 is one-way towards the primary road;
    node 4 (x = 1000) is isolated from the rest.
    """
    g = nx.MultiDiGraph(crs=CRS)
    for n, x in [(1, 0), (2, 100), (3, 200), (4, 1000)]:
        g.add_node(n, x=float(x), y=0.0)
    g.add_edge(1, 2, length=100.0, highway="residential")
    g.add_edge(2, 1, length=100.0, highway="residential")
    g.add_edge(2, 3, length=100.0, highway="residential")
    g.add_edge(3, 3, length=0.0, highway=["primary", "trunk_link"])  # simplified-list form
    return g


def test_is_major_handles_lists_and_links():
    assert network.is_major("primary")
    assert network.is_major("secondary_link")
    assert network.is_major(["residential", "trunk"])
    assert not network.is_major("residential")
    assert not network.is_major(None)


def test_major_nodes_and_directed_distance():
    g = _graph()
    majors = network.major_nodes(g)
    assert majors == {3}
    d = network.distance_to_nodes(g, majors)
    assert d[1] == pytest.approx(200.0) and d[2] == pytest.approx(100.0) and d[3] == 0
    assert 4 not in d  # isolated node cannot reach a major road


def test_one_way_is_respected():
    g = _graph()
    g.remove_edge(2, 3)
    g.add_edge(3, 2, length=100.0, highway="residential")  # now only away from the primary
    assert 1 not in network.distance_to_nodes(g, {3})


def test_site_access_last_mile_and_flags():
    g = _graph()
    to_major = network.distance_to_nodes(g, network.major_nodes(g))
    sites = gpd.GeoDataFrame(geometry=[box(-10, 20, 10, 40), box(990, 50, 1010, 70)], crs=CRS)
    out = network.site_access(sites, g, to_major)
    assert out["access_node"].tolist() == [1, 4]
    assert out["last_mile_m"].tolist() == [pytest.approx(20.0), pytest.approx(50.0)]
    assert out["network_to_major_m"].iloc[0] == pytest.approx(200.0)
    assert np.isinf(out["network_to_major_m"].iloc[1])
    assert out["reaches_major"].tolist() == [True, False]


def test_major_nodes_raises_without_major_roads():
    g = _graph()
    g.remove_edge(3, 3)
    with pytest.raises(ValueError, match="no major"):
        network.major_nodes(g)


def test_prefer_reachable_raises_when_nothing_answers(monkeypatch):
    import socket

    from stp import net

    monkeypatch.setattr(
        net,
        "_original_getaddrinfo",
        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.0.2.1", 443))],
    )

    def refuse(*a, **k):
        raise OSError("unreachable")

    monkeypatch.setattr(net.socket, "create_connection", refuse)
    with pytest.raises(ConnectionError, match="No reachable address"):
        net.prefer_reachable("example.invalid", timeout=0.1)
