import pytest
from shapely.geometry import box

from stp import crosscheck

OSM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="test">
  <node id="1" lat="23.250" lon="77.400" version="1"/>
  <node id="2" lat="23.250" lon="77.401" version="1"/>
  <node id="3" lat="23.251" lon="77.401" version="1"/>
  <node id="4" lat="23.251" lon="77.400" version="1"/>
  <node id="5" lat="23.252" lon="77.400" version="1"/>
  <node id="6" lat="23.252" lon="77.401" version="1"/>
  <node id="7" lat="23.253" lon="77.401" version="1"/>
  <node id="8" lat="23.253" lon="77.400" version="1"/>
  <node id="9" lat="23.300" lon="77.500" version="1"/>
  <node id="10" lat="23.300" lon="77.501" version="1"/>
  <node id="11" lat="23.301" lon="77.501" version="1"/>
  <node id="12" lat="23.301" lon="77.500" version="1"/>
  <way id="100" version="1">
    <nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/>
    <tag k="landuse" v="farmland"/>
  </way>
  <way id="101" version="1">
    <nd ref="5"/><nd ref="6"/><nd ref="7"/><nd ref="8"/><nd ref="5"/>
    <tag k="building" v="yes"/>
  </way>
  <way id="102" version="1">
    <nd ref="9"/><nd ref="10"/><nd ref="11"/><nd ref="12"/><nd ref="9"/>
    <tag k="landuse" v="industrial"/>
  </way>
</osm>
"""


def test_read_osm_landuse_keeps_landuse_areas_inside(tmp_path):
    path = tmp_path / "tiny.osm"
    path.write_text(OSM_XML, encoding="utf-8")
    area = box(77.39, 23.24, 77.45, 23.26)  # contains ways 100 and 101, not 102
    lu = crosscheck.read_osm_landuse(path, area)
    assert list(lu["landuse"]) == ["farmland"]
    assert lu.crs.to_epsg() == 4326
    assert lu.geometry.iloc[0].area > 0


def test_read_osm_landuse_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="download the extract"):
        crosscheck.read_osm_landuse(tmp_path / "absent.osm.pbf", box(0, 0, 1, 1))
