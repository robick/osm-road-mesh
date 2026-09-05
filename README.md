# osm-road-mesh

Turn any city's road network into a watertight, 3D-printable STL — roads that
follow the terrain underneath them instead of floating flat above it.

Give it a latitude/longitude and a radius. It fetches the road network from
OpenStreetMap, scales it to a physical print plate in millimetres, raises each
road class to its own height, drapes the whole thing over a terrain surface,
and writes a manifold mesh your slicer will accept without repair.

## Why it's built this way

**The obvious approach — a heightfield grid — produces meshes that look right
and print wrong.** Sampling roads onto a regular grid and extruding gives you
twisted strips where roads bend, spikes at junctions where several segments
meet, side walls that don't close, and an O(n²) loop that leaves holes when
index arithmetic drifts. A slicer either refuses the file or silently repairs
it into something you didn't design.

This replaces the grid entirely:

- **Real polygon triangulation.** Road centrelines are buffered to width,
  unioned into a single polygon, and triangulated with `shapely.ops.triangulate`.
  Topology comes from the geometry, not from a sampling raster.
- **Per-vertex terrain height, not per-polygon.** Every triangle vertex is
  sampled against the terrain independently, so a road crossing a slope tilts
  with it. Sampling once per polygon centroid — the cheaper thing to do — is
  what causes the visible stepping in naive implementations.
- **Side walls built by ring-walking.** Walls are constructed by walking each
  boundary ring and stitching top to bottom in order, so they are watertight by
  construction rather than watertight if you're lucky.
- **A manifold repair pass at the end.** `trimesh.repair` fixes normals and
  winding, and the exporter validates before writing rather than after.

**Terrain sampling has two modes** because the terrain source differs by
project: `from_stl` builds a KD-tree over an existing base mesh's vertices and
does nearest-neighbour lookups; `from_array` interpolates a regular grid with
`scipy.interpolate.RegularGridInterpolator`. Both present the same interface to
the mesh builder.

**Two STLs come out.** A full-resolution mesh, and a decimated one via
`fast_simplification` (85% reduction by default) for preview on a phone, where
a multi-million-triangle city will not load.

## How it works

| Stage | Class | What happens |
|---|---|---|
| 1 | `RoadNetworkLoader` | Fetch from OSM by point + radius, dedupe, buffer per road class, union, project to millimetre plate space |
| 2 | `TerrainSampler` | Resolve Z for any (x, y) — KD-tree over a base STL, or grid interpolation |
| 3 | `RoadMeshBuilder` | Triangulate the road polygon, lift each vertex to terrain + class height, ring-walk the side walls |
| 4 | — | `trimesh.repair` manifold pass |
| 5 | `STLExporter` | Write full and decimated STLs, validate, report |

`RoadConfig` holds every tunable in one dataclass — no magic numbers scattered
through the code. Road classes (`motorway`, `trunk`, `primary`, `secondary`)
each carry their own width and raise height, which is what makes the printed
result readable as a road hierarchy rather than a uniform web.

## Use it

```bash
pip install -r requirements.txt
python road_mesh.py
```

Defaults render Chennai at an 18 km radius onto a 120 × 160 mm plate. Change
`RoadConfig.point` and `RoadConfig.dist` for anywhere else.

As a library:

```python
from road_mesh import build_road_mesh, RoadConfig, TerrainSampler

sampler = TerrainSampler(mode="from_array", xs_mm=..., ys_mm=..., zz=...)
mesh = build_road_mesh(RoadConfig(point=(51.5074, -0.1278), dist=12000), sampler)
```

`build_road_mesh` returns a `trimesh.Trimesh`, so anything trimesh can do with
a mesh works from there.

## What I'd do differently

Nearest-neighbour KD-tree sampling is fast but blunt — barycentric
interpolation across the base mesh's triangles would give smoother draping on
steep terrain. And road junctions are currently resolved by polygon union,
which is correct but loses the over/under information OSM actually carries in
its `layer` and `bridge` tags; honouring those would let flyovers print as
flyovers.

## Licence

Apache 2.0. See [LICENSE](LICENSE).
