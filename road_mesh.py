"""
road_mesh.py — Terrain-adaptive, watertight road meshes from OpenStreetMap.

Builds a 3D-printable road network for any location, solving the geometry
problems a naive grid-mesh approach runs into:
  - Proper polygon triangulation via shapely.ops.triangulate (no grid mesh)
  - Per-vertex terrain-adaptive Z (no centroid sampling artifacts)
  - Watertight side walls by construction
  - Manifold guarantee via trimesh repair pass

Usage (standalone):
    python road_mesh.py

Usage (importable from map_to_stl.py):
    from road_mesh import build_road_mesh, RoadConfig, TerrainSampler
    sampler = TerrainSampler(mode='from_array', xs_mm=..., ys_mm=..., zz=...)
    config  = RoadConfig()
    mesh    = build_road_mesh(config, sampler)
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
import trimesh.repair
import osmnx as ox
import fast_simplification
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import KDTree, cKDTree
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union, triangulate as shapely_triangulate
from shapely import affinity
from tqdm import tqdm

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("road_mesh")


# ── Configuration dataclass ────────────────────────────────────────────────────
@dataclass
class RoadConfig:
    """All tunable parameters for road mesh generation.

    Attributes:
        point: (lat, lon) center of the map area.
        dist: Radius in metres to fetch from OSM.
        plate_w: Physical width of the print plate in mm.
        plate_h: Physical height of the print plate in mm.
        road_raise: Height above terrain surface per road class (mm).
        road_width: Half-buffer road width per class (mm).
        buffer_resolution: Shapely buffer resolution (arc segments per quarter turn).
        simplify_tolerance: Shapely simplify tolerance (mm) applied after union.
        base_min_z: Fallback Z when terrain sampling fails (mm).
        include: Road classes to process.
        output_full: Output path for full-resolution STL.
        output_mobile: Output path for mobile/simplified STL.
        mobile_target_reduction: fast_simplification target_reduction factor.
        base_stl: Path to terrain base STL (used by TerrainSampler in from_stl mode).
        use_cache: Whether to use osmnx HTTP cache.
    """

    point: Tuple[float, float] = (13.0836939, 80.270186)
    dist: int = 18000
    plate_w: float = 120.0
    plate_h: float = 160.0
    road_raise: Dict[str, float] = field(default_factory=lambda: {
        "motorway":  0.72,
        "trunk":     0.60,
        "primary":   0.384,  # 20% reduction from 0.48
        "secondary": 0.36,
    })
    road_width: Dict[str, float] = field(default_factory=lambda: {
        "motorway":  0.8,
        "trunk":     0.6,
        "primary":   0.5,
        "secondary": 0.4,
    })
    buffer_resolution: int = 128
    simplify_tolerance: float = 0.003
    base_min_z: float = 2.0
    include: frozenset = field(default_factory=lambda: frozenset({
        "motorway", "trunk", "primary", "secondary"
    }))
    output_full: str = "output/roads.stl"
    output_mobile: str = "output/roads_mobile.stl"
    mobile_target_reduction: float = 0.85
    base_stl: str = "output/base.stl"
    use_cache: bool = True
    variant: str = "current"


# ── Terrain Sampler ────────────────────────────────────────────────────────────
class TerrainSampler:
    """Wraps terrain height lookup with proper bounds clamping.

    Supports two construction modes:
      - ``from_stl``: builds a KDTree from a base terrain STL's vertices
      - ``from_array``: builds a RegularGridInterpolator from provided grids

    Args:
        mode: Either ``'from_stl'`` or ``'from_array'``.
        stl_path: Required when mode='from_stl'. Path to base terrain STL.
        xs_mm: 1-D sorted array of X coordinates (mm). Required for 'from_array'.
        ys_mm: 1-D sorted array of Y coordinates (mm). Required for 'from_array'.
        zz: 2-D array of shape (len(xs_mm), len(ys_mm)) heights (mm).
            Required for 'from_array'.
        fallback_z: Z value returned when sampling fails.
    """

    def __init__(
        self,
        mode: str = "from_stl",
        stl_path: Optional[str] = None,
        xs_mm: Optional[np.ndarray] = None,
        ys_mm: Optional[np.ndarray] = None,
        zz: Optional[np.ndarray] = None,
        fallback_z: float = 2.0,
    ) -> None:
        self._fallback_z = fallback_z
        self._mode = mode

        if mode == "from_stl":
            if stl_path is None:
                raise ValueError("stl_path is required for mode='from_stl'")
            self._init_from_stl(stl_path)
        elif mode == "from_array":
            if xs_mm is None or ys_mm is None or zz is None:
                raise ValueError("xs_mm, ys_mm, zz are required for mode='from_array'")
            self._init_from_array(xs_mm, ys_mm, zz)
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use 'from_stl' or 'from_array'.")

    # ── private init helpers ──────────────────────────────────────────────────

    def _init_from_stl(self, stl_path: str) -> None:
        """Load terrain from STL file and build KDTree for nearest-vertex lookup.

        Args:
            stl_path: Path to the base terrain STL file.
        """
        log.info("TerrainSampler: loading %s", stl_path)
        mesh = trimesh.load(stl_path)
        verts = np.asarray(mesh.vertices)

        # Keep only top-surface vertices (above 10th percentile Z)
        z_thresh = np.percentile(verts[:, 2], 10)
        top = verts[verts[:, 2] > z_thresh]

        self._kdtree: cKDTree = cKDTree(top[:, :2])
        self._top_z: np.ndarray = top[:, 2]
        self._x_bounds = (top[:, 0].min(), top[:, 0].max())
        self._y_bounds = (top[:, 1].min(), top[:, 1].max())

        log.info(
            "TerrainSampler (from_stl): %d top vertices, "
            "X=[%.1f,%.1f] Y=[%.1f,%.1f] Z=[%.2f,%.2f]",
            len(top),
            *self._x_bounds,
            *self._y_bounds,
            self._top_z.min(),
            self._top_z.max(),
        )
        self._interp = None  # not used in from_stl mode

    def _init_from_array(
        self,
        xs_mm: np.ndarray,
        ys_mm: np.ndarray,
        zz: np.ndarray,
    ) -> None:
        """Build a RegularGridInterpolator from provided grid arrays.

        Args:
            xs_mm: 1-D sorted array of X grid points (mm).
            ys_mm: 1-D sorted array of Y grid points (mm).
            zz: 2-D height array with shape ``(len(xs_mm), len(ys_mm))``.
        """
        xs_mm = np.asarray(xs_mm, dtype=np.float64)
        ys_mm = np.asarray(ys_mm, dtype=np.float64)
        zz = np.asarray(zz, dtype=np.float64)

        self._x_bounds = (xs_mm[0], xs_mm[-1])
        self._y_bounds = (ys_mm[0], ys_mm[-1])
        self._interp = RegularGridInterpolator(
            (xs_mm, ys_mm),
            zz,
            method="linear",
            bounds_error=False,
            fill_value=None,  # extrapolate nearest
        )
        self._kdtree = None
        self._top_z = None
        log.info(
            "TerrainSampler (from_array): grid %dx%d, X=[%.1f,%.1f] Y=[%.1f,%.1f]",
            len(xs_mm),
            len(ys_mm),
            *self._x_bounds,
            *self._y_bounds,
        )

    # ── public API ────────────────────────────────────────────────────────────

    def sample(self, x: float, y: float) -> float:
        """Return terrain Z (mm) at the given (x, y) coordinate.

        Clamps query to valid bounds before sampling. Returns fallback_z on error.

        Args:
            x: X coordinate (mm).
            y: Y coordinate (mm).

        Returns:
            Terrain height in mm.
        """
        try:
            xc = float(np.clip(x, *self._x_bounds))
            yc = float(np.clip(y, *self._y_bounds))
            if self._mode == "from_stl":
                _, idx = self._kdtree.query([xc, yc])
                z = float(self._top_z[idx])
            else:
                z_arr = self._interp([[xc, yc]])
                if z_arr is None or np.any(np.isnan(z_arr)):
                    return self._fallback_z
                z = float(z_arr[0])
            return max(z, self._fallback_z)
        except Exception:
            return self._fallback_z

    def sample_batch(self, xy: np.ndarray) -> np.ndarray:
        """Batch-sample terrain Z for N points.

        Args:
            xy: Array of shape (N, 2) with (x, y) coordinates in mm.

        Returns:
            Array of shape (N,) with terrain heights in mm.
        """
        xy = np.asarray(xy, dtype=np.float64)
        xc = np.clip(xy[:, 0], *self._x_bounds)
        yc = np.clip(xy[:, 1], *self._y_bounds)
        clamped = np.stack([xc, yc], axis=1)

        if self._mode == "from_stl":
            _, idxs = self._kdtree.query(clamped)
            z = self._top_z[idxs].copy()
        else:
            z = self._interp(clamped)
            z = np.where(np.isnan(z), self._fallback_z, z)

        return np.maximum(z, self._fallback_z)


# ── Road Network Loader ────────────────────────────────────────────────────────
class RoadNetworkLoader:
    """Fetches and preprocesses OSM road network data.

    Args:
        config: RoadConfig instance with project parameters.
    """

    def __init__(self, config: RoadConfig) -> None:
        self._cfg = config

    def load(self) -> Tuple[np.ndarray, float, float, float, float]:
        """Fetch road network and return buffered polygons grouped by road type.

        Returns:
            A tuple of:
              - type_polys: dict mapping road-type str → list of Shapely Polygons (in mm)
              - minx, miny, maxx, maxy: projected bounding box (metres) for scale reference
        """
        cfg = self._cfg
        ox.settings.use_cache = cfg.use_cache

        log.info("Fetching road network from OSM (dist=%dm)…", cfg.dist)
        G = ox.graph_from_point(
            cfg.point,
            dist=cfg.dist,
            network_type="drive",
            retain_all=False,
            simplify=False,
        )
        G_proj = ox.project_graph(G)
        nodes, edges = ox.graph_to_gdfs(G_proj)

        # Deduplicate reversed edges (u,v) == (v,u)
        seen: set = set()
        keep: List = []
        for idx in edges.index:
            u, v = idx[0], idx[1]
            key = (min(u, v), max(u, v))
            if key not in seen:
                seen.add(key)
                keep.append(idx)
        edges = edges.loc[keep]

        minx, miny, maxx, maxy = nodes.total_bounds
        scale = min(cfg.plate_w / (maxx - minx), cfg.plate_h / (maxy - miny))
        off_x = (cfg.plate_w - (maxx - minx) * scale) / 2
        off_y = (cfg.plate_h - (maxy - miny) * scale) / 2
        log.info(
            "%d unique edges | scale=%.6f | offset=(%.2f, %.2f)",
            len(edges),
            scale,
            off_x,
            off_y,
        )

        # Buffer per road type
        type_bufs: Dict[str, List] = {k: [] for k in cfg.include}
        skipped = 0

        for _, row in tqdm(edges.iterrows(), total=len(edges), desc="Buffering roads", unit="edge"):
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            rtype = self._road_type(row.get("highway", ""))
            if rtype is None:
                skipped += 1
                continue
            rw = (cfg.road_width[rtype] / scale) / 2
            try:
                simplify_amt = rw * (0.45 if cfg.variant == 'experimental' else 0.3)
                buf_res = 32 if cfg.variant == 'experimental' else cfg.buffer_resolution
                g = geom.simplify(simplify_amt, preserve_topology=True)
                type_bufs[rtype].append(
                    g.buffer(rw, resolution=buf_res, cap_style=1, join_style=1)
                )
            except Exception as exc:
                log.debug("Buffer failed for edge: %s", exc)

        log.info("Skipped %d non-primary road edges", skipped)

        # Plate boundary for clipping (with small inset to avoid edge artifacts)
        plate_box = shapely_box(0.1, 0.1, cfg.plate_w - 0.1, cfg.plate_h - 0.1)

        # Union and convert to mm-space polygons
        type_polys: Dict[str, List[Polygon]] = {}
        for rtype, bufs in type_bufs.items():
            if not bufs:
                continue
            union = unary_union(bufs)
            union = _to_mm(union, scale, minx, miny, off_x, off_y)
            # Clip to plate bounds — roads outside get trimmed, not dropped
            union = union.intersection(plate_box)
            union = union.simplify(cfg.simplify_tolerance, preserve_topology=True)
            if cfg.variant == 'experimental':
                # Close tiny cracks / pinholes at polygon boundaries.
                union = union.buffer(0.01).buffer(-0.01)
            union = union.buffer(0)
            polys = list(union.geoms) if isinstance(union, MultiPolygon) else [union]
            valid = [p for p in polys if not p.is_empty and p.is_valid and p.area >= 0.02]
            type_polys[rtype] = valid
            log.info("  %-10s: %d valid polygons after union+clip", rtype, len(valid))

        return type_polys, minx, miny, maxx, maxy

    def _road_type(self, highway) -> Optional[str]:
        """Map an OSM highway tag to one of the configured road classes.

        Args:
            highway: OSM highway tag value (str or list).

        Returns:
            Road class string or None if not in include set.
        """
        hw = highway[0] if isinstance(highway, list) else str(highway)
        for k in self._cfg.include:
            if k in hw:
                return k
        return None


# ── Road Mesh Builder ──────────────────────────────────────────────────────────
class RoadMeshBuilder:
    """Converts road Shapely polygons into watertight trimesh.Trimesh objects.

    Uses Delaunay triangulation of the top surface with per-vertex terrain Z,
    and correctly stitched side walls for a fully manifold result.

    Args:
        config: RoadConfig with project parameters.
        terrain: TerrainSampler instance for Z lookups.
    """

    def __init__(self, config: RoadConfig, terrain: TerrainSampler) -> None:
        self._cfg = config
        self._terrain = terrain

    def build_all(self, type_polys: Dict[str, List[Polygon]]) -> trimesh.Trimesh:
        """Build road meshes for all road types and concatenate.

        Args:
            type_polys: dict mapping road class → list of Shapely Polygons (mm space).

        Returns:
            Combined, repaired trimesh.Trimesh of all roads.
        """
        all_meshes: List[trimesh.Trimesh] = []
        total_processed = total_skipped = total_failed = 0

        if self._cfg.variant == 'experimental':
            merged: List[Polygon] = []
            for polys in type_polys.values():
                merged.extend([p for p in polys if p is not None and not p.is_empty and p.is_valid])
            if merged:
                from shapely.ops import unary_union as _uu
                merged_union = _uu(merged).buffer(0)
                merged_polys = list(merged_union.geoms) if isinstance(merged_union, MultiPolygon) else [merged_union]
                type_polys = {'merged': [p for p in merged_polys if not p.is_empty and p.is_valid and p.area >= 0.02]}
                log.info('Experimental mode: merged %d road polygons into %d', len(merged), len(type_polys['merged']))

        for rtype, polys in type_polys.items():
            raise_mm = max(self._cfg.road_raise.values()) if self._cfg.variant == 'experimental' else self._cfg.road_raise[rtype]
            processed = skipped = failed = 0

            log.info("Building mesh for %s (%d polygons, raise=%.2fmm)…", rtype, len(polys), raise_mm)

            for idx, poly in enumerate(tqdm(polys, desc=f"  {rtype}", unit="poly")):
                if poly.is_empty or not poly.is_valid:
                    skipped += 1
                    continue
                try:
                    mesh = self._build_polygon_mesh(poly, raise_mm)
                    if mesh is not None and len(mesh.faces) > 0:
                        all_meshes.append(mesh)
                        processed += 1
                    else:
                        skipped += 1
                except Exception as exc:
                    log.debug(
                        "  [%s] poly #%d (area=%.4f) failed: %s",
                        rtype,
                        idx,
                        poly.area,
                        exc,
                    )
                    failed += 1

            log.info(
                "  %-10s: processed=%d  skipped=%d  failed=%d",
                rtype,
                processed,
                skipped,
                failed,
            )
            total_processed += processed
            total_skipped += skipped
            total_failed += failed

        log.info(
            "Total: processed=%d  skipped=%d  failed=%d",
            total_processed,
            total_skipped,
            total_failed,
        )

        if not all_meshes:
            raise RuntimeError("No road meshes were produced — check OSM data and terrain bounds.")

        combined = trimesh.util.concatenate(all_meshes)
        _repair(combined)
        return combined

    # ── polygon → mesh ────────────────────────────────────────────────────────

    def _build_polygon_mesh(self, poly: Polygon, raise_mm: float) -> Optional[trimesh.Trimesh]:
        """Build a watertight 3D mesh for a single road polygon.

        Algorithm:
          1. Triangulate the 2D polygon top surface with shapely.ops.triangulate.
          2. Assign per-vertex terrain Z + raise.
          3. Create flat bottom cap at Z=0.
          4. Stitch side walls between top and bottom rings.
          5. Combine and repair.

        Args:
            poly: A valid Shapely Polygon in mm-space.
            raise_mm: Height offset above terrain for this road class (mm).

        Returns:
            A trimesh.Trimesh or None if triangulation produced no faces.
        """
        # ── Step 1: triangulate top surface in 2D ────────────────────────────
        triangles = self._triangulate_polygon(poly)
        if not triangles:
            return None

        # Collect unique 2D vertices from triangulation
        top_xy_list: List[Tuple[float, float]] = []
        top_faces_raw: List[Tuple[int, int, int]] = []

        xy_index: Dict[Tuple[float, float], int] = {}

        def _get_or_add(x: float, y: float) -> int:
            key = (round(x, 6), round(y, 6))
            if key not in xy_index:
                xy_index[key] = len(top_xy_list)
                top_xy_list.append(key)
            return xy_index[key]

        for tri in triangles:
            coords = list(tri.exterior.coords)[:3]
            ia = _get_or_add(coords[0][0], coords[0][1])
            ib = _get_or_add(coords[1][0], coords[1][1])
            ic = _get_or_add(coords[2][0], coords[2][1])
            if ia == ib or ib == ic or ia == ic:
                continue
            top_faces_raw.append((ia, ib, ic))

        if not top_faces_raw:
            return None

        top_xy = np.array(top_xy_list, dtype=np.float64)  # (N, 2)
        n_top = len(top_xy)

        # ── Step 2: per-vertex terrain Z ─────────────────────────────────────
        top_z = self._terrain.sample_batch(top_xy) + raise_mm  # (N,)

        top_verts = np.column_stack([top_xy, top_z])           # (N, 3)
        bot_verts = np.column_stack([top_xy, np.zeros(n_top)]) # (N, 3) flat bottom

        top_faces = np.array(top_faces_raw, dtype=np.int64)
        # Bottom cap: reversed winding for inward normals
        bot_faces = top_faces[:, ::-1] + n_top

        # ── Step 3: side walls ────────────────────────────────────────────────
        side_verts, side_faces = self._build_side_walls(poly, raise_mm, n_top * 2)

        # ── Step 4: assemble full mesh ────────────────────────────────────────
        all_verts = np.vstack([top_verts, bot_verts, side_verts])
        all_faces = np.vstack([top_faces, bot_faces, side_faces])

        mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=False)
        _repair(mesh)
        return mesh

    def _triangulate_polygon(self, poly: Polygon) -> List[Polygon]:
        """Triangulate a Shapely Polygon using shapely.ops.triangulate.

        For polygons with interior holes, filters out triangles that fall
        inside the holes.

        Args:
            poly: Shapely Polygon (may have interiors).

        Returns:
            List of triangle Polygons covering the polygon's filled area.
        """
        triangles = shapely_triangulate(poly, edges=False)

        def _good_triangle(t: Polygon) -> bool:
            if not t.centroid.within(poly):
                return False
            coords = list(t.exterior.coords)[:3]
            a = np.array(coords[0])
            b = np.array(coords[1])
            c = np.array(coords[2])
            ab = np.linalg.norm(a - b)
            bc = np.linalg.norm(b - c)
            ca = np.linalg.norm(c - a)
            longest = max(ab, bc, ca)
            area = t.area
            # Experimental path removes very skinny triangles which cause long slivers.
            if self._cfg.variant == 'experimental':
                return area >= 1e-4 and (area / (longest * longest + 1e-9)) >= 0.015
            return True

        return [t for t in triangles if _good_triangle(t)]

    def _build_side_walls(
        self,
        poly: Polygon,
        raise_mm: float,
        base_vert_offset: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build side wall quad-strips for all rings of a polygon.

        For each consecutive edge (p1→p2) in each ring:
          - top1 at (p1.x, p1.y, terrain_z + raise)
          - top2 at (p2.x, p2.y, terrain_z + raise)
          - bot1 at (p1.x, p1.y, 0.0)
          - bot2 at (p2.x, p2.y, 0.0)
        Each quad is two CCW triangles viewed from outside.

        Args:
            poly: The road polygon in mm-space.
            raise_mm: Raise offset for this road class (mm).
            base_vert_offset: Vertex index offset (= n_top_verts + n_bot_verts).

        Returns:
            Tuple of (vertices array (M,3), faces array (K,3)).
        """
        verts: List[np.ndarray] = []
        faces: List[Tuple[int, int, int]] = []

        # Walk exterior ring + all hole rings
        rings = [poly.exterior] + list(poly.interiors)

        vert_idx = base_vert_offset

        for ring_idx, ring in enumerate(rings):
            coords = list(ring.coords)
            if coords[0] == coords[-1]:
                coords = coords[:-1]
            n = len(coords)
            if n < 2:
                continue

            for i in range(n):
                p1 = coords[i]
                p2 = coords[(i + 1) % n]

                x1, y1 = p1[0], p1[1]
                x2, y2 = p2[0], p2[1]

                z_top1 = self._terrain.sample(x1, y1) + raise_mm
                z_top2 = self._terrain.sample(x2, y2) + raise_mm

                top1 = np.array([x1, y1, z_top1])
                top2 = np.array([x2, y2, z_top2])
                bot1 = np.array([x1, y1, 0.0])
                bot2 = np.array([x2, y2, 0.0])

                # Quad indices: top1=0, top2=1, bot2=2, bot1=3 (local)
                i0 = vert_idx
                i1 = vert_idx + 1
                i2 = vert_idx + 2
                i3 = vert_idx + 3
                vert_idx += 4

                verts.extend([top1, top2, bot2, bot1])

                # Exterior ring: outward = CCW looking from outside
                # For exterior ring, winding is already CCW in Shapely
                # For hole rings, Shapely returns CW → winding still correct
                # Triangle 1: top1, bot1, bot2 (outward face)
                # Triangle 2: top1, bot2, top2
                if ring_idx == 0:
                    # Exterior: outward normal faces away from interior
                    faces.append((i0, i3, i2))
                    faces.append((i0, i2, i1))
                else:
                    # Hole: inward normal (flip)
                    faces.append((i0, i2, i3))
                    faces.append((i0, i1, i2))

        if not verts:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64)

        return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int64)


# ── STL Exporter ──────────────────────────────────────────────────────────────
class STLExporter:
    """Handles export, validation, and mobile simplification of road meshes.

    Args:
        config: RoadConfig with output paths and parameters.
    """

    def __init__(self, config: RoadConfig) -> None:
        self._cfg = config

    def export(self, mesh: trimesh.Trimesh) -> None:
        """Export full and mobile STL files and print a validation report.

        Args:
            mesh: Combined road trimesh.Trimesh to export.
        """
        os.makedirs(os.path.dirname(self._cfg.output_full), exist_ok=True)

        # ── Full resolution export ────────────────────────────────────────────
        mesh.export(self._cfg.output_full)
        full_size_kb = os.path.getsize(self._cfg.output_full) // 1024
        log.info("Full STL: %s (%d KB, %d faces)", self._cfg.output_full, full_size_kb, len(mesh.faces))

        # ── Mobile export ─────────────────────────────────────────────────────
        pts, mobile_faces = fast_simplification.simplify(
            mesh.vertices.astype("float32"),
            mesh.faces,
            target_reduction=self._cfg.mobile_target_reduction,
        )
        mobile = trimesh.Trimesh(vertices=pts, faces=mobile_faces, process=False)
        _repair(mobile)
        mobile.export(self._cfg.output_mobile)
        mob_size_kb = os.path.getsize(self._cfg.output_mobile) // 1024
        log.info(
            "Mobile STL: %s (%d KB, %d faces)",
            self._cfg.output_mobile,
            mob_size_kb,
            len(mobile.faces),
        )

        # ── Validation report ─────────────────────────────────────────────────
        self._print_validation(mesh, label="Full")
        self._print_validation(mobile, label="Mobile")

    def _print_validation(self, mesh: trimesh.Trimesh, label: str = "") -> None:
        """Print geometry validation metrics to the log.

        Args:
            mesh: Mesh to validate.
            label: Human-readable label for the report.
        """
        z_min = float(mesh.vertices[:, 2].min())
        z_max = float(mesh.vertices[:, 2].max())
        n_faces = len(mesh.faces)
        watertight = mesh.is_watertight
        is_vol = mesh.is_volume

        report_lines = [
            f"─── Validation [{label}] ───────────────────────",
            f"  Faces      : {n_faces:,}",
            f"  Z range    : [{z_min:.3f}, {z_max:.3f}] mm",
            f"  Watertight : {'✓ YES' if watertight else '✗ NO'}",
            f"  Is volume  : {'✓ YES' if is_vol else '✗ NO'}",
            f"  Face count : {'✓ OK (>1000)' if n_faces > 1000 else '✗ LOW (<1000)'}",
        ]

        # Z range sanity check (expected: > 0, < 20 for Chennai scale)
        z_ok = z_min >= 0.0 and z_max < 30.0
        report_lines.append(f"  Z range ok : {'✓ YES' if z_ok else '✗ OUT OF EXPECTED BOUNDS'}")

        print("\n".join(report_lines))
        if not watertight:
            log.warning("[%s] Mesh is NOT watertight — may not 3D print cleanly.", label)
        if not is_vol:
            log.warning("[%s] Mesh is NOT a valid volume.", label)
        if n_faces <= 1000:
            log.warning("[%s] Very low face count (%d) — possible empty mesh.", label, n_faces)


# ── Public API ─────────────────────────────────────────────────────────────────
def build_road_mesh(
    config: RoadConfig,
    terrain_sampler: TerrainSampler,
) -> trimesh.Trimesh:
    """Build the combined road mesh without exporting.

    Intended for use inside map_to_stl.py or other callers that need
    the mesh object for further processing.

    Args:
        config: RoadConfig with project parameters.
        terrain_sampler: Pre-constructed TerrainSampler.

    Returns:
        A combined, repaired trimesh.Trimesh of all roads.
    """
    loader = RoadNetworkLoader(config)
    type_polys, *_ = loader.load()

    if config.variant == 'experimental':
        log.info("Using experimental raster road generator")
        mesh = _build_raster_road_mesh(config, terrain_sampler, type_polys)
    else:
        builder = RoadMeshBuilder(config, terrain_sampler)
        mesh = builder.build_all(type_polys)
    return mesh


# ── Experimental Raster Road Generator ────────────────────────────────────────

def _build_raster_road_mesh(
    config: RoadConfig,
    terrain_sampler: TerrainSampler,
    type_polys: Dict[str, List[Polygon]],
) -> trimesh.Trimesh:
    """Build road mesh using raster/grid approach instead of polygon triangulation.
    
    Strategy:
    - Create a 2D grid covering the plate (e.g., 0.2mm resolution)
    - Rasterize all road polygons onto this grid
    - For each road cell, sample terrain Z and add road raise
    - Generate quad mesh from the grid
    - Convert quads to triangles
    
    Args:
        config: RoadConfig with project parameters.
        terrain_sampler: Pre-constructed TerrainSampler.
        type_polys: Dict of road type → list of polygons.
        
    Returns:
        A combined, repaired trimesh.Trimesh of rasterized roads.
    """
    from shapely.ops import unary_union as _uu
    
    # Grid resolution (mm per cell) - finer for smoother curves
    # 0.1mm = good balance, 0.05mm = ultra-smooth but 4x slower
    grid_res = 0.05
    
    # Create grid
    nx = int(np.ceil(config.plate_w / grid_res))
    ny = int(np.ceil(config.plate_h / grid_res))
    
    log.info(f"Raster grid: {nx} x {ny} cells ({grid_res}mm resolution)")
    
    # Merge all road polygons
    all_polys = []
    for polys in type_polys.values():
        all_polys.extend([p for p in polys if p is not None and not p.is_empty and p.is_valid])
    
    if not all_polys:
        log.warning("No valid road polygons for raster generation")
        return trimesh.Trimesh()
    
    merged = _uu(all_polys).buffer(0)
    if isinstance(merged, MultiPolygon):
        merged_polys = list(merged.geoms)
    else:
        merged_polys = [merged]
    
    log.info(f"Merged {len(all_polys)} road polygons into {len(merged_polys)}")
    
    # Build raster mask
    road_mask = np.zeros((ny, nx), dtype=bool)
    
    # Get max road raise height
    max_raise = max(config.road_raise.values())
    
    # Rasterize each merged polygon
    log.info("Rasterizing road polygons...")
    for poly in tqdm(merged_polys, desc="Rasterize", unit="poly"):
        if poly.is_empty or not poly.is_valid:
            continue
            
        # Get bounding box in grid coordinates
        minx, miny, maxx, maxy = poly.bounds
        
        i_min = max(0, int(np.floor(minx / grid_res)))
        i_max = min(nx, int(np.ceil(maxx / grid_res)) + 1)
        j_min = max(0, int(np.floor(miny / grid_res)))
        j_max = min(ny, int(np.ceil(maxy / grid_res)) + 1)
        
        # Check each cell in bounding box
        for j in range(j_min, j_max):
            for i in range(i_min, i_max):
                x = (i + 0.5) * grid_res
                y = (j + 0.5) * grid_res
                pt = Point(x, y)
                if poly.contains(pt) or poly.intersects(pt):
                    road_mask[j, i] = True
    
    # Count road cells
    n_road_cells = np.sum(road_mask)
    log.info(f"Road cells: {n_road_cells:,} / {nx * ny:,} ({100.0 * n_road_cells / (nx * ny):.2f}%)")
    
    if n_road_cells == 0:
        log.warning("No road cells rasterized")
        return trimesh.Trimesh()
    
    # Build mesh from grid
    vertices = []
    faces = []
    
    # Create vertices for road cells (top surface)
    log.info("Building mesh from raster...")
    vert_idx_map = np.full((ny, nx), -1, dtype=int)
    
    for j in range(ny):
        for i in range(nx):
            if road_mask[j, i]:
                x = (i + 0.5) * grid_res
                y = (j + 0.5) * grid_res
                
                # Sample terrain Z
                z_terrain = terrain_sampler.sample(x, y)
                z_road = z_terrain + max_raise
                
                vert_idx_map[j, i] = len(vertices)
                vertices.append([x, y, z_road])
    
    # Build quad faces (convert to triangles)
    for j in range(ny - 1):
        for i in range(nx - 1):
            if road_mask[j, i]:
                # Check neighbors for quad
                v00 = vert_idx_map[j, i]
                v10 = vert_idx_map[j, i + 1] if i + 1 < nx and road_mask[j, i + 1] else -1
                v01 = vert_idx_map[j + 1, i] if j + 1 < ny and road_mask[j + 1, i] else -1
                v11 = vert_idx_map[j + 1, i + 1] if j + 1 < ny and i + 1 < nx and road_mask[j + 1, i + 1] else -1
                
                # Create triangles for complete quads
                if v00 >= 0 and v10 >= 0 and v01 >= 0 and v11 >= 0:
                    # Two triangles per quad
                    faces.append([v00, v10, v11])
                    faces.append([v00, v11, v01])
    
    if not faces:
        log.warning("No faces generated from raster")
        return trimesh.Trimesh()
    
    vertices = np.array(vertices, dtype=np.float64)
    faces = np.array(faces, dtype=np.int32)
    
    # Create top surface mesh
    top_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    
    log.info(f"Raster road mesh: {len(vertices):,} verts, {len(faces):,} faces")
    
    # Add bottom surface and side walls for watertight mesh
    # Bottom vertices at base_min_z
    bottom_verts = vertices.copy()
    bottom_verts[:, 2] = config.base_min_z
    
    # Flip face winding for bottom
    bottom_faces = faces[:, [0, 2, 1]] + len(vertices)
    
    # Build side walls by finding boundary edges
    # An edge is on the boundary if it appears in exactly one triangle
    edge_counts = {}
    for face in faces:
        for i in range(3):
            v1 = face[i]
            v2 = face[(i + 1) % 3]
            edge = tuple(sorted([v1, v2]))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    
    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    
    log.info(f"Building {len(boundary_edges):,} side wall quads...")
    
    # Create side wall faces
    wall_faces = []
    for v1, v2 in boundary_edges:
        # Top edge vertices
        v1_top = v1
        v2_top = v2
        # Bottom edge vertices
        v1_bot = v1 + len(vertices)
        v2_bot = v2 + len(vertices)
        
        # Two triangles for the wall quad
        wall_faces.append([v1_top, v2_top, v2_bot])
        wall_faces.append([v1_top, v2_bot, v1_bot])
    
    wall_faces = np.array(wall_faces, dtype=np.int32)
    
    all_verts = np.vstack([vertices, bottom_verts])
    all_faces = np.vstack([faces, bottom_faces, wall_faces])
    
    mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=False)
    
    # Aggressive repair to make watertight
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_inversion(mesh)
    trimesh.repair.fix_winding(mesh)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.merge_vertices()
    
    log.info(f"Pre-smoothing: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces")
    
    # Multi-pass smoothing to reduce grid stepping artifacts
    try:
        # First pass: light smoothing to preserve detail
        mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=2, lamb=0.3)
        log.info(f"After light smoothing: {len(mesh.vertices):,} verts")
        
        # Second pass: stronger smoothing for ultra-smooth result
        mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=3, lamb=0.5)
        log.info(f"After strong smoothing: {len(mesh.vertices):,} verts")
        
        # Final Taubin smoothing pass to reduce shrinkage
        mesh = trimesh.smoothing.filter_taubin(mesh, iterations=5)
        log.info(f"Post-smoothing: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces")
    except Exception as e:
        log.warning(f"Smoothing failed: {e}")
    
    log.info(f"Final raster mesh: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces, watertight={mesh.is_watertight}")
    
    return mesh


# ── Internal helpers ──────────────────────────────────────────────────────────

def _to_mm(
    geom,
    scale: float,
    minx: float,
    miny: float,
    off_x: float,
    off_y: float,
):
    """Transform projected geometry to mm print-plate space.

    Args:
        geom: Shapely geometry in projected CRS (metres).
        scale: Uniform scale factor (mm/m).
        minx: Left bound of projected extent.
        miny: Bottom bound of projected extent.
        off_x: X offset to centre geometry on plate (mm).
        off_y: Y offset to centre geometry on plate (mm).

    Returns:
        Shapely geometry in mm-space.
    """
    g = affinity.translate(geom, xoff=-minx, yoff=-miny)
    g = affinity.scale(g, xfact=scale, yfact=scale, origin=(0, 0))
    return affinity.translate(g, xoff=off_x, yoff=off_y)


def _repair(mesh: trimesh.Trimesh) -> None:
    """Apply standard manifold repair operations to a mesh in-place.

    Args:
        mesh: trimesh.Trimesh to repair.
    """
    mesh.merge_vertices(digits_vertex=4)
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    trimesh.repair.fix_normals(mesh)


# ── Main entry point ──────────────────────────────────────────────────────────

def main() -> None:
    """Run the full road mesh generation pipeline."""
    import sys
    os.makedirs("output", exist_ok=True)

    print("=" * 60)
    print("  road_mesh — OSM road network to printable STL")
    print("=" * 60)

    config = RoadConfig()

    # ── Terrain sampler ───────────────────────────────────────────────────────
    if not os.path.exists(config.base_stl):
        log.warning(
            "Base STL not found at %s — using fallback Z=%.1f for all vertices.",
            config.base_stl,
            config.base_min_z,
        )

        class _FlatSampler:
            """Minimal TerrainSampler substitute that returns constant Z."""

            def sample(self, x: float, y: float) -> float:
                return config.base_min_z

            def sample_batch(self, xy: np.ndarray) -> np.ndarray:
                return np.full(len(xy), config.base_min_z)

        terrain = _FlatSampler()
    else:
        terrain = TerrainSampler(
            mode="from_stl",
            stl_path=config.base_stl,
            fallback_z=config.base_min_z,
        )

    # ── Load road network ─────────────────────────────────────────────────────
    loader = RoadNetworkLoader(config)
    type_polys, *_ = loader.load()

    # ── Build meshes ──────────────────────────────────────────────────────────
    builder = RoadMeshBuilder(config, terrain)
    combined = builder.build_all(type_polys)

    # ── Export + validate ─────────────────────────────────────────────────────
    exporter = STLExporter(config)
    exporter.export(combined)

    print("\n✓  road_mesh complete.")


if __name__ == "__main__":
    main()
