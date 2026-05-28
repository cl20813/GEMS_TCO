#!/usr/bin/env python3
"""Check candidate boxes against Natural Earth 1:10m land polygons."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


LAND_GEOJSON = Path("/private/tmp/ne_10m_land.geojson")
OUT_DIR = Path("/Users/joonwonlee/Documents/GEMS_TCO-1/GEMS_TCO_EDA")
SUMMARY_CSV = OUT_DIR / "open_water_candidate_land_check_052826_summary.csv"
NEARBY_CSV = OUT_DIR / "open_water_candidate_land_check_052826_nearby.csv"
REPORT_MD = OUT_DIR / "open_water_candidate_land_check_052826.md"
MAP_PNG = OUT_DIR / "open_water_candidate_land_check_052826.png"

CANDIDATES = [
    ("lat-5to5_lon75to95", -5.0, 5.0, 75.0, 95.0),
    ("lat15to20_lon123to145", 15.0, 20.0, 123.0, 145.0),
    ("lat15to20_lon123to132", 15.0, 20.0, 123.0, 132.0),
    ("lat15to20_lon122to132", 15.0, 20.0, 122.0, 132.0),
]


def ring_bbox(ring: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [pt[0] for pt in ring]
    ys = [pt[1] for pt in ring]
    return min(xs), max(xs), min(ys), max(ys)


def bboxes_overlap(a, b) -> bool:
    ax0, ax1, ay0, ay1 = a
    bx0, bx1, by0, by1 = b
    return ax0 <= bx1 and ax1 >= bx0 and ay0 <= by1 and ay1 >= by0


def point_in_box(lon: float, lat: float, box) -> bool:
    lat_min, lat_max, lon_min, lon_max = box
    return lon_min <= lon <= lon_max and lat_min <= lat <= lat_max


def point_in_ring(lon: float, lat: float, ring: list[list[float]]) -> bool:
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        crosses = (yi > lat) != (yj > lat)
        if crosses:
            x_at_lat = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < x_at_lat:
                inside = not inside
        j = i
    return inside


def orient(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def on_segment(a, b, c) -> bool:
    return (
        min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
        and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
        and abs(orient(a, b, c)) < 1e-12
    )


def segments_intersect(a, b, c, d) -> bool:
    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    return (
        on_segment(a, b, c)
        or on_segment(a, b, d)
        or on_segment(c, d, a)
        or on_segment(c, d, b)
    )


def ring_intersects_box(ring: list[list[float]], box) -> bool:
    lat_min, lat_max, lon_min, lon_max = box
    box_corners = [
        (lon_min, lat_min),
        (lon_max, lat_min),
        (lon_max, lat_max),
        (lon_min, lat_max),
    ]
    box_edges = list(zip(box_corners, box_corners[1:] + box_corners[:1]))
    if any(point_in_box(lon, lat, box) for lon, lat in ring):
        return True
    if any(point_in_ring(lon, lat, ring) for lon, lat in box_corners):
        return True
    for i in range(len(ring) - 1):
        a = (ring[i][0], ring[i][1])
        b = (ring[i + 1][0], ring[i + 1][1])
        for c, d in box_edges:
            if segments_intersect(a, b, c, d):
                return True
    return False


def km_per_degree(lat: float) -> tuple[float, float]:
    lat_rad = math.radians(lat)
    return 111.32 * math.cos(lat_rad), 110.57


def project(lon: float, lat: float, ref_lat: float) -> tuple[float, float]:
    kx, ky = km_per_degree(ref_lat)
    return lon * kx, lat * ky


def point_segment_distance_km(point, a, b, ref_lat: float) -> tuple[float, tuple[float, float]]:
    px, py = project(point[0], point[1], ref_lat)
    ax, ay = project(a[0], a[1], ref_lat)
    bx, by = project(b[0], b[1], ref_lat)
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay), a
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    qx = ax + t * dx
    qy = ay + t * dy
    lon = qx / km_per_degree(ref_lat)[0]
    lat = qy / km_per_degree(ref_lat)[1]
    return math.hypot(px - qx, py - qy), (lon, lat)


def point_box_distance_km(lon: float, lat: float, box) -> float:
    lat_min, lat_max, lon_min, lon_max = box
    clamped_lon = min(max(lon, lon_min), lon_max)
    clamped_lat = min(max(lat, lat_min), lat_max)
    ref_lat = (lat + clamped_lat) / 2.0
    x1, y1 = project(lon, lat, ref_lat)
    x2, y2 = project(clamped_lon, clamped_lat, ref_lat)
    return math.hypot(x1 - x2, y1 - y2)


def ring_distance_to_box_km(ring: list[list[float]], box) -> tuple[float, tuple[float, float]]:
    lat_min, lat_max, lon_min, lon_max = box
    box_corners = [
        (lon_min, lat_min),
        (lon_max, lat_min),
        (lon_max, lat_max),
        (lon_min, lat_max),
    ]
    best = (float("inf"), (float("nan"), float("nan")))
    for lon, lat in ring:
        d = point_box_distance_km(lon, lat, box)
        if d < best[0]:
            best = (d, (lon, lat))
    for corner in box_corners:
        ref_lat = corner[1]
        for i in range(len(ring) - 1):
            a = (ring[i][0], ring[i][1])
            b = (ring[i + 1][0], ring[i + 1][1])
            d, nearest = point_segment_distance_km(corner, a, b, ref_lat)
            if d < best[0]:
                best = (d, nearest)
    return best


def polygon_rings(geometry: dict) -> list[list[list[float]]]:
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"][0]]
    if geometry["type"] == "MultiPolygon":
        return [poly[0] for poly in geometry["coordinates"]]
    return []


def main() -> None:
    if not LAND_GEOJSON.exists():
        raise FileNotFoundError(f"Missing land GeoJSON: {LAND_GEOJSON}")
    data = json.loads(LAND_GEOJSON.read_text())
    rings = []
    for idx, feature in enumerate(data["features"]):
        for ridx, ring in enumerate(polygon_rings(feature["geometry"])):
            rings.append({"feature_index": idx, "ring_index": ridx, "ring": ring, "bbox": ring_bbox(ring)})

    summary_rows = []
    nearby_rows = []
    for name, lat_min, lat_max, lon_min, lon_max in CANDIDATES:
        box = (lat_min, lat_max, lon_min, lon_max)
        box_bbox = (lon_min, lon_max, lat_min, lat_max)
        intersections = []
        distances = []
        for item in rings:
            if bboxes_overlap(item["bbox"], box_bbox) and ring_intersects_box(item["ring"], box):
                intersections.append(item)
            d_km, nearest = ring_distance_to_box_km(item["ring"], box)
            distances.append((d_km, nearest, item["bbox"]))
        distances.sort(key=lambda x: x[0])
        nearest_10 = distances[:10]
        for rank, (d_km, nearest, rbbox) in enumerate(nearest_10, start=1):
            nearby_rows.append(
                {
                    "candidate": name,
                    "rank": rank,
                    "distance_km": d_km,
                    "nearest_land_lon": nearest[0],
                    "nearest_land_lat": nearest[1],
                    "land_bbox_lon_min": rbbox[0],
                    "land_bbox_lon_max": rbbox[1],
                    "land_bbox_lat_min": rbbox[2],
                    "land_bbox_lat_max": rbbox[3],
                }
            )
        summary_rows.append(
            {
                "candidate": name,
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
                "land_intersects": bool(intersections),
                "n_intersecting_land_polygons": len(intersections),
                "nearest_land_distance_km": nearest_10[0][0],
                "nearest_land_lon": nearest_10[0][1][0],
                "nearest_land_lat": nearest_10[0][1][1],
            }
        )

    summary = pd.DataFrame(summary_rows)
    nearby = pd.DataFrame(nearby_rows)
    summary.to_csv(SUMMARY_CSV, index=False)
    nearby.to_csv(NEARBY_CSV, index=False)
    plot_map(rings, MAP_PNG)

    lines = [
        "# Open Water Candidate Land Check",
        "",
        "Source: Natural Earth 1:10m land polygon GeoJSON (`ne_10m_land`).",
        "Intersection test uses exterior land polygon rings against each candidate bbox.",
        "",
        "| candidate | land_intersects | nearest_land_distance_km | nearest_land_lon | nearest_land_lat |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            f"| {row['candidate']} | {row['land_intersects']} | "
            f"{row['nearest_land_distance_km']:.1f} | {row['nearest_land_lon']:.4f} | "
            f"{row['nearest_land_lat']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Summary CSV: `{SUMMARY_CSV}`",
            f"Nearby CSV: `{NEARBY_CSV}`",
            f"Map PNG: `{MAP_PNG}`",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {NEARBY_CSV}")
    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {MAP_PNG}")


def plot_map(rings: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=160)
    panels = [
        ("Equatorial Indian Ocean", (70, 100, -10, 10), CANDIDATES[0]),
        ("West Pacific / Philippine Sea", (118, 150, 12, 23), CANDIDATES[1]),
    ]
    for ax, (title, extent, candidate) in zip(axes, panels):
        xmin, xmax, ymin, ymax = extent
        ax.set_title(title)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, color="0.85", linewidth=0.6)
        for item in rings:
            rb = item["bbox"]
            if not bboxes_overlap(rb, (xmin, xmax, ymin, ymax)):
                continue
            ring = item["ring"]
            xs = [p[0] for p in ring]
            ys = [p[1] for p in ring]
            ax.fill(xs, ys, color="0.55", alpha=0.9, linewidth=0)
        name, lat_min, lat_max, lon_min, lon_max = candidate
        ax.add_patch(
            Rectangle(
                (lon_min, lat_min),
                lon_max - lon_min,
                lat_max - lat_min,
                fill=False,
                edgecolor="#d62728",
                linewidth=2.0,
            )
        )
        ax.text(lon_min, lat_max + 0.3, name, color="#d62728", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
