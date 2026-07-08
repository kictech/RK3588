#!/usr/bin/env python3
"""Aggregate YOLOv8n + ByteTrack CSV logs into space-analysis JSON.

Usage:
  python aggregate_tracks.py --tracks track_log.csv --zones zones.json --out summary.json --report report.md
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", required=True, help="ByteTrack CSV log path")
    parser.add_argument("--zones", required=True, help="zones.json path")
    parser.add_argument("--out", default="space_summary.json", help="Output summary JSON")
    parser.add_argument("--report", default="space_report.md", help="Output Markdown report")
    parser.add_argument("--sample-sec", type=float, default=1.0, help="Approximate dwell increment per visible row")
    parser.add_argument("--track-width", type=int, default=1920, help="Tracking CSV coordinate width")
    parser.add_argument("--track-height", type=int, default=1080, help="Tracking CSV coordinate height")
    return parser.parse_args()


def point_in_polygon(x: float, y: float, polygon: list[list[float]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def load_zones(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def scale_zones_to_track_frame(zones_doc: dict, track_width: int, track_height: int) -> dict:
    src = zones_doc.get("frame_size") or {}
    src_width = float(src.get("width") or track_width)
    src_height = float(src.get("height") or track_height)
    if src_width <= 0 or src_height <= 0:
        return zones_doc

    sx = track_width / src_width
    sy = track_height / src_height
    scaled = dict(zones_doc)
    scaled_zones = []
    for zone in zones_doc.get("zones", []):
        scaled_zone = dict(zone)
        scaled_zone["polygon"] = [
            [round(point[0] * sx), round(point[1] * sy)]
            for point in zone.get("polygon", [])
        ]
        scaled_zones.append(scaled_zone)
    scaled["zones"] = scaled_zones
    scaled["source_frame_size"] = zones_doc.get("frame_size")
    scaled["frame_size"] = {"width": track_width, "height": track_height}
    scaled["coordinate_scale"] = {"x": sx, "y": sy}
    return scaled


def zone_for_point(x: float, y: float, zones: list[dict]) -> str:
    for zone in zones:
        if point_in_polygon(x, y, zone["polygon"]):
            return zone["id"]
    return "unknown"


def parse_time(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def aggregate(tracks_path: Path, zones_doc: dict, sample_sec: float) -> dict:
    zones = zones_doc["zones"]
    zone_names = {z["id"]: z.get("name", z["id"]) for z in zones}
    zone_names["unknown"] = "unknown"

    visible_rows = []
    all_rows = []
    with tracks_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_rows.append(row)
            if row.get("visible") == "1":
                row["foot_x"] = float(row["foot_x"])
                row["foot_y"] = float(row["foot_y"])
                row["track_id"] = int(row["track_id"])
                row["frame"] = int(row["frame"])
                row["confidence"] = float(row["confidence"])
                row["dt"] = parse_time(row["timestamp"])
                row["zone_id"] = zone_for_point(row["foot_x"], row["foot_y"], zones)
                visible_rows.append(row)

    if not visible_rows:
        return {
            "site": zones_doc.get("site", "unknown"),
            "summary": {
                "visible_rows": 0,
                "unique_tracks": 0,
                "max_people": 0,
                "average_people": 0,
            },
            "zones": [],
            "movement_flows": [],
            "privacy_policy": default_privacy_policy(),
        }

    frames = defaultdict(set)
    zones_seen_by_track = defaultdict(list)
    zone_visit_keys = set()
    zone_dwell_sec = Counter()
    zone_visible_rows = Counter()

    for row in visible_rows:
        frames[row["frame"]].add(row["track_id"])
        zone_id = row["zone_id"]
        track_id = row["track_id"]
        zone_visible_rows[zone_id] += 1
        zone_dwell_sec[zone_id] += sample_sec
        if not zones_seen_by_track[track_id] or zones_seen_by_track[track_id][-1] != zone_id:
            zones_seen_by_track[track_id].append(zone_id)
            zone_visit_keys.add((track_id, zone_id, len(zones_seen_by_track[track_id])))

    people_counts = [len(ids) for ids in frames.values()]
    movement_flows = Counter()
    for path in zones_seen_by_track.values():
        compressed = [z for z in path if z != "unknown"]
        for a, b in zip(compressed, compressed[1:]):
            if a != b:
                movement_flows[(a, b)] += 1

    start = min((r["dt"] for r in visible_rows if r["dt"]), default=None)
    end = max((r["dt"] for r in visible_rows if r["dt"]), default=None)

    zone_results = []
    for zone_id, dwell in zone_dwell_sec.most_common():
        zone_results.append(
            {
                "zone_id": zone_id,
                "name": zone_names.get(zone_id, zone_id),
                "visit_count": sum(1 for key in zone_visit_keys if key[1] == zone_id),
                "total_dwell_sec": round(dwell, 1),
                "visible_rows": zone_visible_rows[zone_id],
            }
        )

    flow_results = [
        {
            "from": a,
            "from_name": zone_names.get(a, a),
            "to": b,
            "to_name": zone_names.get(b, b),
            "count": count,
        }
        for (a, b), count in movement_flows.most_common()
    ]

    return {
        "site": zones_doc.get("site", "unknown"),
        "analysis_period": {
            "start": start.strftime("%Y-%m-%d %H:%M:%S") if start else None,
            "end": end.strftime("%Y-%m-%d %H:%M:%S") if end else None,
        },
        "summary": {
            "visible_rows": len(visible_rows),
            "raw_rows": len(all_rows),
            "unique_tracks": len({r["track_id"] for r in visible_rows}),
            "max_people": max(people_counts),
            "average_people": round(sum(people_counts) / len(people_counts), 2),
            "frame_count": len(frames),
        },
        "zones": zone_results,
        "movement_flows": flow_results,
        "privacy_policy": default_privacy_policy(),
    }


def default_privacy_policy() -> dict:
    return {
        "face_recognition": False,
        "raw_video_storage": False,
        "person_id_type": "temporary_tracking_id",
        "stored_data": "coordinates, zones, dwell time, movement flow statistics",
    }


def write_report(summary: dict, path: Path) -> None:
    lines = [
        "# 공간 분석 요약 리포트",
        "",
        f"- 사이트: {summary.get('site')}",
        f"- 분석 시작: {summary.get('analysis_period', {}).get('start')}",
        f"- 분석 종료: {summary.get('analysis_period', {}).get('end')}",
        "",
        "## 재실 요약",
        "",
        f"- 고유 임시 ID 수: {summary['summary']['unique_tracks']}",
        f"- 최대 동시 인원: {summary['summary']['max_people']}",
        f"- 평균 인원: {summary['summary']['average_people']}",
        "",
        "## 구역별 체류",
        "",
    ]
    for zone in summary.get("zones", []):
        lines.append(
            f"- {zone['name']}: 방문 {zone['visit_count']}회, 체류 {zone['total_dwell_sec']}초"
        )

    lines += ["", "## 주요 이동 흐름", ""]
    for flow in summary.get("movement_flows", []):
        lines.append(f"- {flow['from_name']} -> {flow['to_name']}: {flow['count']}회")

    lines += [
        "",
        "## 개인정보 보호",
        "",
        "- 얼굴 인식은 사용하지 않았습니다.",
        "- 원본 영상은 저장하지 않는 설계를 권장합니다.",
        "- 사람 ID는 세션 내 임시 tracking ID입니다.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    zones_doc = scale_zones_to_track_frame(load_zones(Path(args.zones)), args.track_width, args.track_height)
    summary = aggregate(Path(args.tracks), zones_doc, args.sample_sec)
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, Path(args.report))
    print(json.dumps(summary["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
