#!/usr/bin/env python3
"""Render dwell-time and movement-flow infographic over a camera map image."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Background capture image")
    parser.add_argument("--zones", required=True, help="zones.json")
    parser.add_argument("--summary", required=True, help="space_summary_*.json")
    parser.add_argument("--out", required=True, help="Output PNG")
    return parser.parse_args()


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\malgunbd.ttf" if bold else r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def centroid(points: list[list[float]]) -> tuple[float, float]:
    if not points:
        return 0.0, 0.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def format_time(seconds: float) -> str:
    minutes = seconds / 60.0
    if minutes >= 60:
        return f"{minutes / 60.0:.1f}h"
    return f"{minutes:.1f}m"


def polygon_bbox(points: list[list[float]]) -> tuple[int, int, int, int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def draw_label_box(draw: ImageDraw.ImageDraw, xy: tuple[int, int], lines: list[str],
                   font: ImageFont.ImageFont, fill=(0, 0, 0, 185), text=(255, 255, 255, 255)) -> None:
    x, y = xy
    line_h = font.getbbox("Ag")[3] - font.getbbox("Ag")[1] + 4
    width = max(draw.textlength(line, font=font) for line in lines) + 14
    height = line_h * len(lines) + 10
    draw.rounded_rectangle([x, y, x + width, y + height], radius=6, fill=fill)
    for i, line in enumerate(lines):
        draw.text((x + 7, y + 5 + i * line_h), line, font=font, fill=text)


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float],
               color: tuple[int, int, int, int], width: int) -> None:
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    length = math.hypot(dx, dy)
    if length < 1:
        return
    ux = dx / length
    uy = dy / length
    start2 = (sx + ux * 18, sy + uy * 18)
    end2 = (ex - ux * 22, ey - uy * 22)
    draw.line([start2, end2], fill=color, width=width)

    angle = math.atan2(dy, dx)
    head_len = 14 + width
    left = (
        end2[0] - head_len * math.cos(angle - math.pi / 6),
        end2[1] - head_len * math.sin(angle - math.pi / 6),
    )
    right = (
        end2[0] - head_len * math.cos(angle + math.pi / 6),
        end2[1] - head_len * math.sin(angle + math.pi / 6),
    )
    draw.polygon([end2, left, right], fill=color)


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    zones_path = Path(args.zones)
    summary_path = Path(args.summary)
    out_path = Path(args.out)

    base = Image.open(image_path).convert("RGBA")
    zones_doc = json.loads(zones_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    img_w, img_h = base.size
    src_size = zones_doc.get("frame_size", {"width": img_w, "height": img_h})
    sx = img_w / float(src_size.get("width") or img_w)
    sy = img_h / float(src_size.get("height") or img_h)

    panel_w = 420
    canvas = Image.new("RGBA", (img_w + panel_w, img_h), (245, 247, 250, 255))
    canvas.alpha_composite(base, (0, 0))
    overlay = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    title_font = load_font(22, True)
    font = load_font(15)
    small_font = load_font(13)
    panel_font = load_font(14)
    panel_bold = load_font(16, True)

    zone_stats = {z["zone_id"]: z for z in summary.get("zones", [])}
    max_dwell = max([z.get("total_dwell_sec", 0) for z in summary.get("zones", [])] or [1])
    color_palette = [
        (255, 91, 64, 190),
        (255, 173, 51, 180),
        (62, 166, 255, 175),
        (49, 204, 139, 175),
        (166, 112, 255, 175),
        (255, 91, 159, 175),
        (80, 220, 220, 175),
    ]

    zone_centers: dict[str, tuple[float, float]] = {}
    for idx, zone in enumerate(zones_doc.get("zones", [])):
        zone_id = zone["id"]
        points = [[round(p[0] * sx), round(p[1] * sy)] for p in zone.get("polygon", [])]
        if len(points) < 3:
            continue
        stats = zone_stats.get(zone_id, {})
        dwell = float(stats.get("total_dwell_sec", 0))
        alpha_scale = 0.22 + 0.55 * (dwell / max_dwell if max_dwell else 0)
        color = color_palette[idx % len(color_palette)]
        fill = (color[0], color[1], color[2], int(255 * alpha_scale))
        outline = (color[0], color[1], color[2], 245)
        draw.polygon([tuple(p) for p in points], fill=fill)
        draw.line([tuple(p) for p in points + [points[0]]], fill=outline, width=3)

        cx, cy = centroid(points)
        zone_centers[zone_id] = (cx, cy)
        x1, y1, _, _ = polygon_bbox(points)
        label = zone.get("name") or zone_id
        lines = [
            f"{label} ({zone_id})",
            f"dwell {format_time(dwell)}",
            f"visits {stats.get('visit_count', 0)}",
        ]
        draw_label_box(draw, (max(4, x1 + 4), max(4, y1 + 4)), lines, small_font)

    # Movement arrows
    flows = summary.get("movement_flows", [])[:12]
    max_flow = max([f.get("count", 0) for f in flows] or [1])
    for flow in flows:
        start = zone_centers.get(flow.get("from"))
        end = zone_centers.get(flow.get("to"))
        if not start or not end:
            continue
        width = 3 + int(5 * flow.get("count", 0) / max_flow)
        draw_arrow(draw, start, end, (15, 35, 55, 230), width)
        mx = int((start[0] + end[0]) / 2)
        my = int((start[1] + end[1]) / 2)
        draw_label_box(draw, (mx + 6, my - 18), [f"{flow.get('count', 0)}"], small_font,
                       fill=(255, 255, 255, 220), text=(0, 0, 0, 255))

    canvas.alpha_composite(overlay, (0, 0))
    panel = ImageDraw.Draw(canvas)
    px = img_w
    panel.rectangle([px, 0, img_w + panel_w, img_h], fill=(250, 251, 253, 255))
    panel.line([px, 0, px, img_h], fill=(210, 216, 226, 255), width=1)

    y = 18
    panel.text((px + 20, y), "10min Space Analytics", font=title_font, fill=(20, 30, 42, 255))
    y += 36
    period = summary.get("analysis_period", {})
    panel.text((px + 20, y), f"{period.get('start')} ~", font=small_font, fill=(70, 80, 95, 255))
    y += 18
    panel.text((px + 20, y), f"{period.get('end')}", font=small_font, fill=(70, 80, 95, 255))
    y += 30

    s = summary.get("summary", {})
    metric_lines = [
        ("Max people", s.get("max_people", 0)),
        ("Average people", s.get("average_people", 0)),
        ("Unique temp IDs", s.get("unique_tracks", 0)),
        ("Visible rows", s.get("visible_rows", 0)),
    ]
    for name, value in metric_lines:
        panel.text((px + 20, y), name, font=small_font, fill=(83, 94, 110, 255))
        panel.text((px + 190, y - 2), str(value), font=panel_bold, fill=(20, 30, 42, 255))
        y += 26

    y += 10
    panel.text((px + 20, y), "Dwell Time By Zone", font=panel_bold, fill=(20, 30, 42, 255))
    y += 28
    for zone in summary.get("zones", [])[:8]:
        zid = zone.get("zone_id")
        name = zone.get("name") or zid
        dwell = zone.get("total_dwell_sec", 0)
        bar_w = int(210 * (dwell / max_dwell if max_dwell else 0))
        panel.text((px + 20, y), f"{name} ({zid})", font=small_font, fill=(30, 40, 52, 255))
        panel.rounded_rectangle([px + 20, y + 18, px + 20 + bar_w, y + 28], radius=4,
                                fill=(11, 127, 171, 220))
        panel.text((px + 245, y + 11), format_time(dwell), font=small_font, fill=(30, 40, 52, 255))
        y += 42

    y += 4
    panel.text((px + 20, y), "Top Movement Flows", font=panel_bold, fill=(20, 30, 42, 255))
    y += 26
    for flow in summary.get("movement_flows", [])[:8]:
        line = f"{flow.get('from')} -> {flow.get('to')}: {flow.get('count')}"
        panel.text((px + 20, y), line, font=panel_font, fill=(30, 40, 52, 255))
        y += 22

    footer = "No face recognition | Temporary tracking IDs | No raw video required"
    panel.text((px + 20, img_h - 28), footer, font=small_font, fill=(90, 100, 116, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out_path, quality=95)
    print(out_path)


if __name__ == "__main__":
    main()
