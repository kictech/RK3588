# Space Analysis Pipeline

This folder contains a first implementation for converting Orange Pi YOLOv8n + ByteTrack results into privacy-preserving space analytics.

## Files

- `main_bytetrack_live_display.cc`
  - Live RTSP input
  - 180-degree camera rotation
  - YOLOv8n person detection
  - ByteTrack-style ID tracking
  - Static track suppression
  - UDP low-latency friendly latest-frame reader
  - CSV track logging

- `zones.json`
  - Example polygon zones for a 1920x1080 frame.
  - Edit polygon points to match the real office/store layout.

- `aggregate_tracks.py`
  - Reads ByteTrack CSV and zones.json.
  - Produces `space_summary.json` and `space_report.md`.

## Orange Pi Run Command

```bash
cd /home/orangepi/rknn_model_zoo/install/rk3588_linux_aarch64/rknn_yolov8_demo

DISPLAY=:0 XAUTHORITY=/home/orangepi/.Xauthority \
OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;udp|max_delay;0|fflags;nobuffer|flags;low_delay" \
./rknn_yolov8_bytetrack_live_demo \
model/yolov8n_runtime152.rknn \
rtsp://cmpark:cmpark12@192.168.10.111:554/stream1 \
/tmp/yolov8n_bytetrack_tracks.csv
```

## Aggregate Command

```bash
python aggregate_tracks.py \
  --tracks yolov8n_bytetrack_tracks_sample.csv \
  --zones zones.json \
  --out space_summary.json \
  --report space_report.md
```

## CSV Columns

```text
timestamp,frame,track_id,class_id,class_name,confidence,
x1,y1,x2,y2,cx,cy,foot_x,foot_y,age,missed,total_motion,static_frames,visible
```

Use `foot_x,foot_y` for zone membership because it approximates the person's floor position.

## Privacy Notes

- Raw video does not need to be stored.
- Face recognition is not used.
- Track IDs are temporary IDs for the current session.
- The report should use aggregate counts, dwell time, zone usage, and movement flows.
