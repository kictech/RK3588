#!/usr/bin/env bash
set -u

cd /home/orangepi/rknn_model_zoo/install/rk3588_linux_aarch64/rknn_yolov8_demo

# Camera-side image flip is enabled, so the application does not rotate frames.
while true; do
  if ping -c 1 -W 1 192.168.10.111 >/dev/null 2>&1; then
    echo "camera_ready $(date)"
    env DISPLAY=:0 \
      XAUTHORITY=/home/orangepi/.Xauthority \
      OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;udp|max_delay;0|fflags;nobuffer|flags;low_delay" \
      ./rknn_yolov8_bytetrack_live_demo \
      model/yolov8n_runtime152.rknn \
      rtsp://cmpark:cmpark12@192.168.10.111:554/stream1 \
      /tmp/yolov8n_bytetrack_tracks.csv
  fi
  echo "waiting_camera $(date)"
  sleep 10
done
