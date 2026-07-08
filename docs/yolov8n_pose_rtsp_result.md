# YOLOv8n-pose RTSP Live Demo

작성일: 2026-07-08

## 구성

- Board: Orange Pi 5 Plus / RK3588
- Camera: TP-Link Tapo RTSP camera
- RTSP: `rtsp://cmpark:cmpark12@192.168.10.111:554/stream1`
- Model: `yolov8n-pose.rknn`
- Display: Orange Pi HDMI monitor

## 수정 내용

기존 `rknn_yolov8_pose_demo`는 `/dev/video0` 또는 숫자 camera index만 지원했다.

`main_camera.cc`를 수정해서 아래 입력도 지원하도록 변경했다.

```text
rtsp://...
input.mp4
```

RTSP 입력은 OpenCV `cv::CAP_FFMPEG`로 열고, low latency를 위해:

```cpp
cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
```

을 추가했다.

## 실행 명령

```bash
cd /home/orangepi/rknn_model_zoo/install/rk3588_linux_aarch64/rknn_yolov8_pose_demo

DISPLAY=:0 XAUTHORITY=/home/orangepi/.Xauthority \
OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;udp|max_delay;0|fflags;nobuffer|flags;low_delay" \
./rknn_yolov8_pose_demo \
  model/yolov8n-pose.rknn \
  rtsp://cmpark:cmpark12@192.168.10.111:554/stream1
```

## 현재 실행 상태

```text
PID 6550
./rknn_yolov8_pose_demo model/yolov8n-pose.rknn rtsp://cmpark:cmpark12@192.168.10.111:554/stream1
```

## 성능

최근 로그 기준:

```text
rknn_run time: 약 21.5 ~ 27.4 ms
NPU 추론 기준: 약 36 ~ 46 FPS
post_process: 약 0.2 ~ 0.6 ms
```

실제 화면 표시 FPS는 카메라 RTSP 입력 15 FPS와 OpenCV 표시 영향으로 제한될 수 있다.

## 저장 파일

- `source/main_camera.cc`
- `yolov8n_pose_display_capture.png`
- `yolov8n_pose_live.log`
- `yolov8n_pose_perf_only.log`
- `yolov8n_pose_rtsp_result.md`
