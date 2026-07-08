# YOLOv8n-pose RTSP + ByteTrack Live Demo

작성일: 2026-07-08

## 구성

- Board: Orange Pi 5 Plus / RK3588
- Camera: TP-Link Tapo RTSP camera
- RTSP: `rtsp://cmpark:cmpark12@192.168.10.111:554/stream1`
- Model: `yolov8n-pose.rknn`
- Display: Orange Pi HDMI monitor
- Tracking: ByteTrack-style two-stage IoU matching

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

추가로 pose 결과의 사람 bounding box를 ByteTrack-style tracker에 넣어, 같은 사람에게 가능한 한 같은 ID를 유지하도록 했다.
매칭된 track에는 skeleton keypoints를 같이 저장하므로 화면에는 bbox, ID, pose skeleton, foot trail이 함께 표시된다.

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

현재 ByteTrack 추가 버전:

```text
PID 8768
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

ByteTrack 추가 후 요약 로그 예:

```text
pose_bytetrack frame=750 high=2 low=0 tracks=2 visible=2
rknn_run time=20~30 ms 중심
```

## 저장 파일

- `source/main_camera.cc`
- `yolov8n_pose_display_capture.png`
- `yolov8n_pose_bytetrack_capture.png`
- `yolov8n_pose_live.log`
- `yolov8n_pose_perf_only.log`
- `yolov8n_pose_bytetrack_summary.log`
- `yolov8n_pose_rtsp_result.md`
