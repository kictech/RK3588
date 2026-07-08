# RK3588 Orange Pi 5 Plus Vision Demos

Orange Pi 5 Plus(RK3588)에서 Tapo RTSP 카메라 입력을 받아 YOLOv8n 객체 검출, ByteTrack-style 추적, 공간 분석 집계를 수행한 작업물입니다.

## 주요 내용

- YOLOv8n RKNN 실시간 표시
- SORT-style tracking + ID 표시
- DeepSORT-style appearance tracking 실험
- ByteTrack-style tracking + ID 표시
- YOLOv8n-pose RKNN RTSP 실시간 표시
- RTSP low-latency 입력
- static track suppression
- track CSV 저장
- zone 기반 체류시간/이동흐름 집계
- zone editor 및 인포그래픽 렌더링

## 폴더 구조

```text
src/yolov8-detection-tracker/   RKNN YOLOv8 detection/tracking C++ demo sources
src/yolov8-pose/                RKNN YOLOv8n-pose RTSP C++ demo source
scripts/                        Orange Pi 실행/대기 스크립트
space_analysis/                 zone editor, 집계, 인포그래픽 스크립트
results/screenshots/            결과 화면 캡처
results/pose/                   YOLOv8n-pose 성능 로그
results/summaries/              10분 집계 JSON/Markdown 결과
docs/                           작업 결과 및 수정 이력
```

## Orange Pi 실행 예

```bash
cd /home/orangepi/rknn_model_zoo/install/rk3588_linux_aarch64/rknn_yolov8_demo

DISPLAY=:0 XAUTHORITY=/home/orangepi/.Xauthority \
OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;udp|max_delay;0|fflags;nobuffer|flags;low_delay" \
./rknn_yolov8_bytetrack_live_demo \
  model/yolov8n_runtime152.rknn \
  rtsp://cmpark:cmpark12@192.168.10.111:554/stream1 \
  /tmp/yolov8n_bytetrack_tracks.csv
```

카메라가 자체 반전 처리 중이므로 현재 소스에서는 `cv::rotate(..., ROTATE_180)`을 사용하지 않습니다.

## YOLOv8n-pose 실행 예

```bash
cd /home/orangepi/rknn_model_zoo/install/rk3588_linux_aarch64/rknn_yolov8_pose_demo

DISPLAY=:0 XAUTHORITY=/home/orangepi/.Xauthority \
OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;udp|max_delay;0|fflags;nobuffer|flags;low_delay" \
./rknn_yolov8_pose_demo \
  model/yolov8n-pose.rknn \
  rtsp://cmpark:cmpark12@192.168.10.111:554/stream1
```

## 공간 분석

```bash
python space_analysis/aggregate_tracks.py \
  --tracks results/summaries/yolov8n_bytetrack_tracks_10min.csv \
  --zones space_analysis/zones.json \
  --out results/summaries/space_summary_10min.json \
  --report results/summaries/space_report_10min.md
```

`zone_editor.html`을 브라우저에서 열면 캡처 이미지 위에 polygon zone을 직접 그려 `zones.json`을 만들 수 있습니다.

## 개인정보 보호

- 얼굴 인식은 사용하지 않음
- 원본 영상 저장 없이 좌표/구역/체류시간 중심으로 분석 가능
- ID는 세션 내 임시 tracking ID
