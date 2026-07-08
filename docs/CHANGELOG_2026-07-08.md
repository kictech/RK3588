# 2026-07-08 변경 이력

## 실행 환경

- Board: Orange Pi 5 Plus / RK3588
- Board IP: `192.168.10.175`
- Camera: TP-Link Tapo RTSP camera
- Camera IP: `192.168.10.111`
- RTSP: `rtsp://cmpark:cmpark12@192.168.10.111:554/stream1`
- Model: `yolov8n_runtime152.rknn`
- Pose model: `yolov8n-pose.rknn`

## 오늘 반영한 주요 코드

### ByteTrack live demo

파일:

```text
src/yolov8-detection-tracker/main_bytetrack_live_display.cc
```

주요 기능:

- YOLOv8n person detection
- ByteTrack-style two-stage matching
  - high-score detection 우선 매칭
  - low-score detection으로 미매칭 track 복구
- ID label 표시
- trail 표시
- static track suppression
- CSV track log 저장
- RTSP low-latency 설정 대응
- latest-frame reader thread
- 카메라 자체 반전 처리에 맞춰 앱 내부 180도 회전 제거

### 자동 시작 스크립트

파일:

```text
scripts/start_bytetrack_wait_camera.sh
```

주요 기능:

- 카메라 ping 확인
- 카메라가 준비되면 ByteTrack demo 자동 실행
- UDP low-latency 옵션 적용
- CSV 저장 경로 지정

### 공간 분석 도구

파일:

```text
space_analysis/zone_editor.html
space_analysis/aggregate_tracks.py
space_analysis/render_space_infographic.py
```

주요 기능:

- 이미지 위에서 polygon zone 편집
- track CSV를 zone별 체류시간/방문수/이동흐름으로 집계
- 집계 결과를 캡처 맵 위 인포그래픽 PNG로 렌더링

### YOLOv8n-pose RTSP demo

파일:

```text
src/yolov8-pose/main_camera.cc
```

주요 기능:

- Tapo RTSP 카메라 입력 지원
- OpenCV FFMPEG 입력 및 `CAP_PROP_BUFFERSIZE=1` 설정
- HDMI 모니터에 pose skeleton 실시간 표시
- 결과 화면 캡처와 NPU inference 로그 저장

## 성능 메모

ByteTrack + UDP low-latency + latest-frame reader 기준:

```text
약 14~15 FPS
카메라 입력 15 FPS에 근접
```

YOLOv8n-pose NPU inference 로그 기준:

```text
rknn_run time 약 21~27 ms
NPU inference 기준 약 36~46 FPS
실제 표시 속도는 RTSP 카메라 입력 FPS에 의해 제한
```

## 결과 파일

```text
results/screenshots/
results/summaries/
```

포함 항목:

- ByteTrack 결과 화면
- UDP low-latency 결과 화면
- static suppression 결과 화면
- 공간 분석 인포그래픽
- 10분 집계 JSON/Markdown

## 참고

- 현재 카메라에서 상하 반전을 처리하므로 C++ 코드에서는 프레임 회전을 수행하지 않습니다.
- 구역명은 `zones.json`에서 수정할 수 있습니다.
- 원본 영상 대신 CSV/JSON/인포그래픽 중심으로 보관합니다.
