# Orange Pi 5 Plus YOLOv8n Object Tracking 결과 정리

작성일: 2026-07-07

## 1. 구성

- 보드: Orange Pi 5 Plus / Rockchip RK3588
- 보드 IP: 192.168.10.175
- 입력 카메라: TP-Link Tapo RTSP 카메라
- 카메라 IP: 192.168.10.111
- RTSP 입력: `rtsp://cmpark:cmpark12@192.168.10.111:554/stream1`
- 입력 영상: 1920x1080, 약 15 FPS
- 출력 방식: Orange Pi HDMI 모니터에 OpenCV 창 표시
- 모델: `yolov8n_runtime152.rknn`
- RKNN runtime 계열: runtime 1.5.2 호환 모델

## 2. 진행 내용

1. Tapo 카메라 RTSP 입력을 Orange Pi에서 확인했다.
2. YOLOv8n RKNN 모델을 사용해 실시간 객체 검출 화면을 띄웠다.
3. YOLOv8n 단독 성능을 측정했다.
4. SORT-style IoU 기반 Object Tracking을 추가했다.
5. 사람 객체에 대해 `ID n person xx%` 형식의 ID 라벨을 표시했다.
6. 각 ID의 이동 trail을 화면에 함께 표시했다.
7. HDMI 디스플레이 화면을 PNG로 캡처해 PC 폴더에 저장했다.

## 3. 실행 파일

Orange Pi 실행 위치:

```bash
/home/orangepi/rknn_model_zoo/install/rk3588_linux_aarch64/rknn_yolov8_demo
```

YOLOv8n 단독 표시:

```bash
./rknn_yolov8_live_demo model/yolov8n_runtime152.rknn rtsp://cmpark:cmpark12@192.168.10.111:554/stream1
```

YOLOv8n + Object Tracking + ID 표시:

```bash
./rknn_yolov8_tracker_live_demo model/yolov8n_runtime152.rknn rtsp://cmpark:cmpark12@192.168.10.111:554/stream1
```

현재 실행 중인 tracking 프로세스:

```text
PID 10718
./rknn_yolov8_tracker_live_demo model/yolov8n_runtime152.rknn rtsp://cmpark:cmpark12@192.168.10.111:554/stream1
```

중지 명령:

```bash
pkill -TERM -x rknn_yolov8_tracker_live_demo
```

## 4. 성능 요약

YOLOv8n 단독:

- 실제 처리 FPS: 약 14.3 ~ 16.4 FPS
- 평균 처리 FPS: 약 15 FPS급
- YOLOv8n 추론 시간: 약 27 ~ 33 ms/frame
- 프레임 읽기부터 화면 표시까지 전체 파이프라인: 약 61 ~ 70 ms/frame

YOLOv8n + Object Tracking:

- 최근 측정 FPS: 약 13.6 ~ 17.4 FPS
- 일반적인 동작 범위: 약 15 ~ 16 FPS
- SORT-style IoU tracking은 가벼워서 YOLOv8n 단독 대비 성능 저하는 크지 않았다.
- 현재 병목은 모델 추론보다 RTSP 입력 15 FPS, RTSP 버퍼, OpenCV 디스플레이 쪽 영향이 더 커 보인다.

최근 tracking 로그 예:

```text
tracking_perf frame=132510 fps=16.88 detections=1 tracks=1
tracking_perf frame=132540 fps=16.61 detections=1 tracks=2
tracking_perf frame=132570 fps=17.42 detections=0 tracks=1
tracking_perf frame=132600 fps=16.90 detections=1 tracks=1
tracking_perf frame=132630 fps=16.72 detections=1 tracks=1
tracking_perf frame=132660 fps=16.70 detections=1 tracks=1
tracking_perf frame=132690 fps=14.70 detections=1 tracks=1
tracking_perf frame=132720 fps=15.67 detections=0 tracks=1
tracking_perf frame=132750 fps=14.58 detections=1 tracks=1
tracking_perf frame=132780 fps=14.60 detections=1 tracks=1
tracking_perf frame=132810 fps=15.38 detections=1 tracks=1
tracking_perf frame=132840 fps=14.57 detections=1 tracks=1
tracking_perf frame=132870 fps=15.03 detections=1 tracks=1
tracking_perf frame=132900 fps=15.83 detections=1 tracks=2
tracking_perf frame=132930 fps=15.02 detections=1 tracks=1
tracking_perf frame=132960 fps=13.62 detections=1 tracks=1
tracking_perf frame=132990 fps=14.95 detections=1 tracks=2
tracking_perf frame=133020 fps=15.92 detections=1 tracks=1
tracking_perf frame=133050 fps=14.39 detections=1 tracks=2
tracking_perf frame=133080 fps=14.57 detections=1 tracks=3
```

## 5. 결과 화면 캡처

저장 폴더:

```text
G:\다른 컴퓨터\집PC\RDH202x\Work\PCM\2026\orange5plus\tapocamera
```

저장된 주요 캡처:

- `yolov8n_display_capture.png`
- `yolov8n_tracking_display_capture.png`
- `yolov8n_tracking_display_capture_20260707_121737.png`

## 6. 생성/수정한 주요 소스

- `main_live_display.cc`
  - YOLOv8n 단독 카메라 입력 및 HDMI 표시
  - FPS / pipeline latency 로그 추가

- `main_tracker_live_display.cc`
  - YOLOv8n detection 결과를 사람 객체 중심으로 필터링
  - IoU 기반 lightweight SORT-style tracking
  - ID 라벨 및 이동 trail 표시
  - `tracking_perf` 로그 출력

- `main_tracker_csv.cc`
  - 기존 CSV 저장용 tracker 데모

- `CMakeLists.txt`
  - `rknn_yolov8_live_demo`
  - `rknn_yolov8_tracker_live_demo`
  - `rknn_yolov8_tracker_csv_demo`

## 7. 참고 사항

- 로그에 `RgaCollorFill fail: Invalid argument` 경고가 반복 출력되지만, 현재 YOLOv8n 추론 및 화면 표시는 계속 정상 동작한다.
- 현재 tracking은 DeepSORT처럼 외형 특징을 쓰는 방식이 아니라, 박스 IoU 기반의 가벼운 방식이다.
- 사람이 겹치거나 가려지는 상황에서는 ID가 바뀔 수 있다.
- 다음 단계인 heatmap은 각 ID의 foot point 좌표를 누적해서 화면 위에 컬러맵으로 표시하면 된다.

## 8. 다음 단계 제안

1. Heatmap overlay 추가
2. ID별 체류 시간 계산
3. 구역 ROI 설정
4. 구역별 방문 횟수 / 체류 시간 CSV 저장
5. RTSP low-latency 옵션 적용
6. 필요 시 ByteTrack 방식으로 tracker 개선
