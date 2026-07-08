# YOLOv8n + DeepSORT-style ReID Tracking 결과

작성일: 2026-07-07

## 구성

- 보드: Orange Pi 5 Plus / RK3588
- 보드 IP: `192.168.10.175`
- 카메라: TP-Link Tapo RTSP 카메라
- 카메라 IP: `192.168.10.111`
- 입력: `rtsp://cmpark:cmpark12@192.168.10.111:554/stream1`
- 모델: `yolov8n_runtime152.rknn`
- 출력: Orange Pi HDMI 모니터

## 이번 버전

Heatmap은 제거하고 아래 조합으로 실행했다.

```text
YOLOv8n detection + DeepSORT-style ReID tracking + ID 표시
```

실행 파일:

```bash
/home/orangepi/rknn_model_zoo/install/rk3588_linux_aarch64/rknn_yolov8_demo/rknn_yolov8_deepsort_live_demo
```

실행 명령:

```bash
cd /home/orangepi/rknn_model_zoo/install/rk3588_linux_aarch64/rknn_yolov8_demo
DISPLAY=:0 XAUTHORITY=/home/orangepi/.Xauthority \
./rknn_yolov8_deepsort_live_demo model/yolov8n_runtime152.rknn \
rtsp://cmpark:cmpark12@192.168.10.111:554/stream1
```

현재 실행 PID:

```text
17351
```

중지 명령:

```bash
pkill -TERM -x rknn_yolov8_deepsort_live_demo
```

## Tracking 방식

이전 SORT-style 버전은 박스 IoU 중심으로 ID를 유지했다.

이번 버전은 DeepSORT 구조처럼 다음 정보를 함께 사용한다.

- YOLOv8n person detection box
- 박스 위치 IoU
- 중심점 거리
- person crop에서 추출한 appearance/ReID embedding
- 이전 ID의 appearance embedding과 현재 detection embedding의 cosine similarity

보드 안에는 별도 학습된 ReID `.rknn` 모델이 없어서, 이번 구현은 OpenCV 기반의 가벼운 appearance embedding을 사용했다. 구조는 DeepSORT처럼 detection과 appearance/ReID 매칭을 분리해 두었으므로, 추후 OSNet/MobileNet ReID RKNN 모델을 구하면 `extract_reid_embedding()` 부분만 교체하면 된다.

## 화면 표시

화면에는 다음 정보가 표시된다.

```text
ID n ReID person xx%
```

또한 ID별 이동 trail도 함께 표시된다.

## 성능

최근 측정:

```text
deepsort_perf frame=180 fps=15.44 detections=2 tracks=4
deepsort_perf frame=210 fps=15.74 detections=2 tracks=3
deepsort_perf frame=240 fps=16.40 detections=2 tracks=2
deepsort_perf frame=270 fps=16.89 detections=2 tracks=3
deepsort_perf frame=300 fps=15.01 detections=2 tracks=3
deepsort_perf frame=330 fps=15.85 detections=2 tracks=3
deepsort_perf frame=360 fps=15.48 detections=1 tracks=2
deepsort_perf frame=390 fps=16.18 detections=1 tracks=2
```

요약:

- 약 `15 ~ 16.9 FPS`
- Heatmap 버전보다 빠름
- 기존 YOLOv8n + ID tracking과 거의 비슷한 실시간성 유지

## 저장 파일

저장 폴더:

```text
G:\다른 컴퓨터\집PC\RDH202x\Work\PCM\2026\orange5plus\tapocamera\yolov8n_deepsort
```

주요 파일:

- `yolov8n_deepsort_result.md`
- `yolov8n_deepsort_display_capture.png`
- `yolov8n_deepsort_perf_only.log`
- `yolov8n_deepsort_live.log`
- `source\main_deepsort_live_display.cc`
- `source\CMakeLists.txt`

## 한계와 다음 단계

- 현재 ReID embedding은 학습된 딥러닝 ReID 모델이 아니라 가벼운 appearance descriptor다.
- 조명 변화, 비슷한 옷차림, 심한 occlusion에서는 ID가 바뀔 수 있다.
- 진짜 DeepSORT에 더 가깝게 가려면 OSNet/MobileNet 기반 ReID 모델을 ONNX 또는 RKNN으로 준비한 뒤 NPU/CPU inference를 붙이면 된다.
