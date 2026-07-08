# YOLOv8n + ByteTrack-style Tracking 결과

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

ReID-style tracker 대신 ByteTrack-style tracker를 적용했다.

```text
YOLOv8n detection + ByteTrack-style two-stage matching + ID 표시
```

실행 파일:

```bash
/home/orangepi/rknn_model_zoo/install/rk3588_linux_aarch64/rknn_yolov8_demo/rknn_yolov8_bytetrack_live_demo
```

실행 명령:

```bash
cd /home/orangepi/rknn_model_zoo/install/rk3588_linux_aarch64/rknn_yolov8_demo
DISPLAY=:0 XAUTHORITY=/home/orangepi/.Xauthority \
./rknn_yolov8_bytetrack_live_demo model/yolov8n_runtime152.rknn \
rtsp://cmpark:cmpark12@192.168.10.111:554/stream1
```

현재 실행 PID:

```text
19016
```

중지 명령:

```bash
pkill -TERM -x rknn_yolov8_bytetrack_live_demo
```

## Tracking 방식

ByteTrack-style 방식은 detection을 두 그룹으로 나눠 처리한다.

- High-score detection: confidence `0.45` 이상
- Low-score detection: confidence `0.10 ~ 0.45`

처리 순서:

1. 기존 track과 high-score detection을 먼저 IoU 기준으로 매칭한다.
2. 매칭되지 않은 기존 track은 low-score detection과 한 번 더 매칭한다.
3. low-score detection은 기존 ID를 살리는 용도로만 사용한다.
4. 새 ID는 high-score detection에서만 만든다.
5. 잠깐 놓친 track은 `max_missed=30` 프레임까지 유지한다.

이 방식은 사람이 잠깐 흐려지거나 가려져 confidence가 낮아졌을 때 ID가 바로 끊기는 문제를 줄이는 데 유리하다.

## 화면 표시

화면에는 다음 정보가 표시된다.

```text
ID n ByteTrack person xx%
```

ID별 이동 trail도 함께 표시된다.

## 성능

최근 측정:

```text
bytetrack_perf frame=180 fps=14.11 high=4 low=0 tracks=4
bytetrack_perf frame=210 fps=14.79 high=3 low=1 tracks=5
bytetrack_perf frame=240 fps=15.99 high=2 low=1 tracks=4
bytetrack_perf frame=270 fps=16.59 high=3 low=0 tracks=3
bytetrack_perf frame=300 fps=17.21 high=3 low=0 tracks=3
bytetrack_perf frame=330 fps=16.44 high=3 low=0 tracks=3
bytetrack_perf frame=360 fps=15.62 high=3 low=0 tracks=3
bytetrack_perf frame=390 fps=14.70 high=2 low=0 tracks=3
```

요약:

- 약 `14 ~ 17 FPS`
- ReID 딥러닝 모델이 없으므로 연산 부담이 작다.
- Heatmap 버전보다 빠르고, ReID-style OpenCV embedding 버전과 비슷하거나 안정적인 실시간성을 보인다.

## 저장 파일

저장 폴더:

```text
G:\다른 컴퓨터\집PC\RDH202x\Work\PCM\2026\orange5plus\tapocamera\yolov8n_ByteTrack
```

주요 파일:

- `yolov8n_bytetrack_result.md`
- `yolov8n_bytetrack_display_capture.png`
- `yolov8n_bytetrack_perf_only.log`
- `yolov8n_bytetrack_live.log`
- `source\main_bytetrack_live_display.cc`
- `source\CMakeLists.txt`

## 다음 개선 포인트

- high/low confidence threshold 조정
- IoU threshold 조정
- max_missed 조정
- ROI 기반으로 불필요한 영역의 새 ID 생성 억제
- 필요 시 ByteTrack + ReID appearance를 결합
