# cv2-web-sample

Sample tổng hợp kết nối Jetson Web (REST + MQTT) với YOLO detection/tracking từ `cv2_base`.

## Kiến trúc (4 thread per camera)

| Thread | Vai trò |
|--------|---------|
| `ThreadCapture` | Đọc RTSP từ Web API |
| `ThreadDetection` | YOLO + BYTETracker, lọc polygon |
| `ThreadMetadata` | Publish bbox overlay qua MQTT `camera/{id}/metadata` |
| `ThreadEvent` | Fire event `POST /events` (+ upload snapshot) |

## Chạy local

```bash
cd cv2-web-sample
cp main_config.example.yaml main_config.yaml
# Chỉnh web_api.base_url, mqtt.broker

pip install -r requirements.txt
python main.py
```

## Chạy Docker

```bash
cp main_config.example.yaml main_config.yaml
# Chỉnh config

docker compose up --build
```

`network_mode: host` để RTSP/MQTT truy cập localhost Jetson.

## Verify

1. Log: Web API login + cameras loaded + MQTT connected
2. Frontend live view hiển thị bbox thật
3. `mosquitto_sub -h 127.0.0.1 -t "camera/+/metadata" -v`
4. Khi có object trong zone → event được tạo trên backend

Dừng: `Ctrl+C` hoặc `docker compose down`

## Event toggle for sample

- `general.enable_event: false` (default): chỉ publish bbox metadata, không tạo startup/runtime events.
- `general.enable_event: true`: bật lại `POST /events` và upload snapshot.
