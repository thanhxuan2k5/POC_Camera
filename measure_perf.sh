#!/bin/bash

echo "=========================================================="
echo "    ĐO HIỆU NĂNG HỆ THỐNG (JETSON ORIN NANO)             "
echo "=========================================================="

# Kiểm tra container đang chạy
if ! docker ps | grep -q "poc-camera-sealed"; then
    echo "LỖI: Container poc-camera-sealed chưa chạy!"
    echo "Hãy chạy lệnh: docker compose -f docker-compose.sealed.yml up -d"
    exit 1
fi

echo -e "\n[1] Thống kê RAM và CPU của Docker Container (Đang đo...):"
docker stats poc-camera-sealed --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"

echo -e "\n[2] Kiểm tra Nhịp tim (FPS) từ API:"
# Lấy FPS từ API
API_RESPONSE=$(curl -s http://localhost:8081/api/inference/stats || echo "{}")
FPS=$(echo $API_RESPONSE | grep -o '"fps":[0-9.]*' | cut -d':' -f2 || echo "Không lấy được")
IS_RUNNING=$(echo $API_RESPONSE | grep -o '"is_running":true' || echo "false")

if [ "$IS_RUNNING" == "false" ]; then
    echo "Trạng thái: PIPELINE ĐANG TẮT (Vui lòng bật Camera trên Web để đo FPS)"
else
    echo "Trạng thái: PIPELINE ĐANG CHẠY"
    echo "Tốc độ xử lý (FPS): $FPS khung hình / giây"
fi

echo -e "\n[3] Hiệu suất GPU (Chỉ dành cho Jetson):"
if command -v tegrastats &> /dev/null; then
    echo "Đang lấy mẫu tegrastats trong 2 giây..."
    timeout 2 tegrastats | head -n 1
    echo "(Lưu ý: Mở terminal mới và gõ 'tegrastats' để xem liên tục)"
else
    echo "Lệnh tegrastats không tồn tại trên hệ điều hành này."
fi

echo "=========================================================="
