#!/bin/bash
set -e

echo "=========================================================="
echo "    BẮT ĐẦU QUÁ TRÌNH BUILD DOCKER SEALED & NUITKA       "
echo "=========================================================="
echo "Cảnh báo: Quá trình này có thể tốn 10-20 phút tuỳ vào Jetson của bạn."
echo "Đang dọn dẹp các container/image rác cũ..."

# Tắt container cũ nếu đang chạy
docker compose -f docker-compose.sealed.yml down || true

echo "Đang Build Docker Image (Stage 1: Nuitka, Stage 2: Sealed)..."
# Build image
docker compose -f docker-compose.sealed.yml build

echo "Build thành công! Đang khởi động Container Sealed..."
# Chạy container
docker compose -f docker-compose.sealed.yml up -d

echo "=========================================================="
echo "  HOÀN TẤT! HỆ THỐNG ĐÃ CHẠY Ở CHẾ ĐỘ MÃ HOÁ (.SO)        "
echo "=========================================================="
echo "Bạn có thể kiểm tra logs bằng lệnh:"
echo "  docker logs -f poc-camera-sealed"
