#!/bin/bash
# 训练启动脚本

set -e

CONFIG=${1:-configs/default.yaml}
DEVICE=${2:-cuda}

echo "=== 训练哈尔滨文旅路线模型 ==="
echo "配置文件: $CONFIG"
echo "设备: $DEVICE"

python -m src.train \
    --config "$CONFIG" \
    --device "$DEVICE"

echo "=== 训练完成 ==="
