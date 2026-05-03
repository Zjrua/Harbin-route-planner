#!/bin/bash
# 消融实验批量运行脚本

set -e

echo "=== 消融实验 ==="

EXPERIMENTS=(
    "ablation_no_engram"
    "ablation_no_mhc"
    "ablation_adamw"
    "ablation_baseline"
    "ablation_engram_k3"
    "ablation_engram_k10"
    "ablation_mhc_curvature_05"
    "ablation_mhc_curvature_2"
)

for exp in "${EXPERIMENTS[@]}"; do
    echo "--- 运行实验: $exp ---"
    python -m src.train \
        --config configs/ablation.yaml \
        --experiment "$exp" \
        --device cuda
    echo "--- 完成: $exp ---"
    echo ""
done

echo "=== 全部消融实验完成 ==="
echo "结果保存在 checkpoints/ 目录下"
