#!/bin/bash
set -e

# このスクリプトのあるディレクトリを取得
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="${SCRIPT_DIR}/.."

cd "$ROOT_DIR"

# ログ出力先
LOG_DIR="output/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

echo "==== SimKGC + R-GCN の訓練を開始します ===="
echo "作業ディレクトリ: $ROOT_DIR"
echo "ログ出力先: $LOG_FILE"

# 実行
python train.py \
    --train-path "SimKGC/data/WN18RR/train.txt.json" \
    --valid-path "SimKGC/data/WN18RR/valid.txt.json" \
    --test-path "SimKGC/data/WN18RR/test.txt.json" \
    > "$LOG_FILE" 2>&1

echo "==== 訓練が完了しました ===="
