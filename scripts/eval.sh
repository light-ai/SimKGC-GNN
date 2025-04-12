#!/usr/bin/env bash

set -x
set -e

# 訓練済みモデルのパス（引数で指定可能）
model_path="output/model_best.pt"
if [[ $# -ge 1 ]]; then
    model_path=$1
fi

# タスク名 (WN18RRなど)
task="WN18RR"

# プロジェクトのルートディレクトリ（scriptsの一つ上）
DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "作業ディレクトリ: ${DIR}"

# データディレクトリ
DATA_DIR="${DIR}/data/${task}"

# テストデータパス
test_path="${DATA_DIR}/test.json"

# 評価スクリプトを実行
python -u eval.py \
    --task "${task}" \
    --eval-model-path "${model_path}" \
    --train-path "${DATA_DIR}/train.json" \
    --valid-path "${test_path}" \
    --batch-size 128 \
    "$@"
