import os
import torch
from config import args
from dataset import load_data, Dataset, collate
from models import build_model
from evaluate import evaluate_model
from SimKGC.dict_hub import get_entity_dict, get_tokenizer
from SimKGC.logger_config import logger

from torch.utils.data import DataLoader

def main():
    # モデルパスを明示的に設定（必要に応じて変更）
    model_path = os.path.join(args.output_dir, "model_best.ckpt")

    assert os.path.exists(model_path), f"モデルが見つかりません: {model_path}"
    logger.info(f"モデルを読み込みます: {model_path}")

    # モデル構築と重みの読み込み
    model = build_model(args)
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    # テストデータの読み込み
    args.is_test = True  # 評価モードにする
    examples = load_data(args.test_path,
                         add_forward_triplet=True,
                         add_backward_triplet=True)

    test_dataset = Dataset(path=args.test_path, examples=examples)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             collate_fn=collate,
                             shuffle=False)

    # 評価実行
    logger.info("=== テストセットで評価を開始 ===")
    metrics = evaluate_model(model, test_loader)

    # 結果保存
    save_path = os.path.join(args.output_dir, "test_metrics.json")
    with open(save_path, "w", encoding="utf-8") as f:
        import json
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    logger.info(f"テスト結果を保存しました: {save_path}")


if __name__ == "__main__":
    main()
