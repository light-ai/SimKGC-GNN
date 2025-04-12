import os
import json
import torch
from torch.utils.data import DataLoader

from config import args
from dataset import load_data, TripleDataset, collate
from models import build_model
from evaluate import evaluate_model
from SimKGC.dict_hub import get_entity_dict
from SimKGC.logger_config import logger
from link_graph import LinkGraph  # R-GCNグラフ構造構築用

def main():
    logger.info("=== SimKGC + R-GCN の評価を開始します ===")

    # モデルパス
    model_path = os.path.join(args.output_dir, "model_best.pt")
    assert os.path.exists(model_path), f"モデルが見つかりません: {model_path}"
    logger.info(f"モデルを読み込みます: {model_path}")

    # エンティティ辞書とグラフ構築
    entity_dict = get_entity_dict()
    link_graph = LinkGraph(args.train_path)
    edge_index, edge_type = link_graph.get_edge_index_and_type(entity_dict)

    # ランダム初期のエンティティ埋め込み（GNN入力用）
    entity_dim = args.rgcn_hidden_dim
    num_entities = len(entity_dict)
    entity_embeddings = torch.randn(num_entities, entity_dim)

    # モデル構築とロード
    model = build_model(args, entity_embeddings, edge_index, edge_type)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    # 評価データの読み込み
    args.is_test = True
    examples = load_data(args.valid_path,
                         add_forward_triplet=True,
                         add_backward_triplet=True)
    test_dataset = TripleDataset(path=args.valid_path, examples=examples)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             collate_fn=collate,
                             shuffle=False)

    # 全エンティティベクトルを GNN + BERT で取得
    logger.info("全エンティティのベクトルを取得中...")
    ent_dataset = TripleDataset(path=args.valid_path, examples=examples)
    ent_loader = DataLoader(ent_dataset,
                            batch_size=args.batch_size,
                            collate_fn=collate,
                            shuffle=False)

    ent_vecs = []
    for batch in ent_loader:
        batch['only_ent_embedding'] = True
        if torch.cuda.is_available():
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = model(**batch)
        ent_vecs.append(out['ent_vectors'])

    entity_tensor = torch.cat(ent_vecs, dim=0)

    # モデル評価
    logger.info("評価を開始します...")
    metrics = evaluate_model(model, test_loader, entity_tensor=entity_tensor)

    # 結果保存
    result_path = os.path.join(args.output_dir, "test_metrics.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    logger.info(f"評価完了。結果は {result_path} に保存されました。")

if __name__ == "__main__":
    main()
