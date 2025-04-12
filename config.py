# config.py

from argparse import Namespace
from transformers import AutoTokenizer

args = Namespace(
    # パス設定
    train_path="SimKGC/data/WN18RR/train.txt.json",
    valid_path="SimKGC/data/WN18RR/valid.txt.json",
    test_path="SimKGC/data/WN18RR/test.txt.json",
    entity_path="SimKGC/data/WN18RR/entities.json",
    relation_path="SimKGC/data/WN18RR/relations.json",

    # モデル・トークナイザ
    pretrained_model="bert-base-uncased",
    tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased"),

    # モデル構造設定
    max_num_tokens=128,
    pooling="cls",  # または "mean", "max"
    rgcn_hidden_dim=256,
    num_relations=11,  # WN18RR では 11個の関係がある

    # 学習設定
    batch_size=128,
    epochs=10,
    lr=2e-5,
    weight_decay=0.01,
    warmup=200,
    grad_clip=1.0,
    additive_margin=0.02,
    t=0.05,  # 温度パラメータ
    finetune_t=True,
    pre_batch=0,
    pre_batch_weight=1.0,
    use_self_negative=True,

    # その他
    output_dir="./output",
    log_steps=50,
    use_fp16=False,
    seed=42,

    # 評価・リンクグラフ
    use_link_graph=False,
    rerank_n_hop=2,
    neighbor_weight=0.05,
    is_test=False
)
