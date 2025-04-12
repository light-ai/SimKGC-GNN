import json
import torch
from collections import defaultdict

class LinkGraph:
    def __init__(self, train_path: str):
        self.train_path = train_path
        self.edges = []  # (head_id, relation, tail_id)
        self._load()

    def _load(self):
        with open(self.train_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for ex in data:
            h, r, t = ex['head_id'], ex['relation'], ex['tail_id']
            self.edges.append((h, r, t))
            # inverse edge も追加（双方向 GCN のため）
            self.edges.append((t, r + '_inv', h))

        self.relation2id = self._build_relation2id()

    def _build_relation2id(self):
        relation_set = set([r for _, r, _ in self.edges])
        return {rel: idx for idx, rel in enumerate(sorted(relation_set))}

    def get_edge_index_and_type(self, entity_dict):
        src_list, dst_list, rel_list = [], [], []

        for h, r, t in self.edges:
            if h not in entity_dict.entity_to_idx or t not in entity_dict.entity_to_idx:
                continue
            src = entity_dict.entity_to_idx[h]
            dst = entity_dict.entity_to_idx[t]
            rel = self.relation2id[r]

            src_list.append(src)
            dst_list.append(dst)
            rel_list.append(rel)

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_type = torch.tensor(rel_list, dtype=torch.long)

        return edge_index, edge_type
