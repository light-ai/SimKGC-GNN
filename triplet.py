# triplet.py

from collections import defaultdict
import json
from config import args
from SimKGC.logger_config import logger

class AllTripletDict:
    def __init__(self, path: str):
        self.triplet_dict = defaultdict(set)  # (head_id, relation) → set(tail_id)
        self._load(path)

    def _load(self, path: str):
        logger.info(f"Loading triplets from {path}")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for ex in data:
            key = (ex['head_id'], ex['relation'])
            self.triplet_dict[key].add(ex['tail_id'])

    def get_neighbors(self, head_id: str, relation: str):
        return self.triplet_dict.get((head_id, relation), set())

# グローバル変数として一度だけロード
_all_triplet_dict = None

def get_all_triplet_dict():
    global _all_triplet_dict
    if _all_triplet_dict is None:
        # train + valid + test 全部読み込むと安全（フィルタリング時に使うため）
        all_paths = [args.train_path, args.valid_path, args.test_path]
        _all_triplet_dict = AllTripletDict(path=all_paths[0])
        for p in all_paths[1:]:
            _all_triplet_dict._load(p)
    return _all_triplet_dict
