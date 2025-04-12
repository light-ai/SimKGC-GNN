import os
import json


def load_definitions(def_path):
    """wordnet-mlj12-definitions.txt を読み込む"""
    id2desc = {}
    with open(def_path, encoding='utf-8') as f:
        for line in f:
            fs = line.strip().split('\t')
            if len(fs) == 3:
                eid, _, desc = fs
                id2desc[eid] = desc
    print(f"Loaded {len(id2desc)} entity definitions.")
    return id2desc


def convert_triplets_to_json(txt_path, desc_dict, out_path):
    """train/valid/test.txt → train/valid/test.txt.json に変換"""
    examples = []
    with open(txt_path, encoding='utf-8') as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            examples.append({
                "head_id": h,
                "head": h,
                "relation": r.replace('_', ' '),  # SimKGCスタイルに変換
                "tail_id": t,
                "tail": t,
                "head_desc": desc_dict.get(h, ""),
                "tail_desc": desc_dict.get(t, "")
            })
    with open(out_path, 'w', encoding='utf-8') as wf:
        json.dump(examples, wf, ensure_ascii=False, indent=2)
    print(f"Saved {len(examples)} examples to {out_path}")


def convert_entities_dict_to_json(dict_path, desc_dict, out_path):
    """entities.dict → entities.json に変換"""
    with open(dict_path, encoding='utf-8') as f:
        entities = [line.strip().split('\t')[0] for line in f]
    data = [{
        "entity_id": eid,
        "entity": eid,
        "entity_desc": desc_dict.get(eid, "")
    } for eid in entities]
    with open(out_path, 'w', encoding='utf-8') as wf:
        json.dump(data, wf, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} entities to {out_path}")


def main():
    data_dir = "./SimKGC/data/WN18RR"
    def_path = os.path.join(data_dir, "wordnet-mlj12-definitions.txt")

    desc_dict = load_definitions(def_path)

    for split in ['train', 'valid', 'test']:
        txt_path = os.path.join(data_dir, f"{split}.txt")
        out_path = txt_path + ".json"
        convert_triplets_to_json(txt_path, desc_dict, out_path)

    ent_dict_path = os.path.join(data_dir, "entities.dict")
    out_ent_path = os.path.join(data_dir, "entities.json")
    convert_entities_dict_to_json(ent_dict_path, desc_dict, out_ent_path)


if __name__ == "__main__":
    main()
