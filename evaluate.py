import torch
from tqdm import tqdm
from SimKGC.logger_config import logger
from dataset import entity_dict
from triplet import get_all_triplet_dict

all_triplet_dict = get_all_triplet_dict()

@torch.no_grad()
def evaluate_model(model, dataloader, k=10):
    model.eval()
    device = next(model.parameters()).device

    hr_tensor_list = []
    tail_indices_list = []
    examples = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        for k_, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k_] = v.to(device)

        outputs = model(**batch)
        hr_tensor = outputs["hr_vector"]  # (B, dim)
        hr_tensor_list.append(hr_tensor)
        tail_indices_list.extend(batch["tail_indices"].tolist())
        examples.extend(batch["batch_data"])

    hr_tensor = torch.cat(hr_tensor_list, dim=0)
    entity_tensor = model.predict_ent_embedding(
        tail_token_ids=None, tail_mask=None, tail_token_type_ids=None,
        all=True  # 自作モデル用に全エンティティ出力
    )["ent_vectors"].to(device)

    return compute_metrics(hr_tensor, entity_tensor, tail_indices_list, examples, k)


def compute_metrics(hr_tensor, entity_tensor, target, examples, k=10):
    total = hr_tensor.size(0)
    scores = torch.mm(hr_tensor, entity_tensor.t())  # (batch, entities)

    mean_rank, mrr, hit1, hit3, hit10 = 0, 0, 0, 0, 0

    for i in range(total):
        cur_score = scores[i]
        cur_target = target[i]

        # 既知の triple を除外
        mask_ids = all_triplet_dict.get_neighbors(examples[i].head_id, examples[i].relation)
        mask_ids = [entity_dict.entity_to_idx(eid) for eid in mask_ids if eid != examples[i].tail_id]
        if mask_ids:
            cur_score[mask_ids] = -1e4

        sorted_idx = torch.argsort(cur_score, descending=True)
        rank = (sorted_idx == cur_target).nonzero(as_tuple=False).item() + 1

        mean_rank += rank
        mrr += 1.0 / rank
        hit1 += 1 if rank <= 1 else 0
        hit3 += 1 if rank <= 3 else 0
        hit10 += 1 if rank <= 10 else 0

    metrics = {
        "mean_rank": round(mean_rank / total, 4),
        "mrr": round(mrr / total, 4),
        "hit@1": round(hit1 / total, 4),
        "hit@3": round(hit3 / total, 4),
        "hit@10": round(hit10 / total, 4),
    }

    logger.info("Evaluation metrics:")
    for k_, v in metrics.items():
        logger.info(f"{k_}: {v:.4f}")
    return metrics
