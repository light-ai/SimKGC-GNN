import os
import json
import torch
from torch.utils.data import Dataset
from typing import List

from SimKGC.config import args
from SimKGC.dict_hub import get_entity_dict, get_tokenizer
from SimKGC.triplet import reverse_triplet
from SimKGC.triplet_mask import construct_mask, construct_self_negative_mask
from SimKGC.logger_config import logger


entity_dict = get_entity_dict()
tokenizer = get_tokenizer()


class Example:
    def __init__(self, head_id, relation, tail_id, head="", tail="", head_desc="", tail_desc=""):
        self.head_id = head_id
        self.relation = relation
        self.tail_id = tail_id
        self.head = head or head_id
        self.tail = tail or tail_id
        self.head_desc = head_desc
        self.tail_desc = tail_desc

    def vectorize(self):
        head_text = f"{self.head}: {self.head_desc}" if self.head_desc else self.head
        tail_text = f"{self.tail}: {self.tail_desc}" if self.tail_desc else self.tail

        hr_enc = tokenizer(text=head_text, text_pair=self.relation, max_length=args.max_num_tokens,
                           padding='max_length', truncation=True, return_token_type_ids=True)
        tail_enc = tokenizer(text=tail_text, max_length=args.max_num_tokens,
                             padding='max_length', truncation=True, return_token_type_ids=True)
        head_enc = tokenizer(text=head_text, max_length=args.max_num_tokens,
                             padding='max_length', truncation=True, return_token_type_ids=True)

        return {
            'hr_token_ids': hr_enc['input_ids'],
            'hr_token_type_ids': hr_enc['token_type_ids'],
            'tail_token_ids': tail_enc['input_ids'],
            'tail_token_type_ids': tail_enc['token_type_ids'],
            'head_token_ids': head_enc['input_ids'],
            'head_token_type_ids': head_enc['token_type_ids'],
            'head_indices': entity_dict.entity_to_idx(self.head_id),
            'tail_indices': entity_dict.entity_to_idx(self.tail_id),
            'obj': self
        }


def load_data(path: str,
              add_forward_triplet: bool = True,
              add_backward_triplet: bool = True) -> List[Example]:
    assert path.endswith('.json'), f"Expected .json file, got: {path}"
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} triples from {path}")
    examples = []
    for obj in data:
        if add_forward_triplet:
            examples.append(Example(**obj))
        if add_backward_triplet:
            examples.append(Example(**reverse_triplet(obj)))
    return examples


class TripleDataset(Dataset):
    def __init__(self, path: str, examples=None):
        self.path = path
        self.examples = examples or load_data(path)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx].vectorize()


def to_indices_and_mask(batch_tensor_list: List[torch.LongTensor], pad_token_id=0, need_mask=True):
    max_len = max(t.size(0) for t in batch_tensor_list)
    batch_size = len(batch_tensor_list)
    indices = torch.LongTensor(batch_size, max_len).fill_(pad_token_id)
    mask = torch.BoolTensor(batch_size, max_len).fill_(0) if need_mask else None
    for i, t in enumerate(batch_tensor_list):
        indices[i, :len(t)].copy_(t)
        if need_mask:
            mask[i, :len(t)].fill_(1)
    return (indices, mask) if need_mask else indices


def collate(batch_data: List[dict]) -> dict:
    def to_tensor(field):
        return [torch.LongTensor(ex[field]) for ex in batch_data]

    hr_token_ids, hr_mask = to_indices_and_mask(to_tensor('hr_token_ids'), pad_token_id=tokenizer.pad_token_id)
    hr_token_type_ids = to_indices_and_mask(to_tensor('hr_token_type_ids'), need_mask=False)

    tail_token_ids, tail_mask = to_indices_and_mask(to_tensor('tail_token_ids'), pad_token_id=tokenizer.pad_token_id)
    tail_token_type_ids = to_indices_and_mask(to_tensor('tail_token_type_ids'), need_mask=False)

    head_token_ids, head_mask = to_indices_and_mask(to_tensor('head_token_ids'), pad_token_id=tokenizer.pad_token_id)
    head_token_type_ids = to_indices_and_mask(to_tensor('head_token_type_ids'), need_mask=False)

    batch_exs = [ex['obj'] for ex in batch_data]
    head_indices = torch.LongTensor([ex['head_indices'] for ex in batch_data])
    tail_indices = torch.LongTensor([ex['tail_indices'] for ex in batch_data])

    return {
        'hr_token_ids': hr_token_ids,
        'hr_mask': hr_mask,
        'hr_token_type_ids': hr_token_type_ids,
        'tail_token_ids': tail_token_ids,
        'tail_mask': tail_mask,
        'tail_token_type_ids': tail_token_type_ids,
        'head_token_ids': head_token_ids,
        'head_mask': head_mask,
        'head_token_type_ids': head_token_type_ids,
        'batch_data': batch_exs,
        'triplet_mask': construct_mask(batch_exs) if not args.is_test else None,
        'self_negative_mask': construct_self_negative_mask(batch_exs) if not args.is_test else None,
        'head_indices': head_indices,
        'tail_indices': tail_indices,
    }
