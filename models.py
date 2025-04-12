import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv
from SimKGC.models import CustomBertModel

class CustomBertModelWithRGCN(CustomBertModel):
    def __init__(self, args, entity_embeddings: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor):
        super().__init__(args)
        
        self.rgcn1 = RGCNConv(entity_embeddings.size(1), args.rgcn_hidden_dim, args.num_relations)
        self.rgcn2 = RGCNConv(args.rgcn_hidden_dim, args.rgcn_hidden_dim, args.num_relations)

        self.entity_embedding = nn.Parameter(entity_embeddings, requires_grad=True)
        self.edge_index = edge_index
        self.edge_type = edge_type
        
        # 重み付き和の係数を学習可能パラメータとして導入
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))

    def forward(self, hr_token_ids, hr_mask, hr_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids,
                head_indices, tail_indices,
                only_ent_embedding=False, **kwargs):

        # GNNベクトルを計算
        all_entity_vecs = self.rgcn1(self.entity_embedding, self.edge_index, self.edge_type)
        all_entity_vecs = torch.relu(all_entity_vecs)
        all_entity_vecs = self.rgcn2(all_entity_vecs, self.edge_index, self.edge_type)

        # 重み係数 (0~1の範囲に制限)
        alpha = torch.sigmoid(self.alpha_raw)

        # 推論専用モード (entity embedding のみ)
        if only_ent_embedding:
            tail_vec_bert = self._encode(self.tail_bert,
                                         token_ids=tail_token_ids,
                                         mask=tail_mask,
                                         token_type_ids=tail_token_type_ids)
            tail_vec_gnn = all_entity_vecs[tail_indices]
            tail_vec = (1 - alpha) * tail_vec_bert + alpha * tail_vec_gnn
            return {'ent_vectors': tail_vec.detach()}

        # 通常のBERTエンコーディング
        hr_vector_raw = self._encode(self.hr_bert, hr_token_ids, hr_mask, hr_token_type_ids)
        tail_vector_bert = self._encode(self.tail_bert, tail_token_ids, tail_mask, tail_token_type_ids)
        head_vector = self._encode(self.tail_bert, head_token_ids, head_mask, head_token_type_ids)

        # GNN構造ベクトル
        head_gnn_vecs = all_entity_vecs[head_indices]
        tail_gnn_vecs = all_entity_vecs[tail_indices]

        # BERTとGNNを重み付き和で融合
        hr_vector = (1 - alpha) * hr_vector_raw + alpha * head_gnn_vecs
        tail_vector = (1 - alpha) * tail_vector_bert + alpha * tail_gnn_vecs

        return {
            'hr_vector': hr_vector,
            'tail_vector': tail_vector,
            'head_vector': head_vector
        }
    
    @torch.no_grad()
    def predict_ent_embedding(self, tail_token_ids, tail_mask, tail_token_type_ids, tail_indices, **kwargs) -> dict:
        # GNN構造ベクトルを計算
        all_entity_vecs = self.rgcn1(self.entity_embedding, self.edge_index, self.edge_type)
        all_entity_vecs = torch.relu(all_entity_vecs)
        all_entity_vecs = self.rgcn2(all_entity_vecs, self.edge_index, self.edge_type)

        # GNNベクトル
        tail_vec_gnn = all_entity_vecs[tail_indices]

        # BERT意味ベクトル
        tail_vec_bert = self._encode(self.tail_bert,
                                     token_ids=tail_token_ids,
                                     mask=tail_mask,
                                     token_type_ids=tail_token_type_ids)

        # 重み付き融合
        alpha = torch.sigmoid(self.alpha_raw)
        ent_vectors = (1 - alpha) * tail_vec_bert + alpha * tail_vec_gnn

        return {'ent_vectors': ent_vectors.detach()}
