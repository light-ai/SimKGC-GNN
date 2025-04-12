import argparse

parser = argparse.ArgumentParser()

# ===== Dataset =====
parser.add_argument('--task', type=str, default='WN18RR')
parser.add_argument('--train_path', type=str, default='simkgc/data/WN18RR/train.txt.json')
parser.add_argument('--valid_path', type=str, default='simkgc/data/WN18RR/valid.txt.json')
parser.add_argument('--test_path', type=str, default='simkgc/data/WN18RR/test.txt.json')
parser.add_argument('--use_link_graph', action='store_true')
parser.add_argument('--is_test', action='store_true')

# ===== Model =====
parser.add_argument('--model_name', type=str, default='bert_rgcn')
parser.add_argument('--pretrained_model', type=str, default='bert-base-uncased')
parser.add_argument('--hidden_dim', type=int, default=768)
parser.add_argument('--rgcn_hidden_dim', type=int, default=768)
parser.add_argument('--num_relations', type=int, default=18)  # WN18RR = 18 relation types

# ===== Training =====
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--grad_clip', type=float, default=1.0)
parser.add_argument('--warmup_steps', type=int, default=500)

# ===== Tokenization =====
parser.add_argument('--max_num_tokens', type=int, default=128)

# ===== Pre-batch / Self-negative settings =====
parser.add_argument('--pre_batch', type=int, default=0)
parser.add_argument('--pre_batch_weight', type=float, default=1.0)
parser.add_argument('--use_self_negative', action='store_true')
parser.add_argument('--additive_margin', type=float, default=0.0)
parser.add_argument('--t', type=float, default=0.05)
parser.add_argument('--finetune_t', action='store_true')

# ===== Output =====
parser.add_argument('--output_dir', type=str, default='output/checkpoints')
parser.add_argument('--log_dir', type=str, default='output/logs')

args = parser.parse_args()
