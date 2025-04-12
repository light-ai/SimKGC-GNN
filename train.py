import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

from SimKGC.config import args
from dataset import TripleDataset, collate
from models import build_model
from evaluate import evaluate_model
from SimKGC.logger_config import logger

def train():
    # ===== Load Data =====
    logger.info("Loading data...")
    train_dataset = TripleDataset(args.train_path)
    valid_dataset = TripleDataset(args.valid_path)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate, num_workers=4)

    # ===== Model =====
    model = build_model(args)
    model = model.cuda() if torch.cuda.is_available() else model
    logger.info("Model loaded")

    # ===== Optimizer & Scheduler =====
    # # 最も標準的で推奨される設定（これを使うことを推奨）
    # no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight', 'embedding']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #     'weight_decay': 0.01},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
    #     'weight_decay': 0.0},
    # ]

    # optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()

    best_mrr = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda() if torch.cuda.is_available() else v
            outputs = model(**batch)
            logit_output = model.compute_logits(outputs, batch)

            logits = logit_output['logits']
            labels = logit_output['labels']
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.info(f"[Epoch {epoch}] Loss: {avg_loss:.4f}")
        # train.pyの訓練ループ内に追加（オプション）
        logger.info(f"[Epoch {epoch}] alpha = {torch.sigmoid(model.alpha_raw).item():.4f}")


        # ===== Validation =====
        logger.info("Evaluating...")
        metrics = evaluate_model(model, valid_loader)
        logger.info(f"[Epoch {epoch}] Valid MRR: {metrics['mrr']:.4f}")

        # Save best model
        if metrics['mrr'] > best_mrr:
            best_mrr = metrics['mrr']
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            logger.info("Best model saved.")
        
        # epochごとに最新のモデルを保存
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'model_last.pt'))


    logger.info("Training finished.")

if __name__ == "__main__":
    train()
