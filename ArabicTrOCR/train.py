"""
Arabic TrOCR Training Script
=============================
Trains the VisionEncoderDecoder (ViT + Arabic BERT) on the KHATT dataset.

Usage:
    python train.py --data_dir /path/to/khatt --output_dir ./outputs
"""

import os
import argparse
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
from datasets import load_dataset

import torch
import numpy as np
from PIL import Image
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    ViTImageProcessor,
    AutoTokenizer,
    EarlyStoppingCallback,
    default_data_collator,
)
from datasets import load_from_disk, DatasetDict
import evaluate

from ArabicTrOCR.model import build_arabic_trocr

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Argument Parsing
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Train Arabic TrOCR on KHATT")

    # paths
    parser.add_argument("--output_dir",       type=str, default="./outputs")
    parser.add_argument("--resume_from",      type=str, default=None,   help="Path to checkpoint to resume from")

    # model
    parser.add_argument("--encoder_name",     type=str, default="shaheen6/arapixel_pretraining_sixteenth")
    parser.add_argument("--decoder_name",     type=str, default="asafaya/bert-base-arabic")
    parser.add_argument("--freeze_encoder",   action="store_true",      help="Freeze ViT encoder weights")

    # training
    parser.add_argument("--epochs",           type=int,   default=20)
    parser.add_argument("--train_batch_size", type=int,   default=32)
    parser.add_argument("--eval_batch_size",  type=int,   default=32)
    parser.add_argument("--lr",               type=float, default=5e-5)
    parser.add_argument("--warmup_steps",     type=int,   default=500)
    parser.add_argument("--weight_decay",     type=float, default=0.01)
    parser.add_argument("--max_target_length",type=int,   default=64)
    parser.add_argument("--fp16",             action="store_true",      help="Use mixed precision training")
    parser.add_argument("--grad_accum_steps", type=int,   default=1,    help="Gradient accumulation steps")
    parser.add_argument("--early_stopping_patience", type=int, default=3)

    return parser.parse_args()


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class KHATTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, feature_extractor, tokenizer, max_target_length=64):
        self.dataset           = dataset
        self.feature_extractor = feature_extractor
        self.tokenizer         = tokenizer
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        # image is already a PIL object, no need to open from path
        pixel_values = self.feature_extractor(
            example["image"].convert("RGB"), return_tensors="pt"
        ).pixel_values.squeeze(0)

        labels = self.tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        labels[labels == self.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}

        

# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────
def build_compute_metrics(tokenizer):
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids   = pred.predictions
        label_ids  = pred.label_ids

        # replace -100 back to pad token so decode doesn't crash
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"cer": round(cer, 4), "wer": round(wer, 4)}

    return compute_metrics


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    args = parse_args()

    # ── Device ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Model ──
    logger.info("Building model...")
    model, feature_extractor, tokenizer = build_arabic_trocr(
        encoder_name   = args.encoder_name,
        decoder_name   = args.decoder_name,
        freeze_encoder = args.freeze_encoder,
    )
    model.to(device)

    # ── Dataset ──
    logger.info(f"Loading dataset from  KHATT_v1.0_dataset...")
    raw_dataset = load_from_disk("/home/abdelrahman.shaheen/Transformer-based-Optical-Character-Recognition/src_hf/data/KHATT_v1_processed")
    train_dataset = KHATTDataset(raw_dataset["train"],      feature_extractor, tokenizer, args.max_target_length)
    eval_dataset  = KHATTDataset(raw_dataset["validation"], feature_extractor, tokenizer, args.max_target_length)

    logger.info(f"Train examples : {len(train_dataset):,}")
    logger.info(f"Eval  examples : {len(eval_dataset):,}")

    # ── Training Arguments ──
    training_args = Seq2SeqTrainingArguments(
        output_dir                  = args.output_dir,

        # evaluation
        evaluation_strategy         = "epoch",
        predict_with_generate       = True,          # required for CER/WER metrics
        generation_max_length       = args.max_target_length,

        # batch size & gradient accumulation
        per_device_train_batch_size = args.train_batch_size,
        per_device_eval_batch_size  = args.eval_batch_size,
        gradient_accumulation_steps = args.grad_accum_steps,

        # optimiser
        learning_rate               = args.lr,
        warmup_steps                = args.warmup_steps,
        weight_decay                = args.weight_decay,
        lr_scheduler_type           = "cosine",      # cosine decay is better than linear for OCR

        # training length
        num_train_epochs            = args.epochs,

        # mixed precision
        fp16                        = args.fp16 and device == "cuda",

        # checkpointing
        save_strategy               = "epoch",
        save_total_limit            = 3,             # keep only the 3 best checkpoints
        load_best_model_at_end      = True,
        metric_for_best_model       = "cer",         # lower CER = better
        greater_is_better           = False,

        # logging
        logging_dir                 = os.path.join(args.output_dir, "logs"),
        logging_steps               = 10,
        report_to                   = "wandb",
    )

    # ── Trainer ──
    trainer = Seq2SeqTrainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = eval_dataset,
        tokenizer       = tokenizer,
        data_collator   = default_data_collator,
        compute_metrics = build_compute_metrics(tokenizer),
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    # ── Train ──
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from)

    # ── Save final model ──
    best_model_path = os.path.join(args.output_dir, "best_model")
    trainer.save_model(best_model_path)
    feature_extractor.save_pretrained(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    logger.info(f"Best model saved to {best_model_path}")

    # ── Final evaluation ──
    logger.info("Running final evaluation on validation set...")
    metrics = trainer.evaluate()
    logger.info(f"Final CER : {metrics['eval_cer']:.4f}")
    logger.info(f"Final WER : {metrics['eval_wer']:.4f}")


if __name__ == "__main__":
    main()