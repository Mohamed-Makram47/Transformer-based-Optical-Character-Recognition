"""
Arabic TrOCR Inference Script
==============================
Runs the trained model on a sample of the validation set and produces
a visual report: one image card per sample showing the input handwriting,
the ground truth, and the model's prediction side by side.

Usage:
    python inference.py \
        --model_path ./outputs/best_model \
        --num_samples 50 \
        --output_dir ./inference_results

Output:
    inference_results/
        cards/          ← one PNG per sample
        metrics.txt     ← overall CER / WER
"""

import os
import argparse
import logging
import textwrap
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from datasets import load_from_disk
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",   type=str, required=True,
                        help="Path to saved model dir (from trainer.save_model)")
    parser.add_argument("--data_path",    type=str,
                        default="/home/abdelrahman.shaheen/Transformer-based-Optical-Character-Recognition/src_hf/KHATT_v1_processed",
                        help="Path to the processed KHATT dataset on disk")
    parser.add_argument("--encoder_name", type=str,
                        default="microsoft/trocr-base-stage1",
                        help="Needed to load the image processor if not saved in model_path")
    parser.add_argument("--num_samples",  type=int, default=50,
                        help="How many validation samples to run")
    parser.add_argument("--output_dir",   type=str, default="./inference_results")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--num_beams",    type=int, default=4)
    parser.add_argument("--max_length",   type=int, default=64)
    return parser.parse_args()


# ──────────────────────────────────────────────
# Card rendering
# ──────────────────────────────────────────────
CARD_W      = 900
IMG_H       = 120      # height the input image is resized to for display
PAD         = 20
TEXT_H      = 90       # space for two lines of Arabic text below the image
CARD_H      = PAD + IMG_H + PAD + TEXT_H + PAD

BG          = (245, 245, 242)
CORRECT_COL = (34,  139, 34)    # green  – ground truth
PRED_COL    = (180, 20,  20)    # red    – prediction (wrong chars show up clearly)
LABEL_COL   = (100, 100, 100)
BORDER_OK   = (180, 220, 180)
BORDER_BAD  = (220, 180, 180)


def _get_font(size: int):
    """Try to load a system Arabic-capable font, fall back to default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def make_card(idx: int, image: Image.Image, gt: str, pred: str, cer: float) -> Image.Image:
    """Render a single result card as a PIL image."""
    card = Image.new("RGB", (CARD_W, CARD_H), BG)
    draw = ImageDraw.Draw(card)

    # ── border colour based on CER ──
    border_col = BORDER_OK if cer < 0.2 else BORDER_BAD
    draw.rectangle([0, 0, CARD_W - 1, CARD_H - 1], outline=border_col, width=3)

    # ── input image (resize to fixed height, keep aspect ratio) ──
    aspect = image.width / image.height
    disp_w = min(int(IMG_H * aspect), CARD_W - 2 * PAD)
    disp   = image.resize((disp_w, IMG_H), Image.LANCZOS).convert("RGB")
    card.paste(disp, (PAD, PAD))

    # ── CER badge ──
    badge_font = _get_font(14)
    badge_text = f"CER {cer:.3f}"
    badge_col  = CORRECT_COL if cer < 0.2 else PRED_COL
    draw.text((CARD_W - PAD - 90, PAD), badge_text, font=badge_font, fill=badge_col)

    # ── sample index ──
    draw.text((PAD, PAD + IMG_H + 6), f"#{idx}", font=_get_font(13), fill=LABEL_COL)

    # ── ground truth ──
    y_text = PAD + IMG_H + PAD
    lbl_font = _get_font(13)
    txt_font = _get_font(17)
    draw.text((PAD, y_text),      "GT   :",  font=lbl_font, fill=LABEL_COL)
    draw.text((PAD + 60, y_text), gt[:80],   font=txt_font, fill=CORRECT_COL)

    # ── prediction ──
    y_pred = y_text + 34
    draw.text((PAD, y_pred),      "PRED :",  font=lbl_font, fill=LABEL_COL)
    draw.text((PAD + 60, y_pred), pred[:80], font=txt_font, fill=PRED_COL)

    return card


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # ── output dirs ──
    out_dir   = Path(args.output_dir)
    cards_dir = out_dir / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)

    # ── load model ──
    logger.info(f"Loading model from {args.model_path} ...")
    model = VisionEncoderDecoderModel.from_pretrained(args.model_path).to(device)
    model.eval()

    # ── load processor and tokenizer ──
    # try model_path first (saved there by trainer), fall back to encoder_name
    try:
        processor = ViTImageProcessor.from_pretrained(args.model_path)
        logger.info("Loaded image processor from model_path")
    except Exception:
        processor = ViTImageProcessor.from_pretrained(args.encoder_name)
        logger.info(f"Loaded image processor from {args.encoder_name}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        logger.info("Loaded tokenizer from model_path")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")
        logger.info("Loaded tokenizer from asafaya/bert-base-arabic")

    # ── load dataset ──
    logger.info(f"Loading dataset from {args.data_path} ...")
    dataset   = load_from_disk(args.data_path)
    val_split = dataset["validation"]

    # random subset
    total      = len(val_split)
    n          = min(args.num_samples, total)
    indices    = torch.randperm(total)[:n].tolist()
    logger.info(f"Running inference on {n} / {total} validation samples ...")

    # ── metrics ──
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")

    results    = []
    all_preds  = []
    all_gts    = []

    for i, idx in enumerate(indices):
        example = val_split[idx]
        image   = example["image"].convert("RGB")
        gt      = example["text"]

        # preprocess
        pixel_values = processor(
            images=image, return_tensors="pt"
        ).pixel_values.to(device)

        # generate
        generated_ids = model.generate(
            pixel_values,
            max_length  = args.max_length,
            num_beams   = args.num_beams,
            early_stopping = True,
        )
        pred = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # per-sample CER
        sample_cer = cer_metric.compute(predictions=[pred], references=[gt])
        sample_wer = wer_metric.compute(predictions=[pred], references=[gt])

        all_preds.append(pred)
        all_gts.append(gt)

        results.append({
            "idx":  idx,
            "gt":   gt,
            "pred": pred,
            "cer":  round(sample_cer, 4),
            "wer":  round(sample_wer, 4),
        })

        # save card image
        card_path = cards_dir / f"sample_{idx:04d}.png"
        card      = make_card(idx, image, gt, pred, sample_cer)
        card.save(card_path)

        if (i + 1) % 10 == 0:
            logger.info(f"  {i+1}/{n} done ...")

    # ── overall metrics ──
    overall_cer = cer_metric.compute(predictions=all_preds, references=all_gts)
    overall_wer = wer_metric.compute(predictions=all_preds, references=all_gts)

    logger.info("=" * 40)
    logger.info(f"Overall CER : {overall_cer:.4f}")
    logger.info(f"Overall WER : {overall_wer:.4f}")
    logger.info("=" * 40)

    # ── write metrics.txt ──
    metrics_path = out_dir / "metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"Samples evaluated : {n}\n")
        f.write(f"Overall CER       : {overall_cer:.4f}\n")
        f.write(f"Overall WER       : {overall_wer:.4f}\n")
        f.write(f"\nPer-sample results:\n")
        f.write(f"{'idx':>6}  {'CER':>6}  {'WER':>6}  GT  →  PRED\n")
        f.write("-" * 80 + "\n")
        for r in sorted(results, key=lambda x: x["cer"], reverse=True):
            f.write(f"{r['idx']:>6}  {r['cer']:>6.4f}  {r['wer']:>6.4f}  "
                    f"{r['gt'][:30]}  →  {r['pred'][:30]}\n")


    logger.info(f"\nResults saved to {out_dir}/")
    logger.info(f"  cards/          ← {n} PNG cards")
    logger.info(f"  metrics.txt     ← CER / WER summary")


if __name__ == "__main__":
    main()