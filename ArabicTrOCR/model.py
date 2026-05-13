"""
Arabic TrOCR Architecture
=========================
Fuses a ViT vision encoder (from microsoft/trocr-base-stage1)
with an Arabic BERT decoder (asafaya/bert-base-arabic) via cross-attention.

Architecture:
  [Image] --> [ViT Encoder] --> [encoder_hidden_states]
                                        |
                                        v
  [Text tokens] --> [Arabic BERT Decoder w/ Cross-Attention] --> [logits]
"""

import torch
import torch.nn as nn
from transformers import (
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
    AutoTokenizer,
    AutoProcessor,
    BertConfig,
    BertLMHeadModel,
    ViTModel,
    ViTConfig,
    ViTImageProcessor
)


from PIL import Image
import requests


# ──────────────────────────────────────────────
# 1.  Build the model
# ──────────────────────────────────────────────

def build_arabic_trocr(
    encoder_name: str = "microsoft/trocr-base-stage1",
    decoder_name: str = "asafaya/bert-base-arabic",
    freeze_encoder: bool = False,
) -> tuple:
    """
    Builds a VisionEncoderDecoderModel that fuses:
      - ViT encoder  (from trocr-base-stage1)
      - Arabic BERT decoder with cross-attention injected

    Returns:
        model      : VisionEncoderDecoderModel
        processor  : AutoProcessor  (for image preprocessing)
        tokenizer  : AutoTokenizer  (for Arabic text)
    """

    print(f"[1/4] Loading ViT encoder from '{encoder_name}' ...")
    # trocr-base-stage1 is itself a VisionEncoderDecoder; we extract the ViT
    trocr_full = VisionEncoderDecoderModel.from_pretrained(encoder_name)
    encoder = trocr_full.encoder          # pure ViT, outputs BaseModelOutput
    del trocr_full                         # free the rest

    print(f"[2/4] Loading Arabic BERT decoder from '{decoder_name}' ...")
    # BERT must be reconfigured as a *decoder* so that:
    #   • each self-attention layer becomes causal (masked)
    #   • cross-attention layers are added to attend over encoder outputs
    decoder_config = BertConfig.from_pretrained(decoder_name)
    decoder_config.is_decoder = True           # enables causal self-attention mask
    decoder_config.add_cross_attention = True  # inserts cross-attention sublayers

    decoder = BertLMHeadModel.from_pretrained(
        decoder_name,
        config=decoder_config,
        ignore_mismatched_sizes=True,   # cross-attn weights are new → random init
    )

    print("[3/4] Assembling VisionEncoderDecoderModel ...")
    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)

    # ── token-id configuration ──
    tokenizer = AutoTokenizer.from_pretrained(decoder_name)

    model.config.decoder_start_token_id = tokenizer.cls_token_id  # [CLS] to start
    model.config.pad_token_id           = tokenizer.pad_token_id
    model.config.eos_token_id           = tokenizer.sep_token_id  # [SEP] to stop
    model.config.vocab_size             = decoder_config.vocab_size

    # ── generation hyper-parameters ──
    model.config.max_length             = 64
    model.config.early_stopping         = True
    model.config.no_repeat_ngram_size   = 3
    model.config.length_penalty         = 2.0
    model.config.num_beams              = 4

    # ── optional: freeze the vision encoder ──
    if freeze_encoder:
        print("   Freezing encoder weights.")
        for param in model.encoder.parameters():
            param.requires_grad = False

    # ── processor (handles image resizing / normalisation) ──
    processor = ViTImageProcessor.from_pretrained(encoder_name)


    print("[4/4] Done.\n")
    return model, processor, tokenizer


# ──────────────────────────────────────────────
# 2.  Parameter summary
# ──────────────────────────────────────────────

def parameter_summary(model: VisionEncoderDecoderModel) -> None:
    def fmt(n): return f"{n:,}"

    enc_total  = sum(p.numel() for p in model.encoder.parameters())
    enc_train  = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    dec_total  = sum(p.numel() for p in model.decoder.parameters())
    dec_train  = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)

    # isolate cross-attention layers specifically
    cross_attn_params = sum(
        p.numel()
        for name, p in model.decoder.named_parameters()
        if "crossattention" in name
    )

    total      = enc_total + dec_total
    trainable  = enc_train + dec_train

    print("=" * 45)
    print("  Model Architecture Summary")
    print("=" * 45)
    print(f"  Encoder  (ViT)          : {fmt(enc_total):>15}  params  ({fmt(enc_train)} trainable)")
    print(f"  Decoder  (Arabic BERT)  : {fmt(dec_total):>15}  params  ({fmt(dec_train)} trainable)")
    print(f"    └─ Cross-Attention    : {fmt(cross_attn_params):>15}  params  (always trainable)")
    print("-" * 45)
    print(f"  Total                   : {fmt(total):>15}  params")
    print(f"  Trainable               : {fmt(trainable):>15}  params")
    print("=" * 45)


# ──────────────────────────────────────────────
# 3.  Single-image inference helper
# ──────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, processor, tokenizer, image: Image.Image, device: str) -> str:
    """Run OCR on a single PIL image and return the decoded Arabic string."""
    model.eval()
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    generated_ids = model.generate(
        pixel_values,
        max_length=model.config.max_length,
        num_beams=model.config.num_beams,
        early_stopping=model.config.early_stopping,
    )
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


# ──────────────────────────────────────────────
# 4.  Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    model, processor, tokenizer = build_arabic_trocr(
        encoder_name  = "microsoft/trocr-base-stage1",
        decoder_name  = "asafaya/bert-base-arabic",
        freeze_encoder= False,   # set True to train only decoder + cross-attn
    )

    model.to(device)
    parameter_summary(model)

    # ── quick sanity-check with a blank image ──
    dummy_image = Image.new("RGB", (384, 384), color=(255, 255, 255))
    text = run_inference(model, processor, tokenizer, dummy_image, device)
    print(f"\nSanity-check output (blank image): '{text}'")