# Models

This directory contains the model architecture definitions for TrOCR.

## Architecture

TrOCR follows an **encoder-decoder** Transformer architecture:

### Encoder (Image Transformer)
- **DeiT** (Data-efficient Image Transformer) or **BEiT** (Bidirectional Encoder from Image Transformers)
- Splits the input 384×384 image into 16×16 patches → 576 patch tokens + 2 special tokens
- Outputs a sequence of `(N+2) × D` embeddings passed to the decoder

### Decoder (Text Transformer)
- **RoBERTa** (for base/large models) or **UniLM** (for small models)
- Cross-attention to the encoder's visual embeddings
- Auto-regressively generates wordpiece tokens using beam search

### Registered Architectures (fairseq)

| Architecture Name | Encoder | Decoder |
|-------------------|---------|---------|
| `trocr_small` | DeiT-Small (384) | 6-layer, 256-dim |
| `trocr_base` | BEiT-Base (384) | 12-layer, 1024-dim |
| `trocr_large` | BEiT-Large (384) | 12-layer, 1024-dim |

## Source Files (in `src/`)

- `trocr_models.py` — Main TrOCR model (encoder-decoder) and architecture registrations
- `vit_models.py` — ViT/DeiT/BEiT encoder implementations
- `unilm_models.py` — UniLM-based decoder for small models
- `deit.py` — DeiT model definitions registered with `timm`
