# Pre-trained Model Checkpoints

Download pre-trained model checkpoints and place them in this directory.

## Available Checkpoints

### Stage 1 Pre-trained Models (for fine-tuning)

| Model | Download |
|-------|----------|
| TrOCR-Small (Stage 1) | [trocr-small-stage1.pt](https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-small-stage1.pt) |
| TrOCR-Base (Stage 1) | [trocr-base-stage1.pt](https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-base-stage1.pt) |
| TrOCR-Large (Stage 1) | [trocr-large-stage1.pt](https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-large-stage1.pt) |

### Fine-tuned Models (ready for inference)

| Model | Task | Download |
|-------|------|----------|
| TrOCR-Small | IAM Handwritten | [trocr-small-handwritten.pt](https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-small-handwritten.pt) |
| TrOCR-Base | IAM Handwritten | [trocr-base-handwritten.pt](https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-base-handwritten.pt) |
| TrOCR-Large | IAM Handwritten | [trocr-large-handwritten.pt](https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-large-handwritten.pt) |
| TrOCR-Small | SROIE Printed | [trocr-small-printed.pt](https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-small-printed.pt) |
| TrOCR-Base | SROIE Printed | [trocr-base-printed.pt](https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-base-printed.pt) |
| TrOCR-Large | SROIE Printed | [trocr-large-printed.pt](https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-large-printed.pt) |
| TrOCR-Base | STR Benchmarks | [trocr-base-str.pt](https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-base-str.pt) |
| TrOCR-Large | STR Benchmarks | [trocr-large-str.pt](https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-large-str.pt) |

> **If downloads fail**, append this suffix to the URL:
> `?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D`

## Usage

Reference a checkpoint file in the training/evaluation commands:
```bash
--finetune-from-model ./pretrained/trocr-large-stage1.pt   # for fine-tuning
--path ./pretrained/trocr-large-handwritten.pt              # for evaluation
```
