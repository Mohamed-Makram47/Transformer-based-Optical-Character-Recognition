# Scripts

Shell scripts for training and evaluation.

## Planned Scripts

### Training

- `train_iam.sh` — Fine-tune TrOCR on IAM handwriting dataset
- `train_sroie.sh` — Fine-tune TrOCR on SROIE receipt dataset
- `train_str.sh` — Fine-tune TrOCR on STR benchmarks (scene text)

### Evaluation

- `eval_iam.sh` — Evaluate on IAM test set (CER metric)
- `eval_sroie.sh` — Evaluate on SROIE test set (F1 metric)
- `eval_str.sh` — Evaluate on STR benchmarks (WPA metric)

### Utilities

- `download_models.sh` — Download all pre-trained checkpoints
- `download_data.sh` — Download all datasets

## Usage

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Fine-tune on IAM
bash scripts/train_iam.sh

# Evaluate on IAM
bash scripts/eval_iam.sh
```

## Key Environment Variables

Each script expects:
- `DATA` — Path to the dataset directory
- `SAVE_PATH` — Path to save checkpoints
- `LOG_DIR` — Path for TensorBoard logs
- `MODEL` — Path to the model checkpoint (evaluation only)
