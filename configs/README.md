# Configs

This directory contains configuration files for training and evaluation runs.

## Planned Configuration Files

- `train_iam.yaml` — Fine-tuning hyperparameters for IAM handwriting dataset
- `train_sroie.yaml` — Fine-tuning hyperparameters for SROIE receipt dataset
- `train_str.yaml` — Fine-tuning hyperparameters for STR benchmarks
- `eval_iam.yaml` — Evaluation config for IAM
- `eval_sroie.yaml` — Evaluation config for SROIE
- `eval_str.yaml` — Evaluation config for STR benchmarks

## Config Format

Configs specify:
- Model architecture (`trocr_small`, `trocr_base`, `trocr_large`)
- Data paths and data type (`STR`, `SROIE`, `Receipt53K`)
- Training hyperparameters (LR, batch size, epochs, patience)
- Preprocessing method (`DA2`, `RandAugment`)
- BPE tokenizer (`gpt2` or `sentencepiece`)
- Decoder pretrained weights (`roberta2` or `unilm`)
