# Data

This directory holds all datasets used for training and evaluation.

## Downloading Datasets

| Dataset | Link | Extract To |
|---------|------|-----------|
| IAM Handwriting | [IAM.tar.gz](https://layoutlm.blob.core.windows.net/trocr/dataset/IAM.tar.gz) | `data/IAM/` |
| SROIE Task 2 | [SROIE_Task2_Original.tar.gz](https://layoutlm.blob.core.windows.net/trocr/dataset/SROIE_Task2_Original.tar.gz) | `data/SROIE/` |
| STR Benchmarks | [STR_BENCHMARKS.zip](https://layoutlm.blob.core.windows.net/trocr/dataset/STR_BENCHMARKS.zip) | `data/STR_Benchmarks/` |

> **If downloads fail**, append this suffix to the URL:
> `?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D`

## Expected Directory Structure

```
data/
├── IAM/
│   ├── train/
│   │   └── gt_train.txt
│   ├── valid/
│   │   └── gt_valid.txt
│   ├── test/
│   │   └── gt_test.txt
│   └── image/
│       └── *.png
├── SROIE/
│   ├── train/
│   │   ├── *.jpg
│   │   └── *.txt
│   └── test/
│       ├── *.jpg
│       └── *.txt
└── STR_Benchmarks/
    ├── gt_train.txt
    ├── gt_test.txt
    └── image/
        └── *.png
```

## Ground Truth Format

- **IAM / STR:** Tab-separated `image_filename\ttext_label`
- **SROIE:** Each `.txt` file contains lines with `x1,y1,x2,y2,x3,y3,x4,y4,text`
