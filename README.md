# Transformer-based Optical Character Recognition for Arabic

This repository hosts the training and evaluation framework for an Arabic Optical Character Recognition (OCR) system. It builds upon the foundational architecture of **TrOCR (Transformer-based Optical Character Recognition)**, leveraging its vision-encoder and text-decoder structure but customized for Arabic text.

## Project Structure

The codebase has been reorganized into two main components:

- **`TrOCR/`**: Contains the original, unmodified source code for Microsoft's TrOCR. This serves as a reference and baseline implementation.
- **`ArabicTrOCR/`**: The core directory for this project. It contains all our custom code, including data pipelines, model configurations, and training scripts built natively in Hugging Face ecosystem to bypass complex Fairseq dependencies.

## ArabicTrOCR: Training Focus

The primary objective of this project is to train an effective Arabic OCR model. We initialize the model using weights from a pre-trained TrOCR vision encoder, and couple it with an Arabic language decoder (`asafaya/bert-base-arabic`).

### Key Features
- **Hugging Face Native**: Uses `VisionEncoderDecoderModel` to fuse the models.
- **Arabic Handling**: Custom dataset loading and processing tailored for Arabic script.
- **Seq2Seq Training**: Training loop implemented using the Hugging Face `Seq2SeqTrainer` with multi-GPU support via `accelerate`.
- **Metrics Tracking**: Real-time evaluation mapping Word Error Rate (WER) and Character Error Rate (CER), with experiment tracking logged to Weights & Biases (WandB).

### Directory Overview (`ArabicTrOCR/`)
- `model.py`: Handles model initialization. Instantiates the `VisionEncoderDecoderModel` by matching pre-trained ViT weights to an Arabic sequence decoder.
- `train.py`: The executable training script. It handles dataset loading, evaluation logic, and kicks off the `Seq2SeqTrainer`.
- `inference.py` / `one_inference.py`: Scripts used to generate text predictions from sample images to evaluate the custom-trained weights.

## Getting Started

1. **Environment Setup**:
Ensure you have the required packages using Conda:
```bash
conda create -n hf_ocr python=3.10 -y
conda activate hf_ocr
pip install torch torchvision transformers datasets accelerate evaluate python-bidi arabic-reshaper wandb
```

2. **Training**:
From the project root, start the training process:
```bash
cd ArabicTrOCR
python train.py
```
*(Ensure you have updated your data paths and parameters like `encoder_name` in the training script if using custom ViT weights).*

3. **Inference**:
To run text generation on a target image:
```bash
python inference.py
```

## Acknowledgments
- Based on the [TrOCR architecture by Microsoft](https://github.com/microsoft/unilm/tree/master/trocr).
- Arabic language decoding powered by [asafaya's Arabic BERT](https://huggingface.co/asafaya/bert-base-arabic).
