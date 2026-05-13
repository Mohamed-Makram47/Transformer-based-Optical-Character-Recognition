# Transformer-based Optical Character Recognition for Arabic

This repository hosts the training and evaluation framework for an Arabic Optical Character Recognition (OCR) system. It builds upon the foundational architecture of **TrOCR (Transformer-based Optical Character Recognition)**, leveraging its vision-encoder and text-decoder structure but customized for Arabic text.

---

# Project Resources


* **Project Poster:**
   https://arabic-handwritten-textrecognition.netlify.app/

  * **Video:**
     https://drive.google.com/file/d/1AHnMD4NUtd1vpREKj7Yg4C0D0CpJhB0Y/view?usp=sharing
    

---

## Project Structure

The codebase has been reorganized into two main components:

* **`TrOCR/`**: Contains the original, unmodified source code for Microsoft's TrOCR. This serves as a reference and baseline implementation.
* **`ArabicTrOCR/`**: The core directory for this project. It contains all our custom code, including data pipelines, model configurations, and training scripts built natively in Hugging Face ecosystem to bypass complex Fairseq dependencies.

---

## ArabicTrOCR: Training Focus

The primary objective of this project is to train an effective Arabic OCR model. We initialize the model using weights from a pre-trained TrOCR vision encoder, and couple it with an Arabic language decoder (`asafaya/bert-base-arabic`).

### Key Features

* **Hugging Face Native**: Uses `VisionEncoderDecoderModel` to fuse the models.
* **Arabic Handling**: Custom dataset loading and processing tailored for Arabic script.
* **Arabic Decoder Initialization**: Replaces the original English-oriented decoder with `bert-base-arabic` to improve Arabic language understanding and contextual generation.
* **Seq2Seq Training**: Training loop implemented using the Hugging Face `Seq2SeqTrainer` with multi-GPU support via `accelerate`.
* **Metrics Tracking**: Real-time evaluation using Word Error Rate (WER) and Character Error Rate (CER), with experiment tracking logged to Weights & Biases (WandB).

---

## Directory Overview (`ArabicTrOCR/`)

* `model.py`: Handles model initialization. Instantiates the `VisionEncoderDecoderModel` by matching pre-trained ViT weights to an Arabic sequence decoder.
* `train.py`: The executable training script. It handles dataset loading, evaluation logic, and launches the `Seq2SeqTrainer`.
* `inference.py` / `one_inference.py`: Scripts used to generate text predictions from sample images to evaluate the custom-trained weights.

---

## Getting Started

### 1. Environment Setup

Ensure you have the required packages using Conda:

```bash
conda create -n hf_ocr python=3.10 -y
conda activate hf_ocr
pip install torch torchvision transformers datasets accelerate evaluate python-bidi arabic-reshaper wandb
```

### 2. Training

From the project root, start the training process:

```bash
cd ArabicTrOCR
python train.py
```

> Ensure you update your dataset paths and parameters such as `encoder_name` if using custom ViT checkpoints.

### 3. Inference

To generate text predictions from a target image:

```bash
python inference.py
```

---

## Evaluation Metrics

The model is evaluated using:

* **CER (Character Error Rate)** — primary OCR metric
* **WER (Word Error Rate)**
* **BLEU Score**

Our fine-tuned TrOCR model achieved:

| Metric | Result |
| ------ | ------ |
| CER    | 8.10%  |
| WER    | 12.50% |

---

## Acknowledgments

* Based on the [TrOCR architecture by Microsoft](https://github.com/microsoft/unilm/tree/master/trocr).
* Arabic language decoding powered by [asafaya's Arabic BERT](https://huggingface.co/asafaya/bert-base-arabic).
* Dataset used: **KHATT v1.0 Arabic Handwritten Text Dataset**.
