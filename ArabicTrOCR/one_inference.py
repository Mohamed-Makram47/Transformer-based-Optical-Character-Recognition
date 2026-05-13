import sys
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

def run(image_path, model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model     = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)
    processor = ViTImageProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_length=64, num_beams=4)

    print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python predict.py <image_path> <model_path>")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])