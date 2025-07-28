from transformers import DonutProcessor, VisionEncoderDecoderModel
from app.config import MODEL_NAME, DEVICE

def load_models():
    processor = DonutProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(DEVICE)
    return processor, model