from app.model import load_models
from app.config import FIELDS, DEVICE
import torch

processor, model = load_models()

def prepare_prompts():
    return {
        field: f"<s_docvqa><s_question>{q}</s_question><s_answer>"
        for field, q in FIELDS.items()
    }

def process_image(image):
    return processor(image, return_tensors="pt").pixel_values.to(DEVICE)

def generate_output(pixel_values, prompt):
    decoder_input = processor.tokenizer(
        prompt, 
        add_special_tokens=False, 
        return_tensors="pt"
    ).input_ids.to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input,
            max_length=model.config.decoder.max_position_embeddings,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            return_dict_in_generate=True
        )
    
    # Handle both old and new output formats
    sequences = outputs.sequences if hasattr(outputs, 'sequences') else outputs
    return processor.batch_decode(sequences, skip_special_tokens=True)[0]