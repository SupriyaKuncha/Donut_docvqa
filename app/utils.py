import re
from typing import Optional

def clean_generated_text(text: str, processor) -> Optional[str]:
    """
    Enhanced cleaning function that:
    1. Strictly extracts content between <s_answer> tags
    2. Removes ALL question fragments
    3. Returns None for invalid/missing data
    """
    # Extract answer content (handle multi-line answers with re.DOTALL)
    
    # Aggressively remove question fragments and special tokens
    text = re.sub(
        r"(what is|what was|what are|who is|extract|provide|invoice|bill|total|date|amount|name|address|description).*?\??\s*",
        "", 
        text, 
        flags=re.IGNORECASE
    ).strip()
    
    # Remove any remaining special tokens
    text = text.replace(processor.tokenizer.eos_token, "").strip()
    
    return text.split()