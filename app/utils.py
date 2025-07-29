import re
from typing import Optional


def clean_generated_text(text: str, processor) -> Optional[str]:
    text = text.split('?')[1]
    return text.strip()

