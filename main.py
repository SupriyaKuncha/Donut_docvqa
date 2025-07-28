from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
from app.config import DEVICE
from app.services import processor, prepare_prompts, process_image, generate_output
from app.utils import clean_generated_text

app = FastAPI()

@app.post("/process_invoice")
async def process_invoice(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        pixel_values = process_image(image)
        prompts = prepare_prompts()
        
        results = {}
        for field, prompt in prompts.items():
            decoded = generate_output(pixel_values, prompt)
            results[field] = clean_generated_text(decoded, processor)
        
        return JSONResponse(content=results)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))