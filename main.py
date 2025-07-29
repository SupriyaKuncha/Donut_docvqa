from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from app.config import DEVICE
from app.services import processor, prepare_prompts, process_image, generate_output
from app.utils import clean_generated_text


app = FastAPI()
origins = [
    "http://localhost",          # Allows requests from 'http://localhost' directly (e.g., if you opened index.html directly)
    "http://localhost:8000",     # Allows requests from your backend's own specific address
    "http://localhost:8501",     # This is crucial for your Streamlit frontend
]

# Add the CORS middleware to your FastAPI application.
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # List of allowed origins
    allow_credentials=True,         # Allow cookies to be included in cross-origin requests
    allow_methods=["*"],            # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],            # Allow all headers in the request
)

@app.post("/process_invoice")
async def process_invoice(file: UploadFile = File(...)):
    try:
        # Receive Image: Read the uploaded file contents
        contents = await file.read() 
        image = Image.open(io.BytesIO(contents)).convert("RGB") # Open the image using Pillow (PIL) and convert to RGB

        # Prepare Image and Prompts:
        pixel_values = process_image(image) # Preprocess image using Donut processor
        prompts = prepare_prompts()  # Get the list of questions (prompts)
        
        # Process Each Field:
        results = {}
        for field, prompt in prompts.items():
            decoded = generate_output(pixel_values, prompt)  # Generate output (answer) for each specific question
            results[field] = clean_generated_text(decoded, processor) 
        
        # Return Results: Return a JSON response with all extracted field
        return JSONResponse(content=results)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
