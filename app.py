import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import tensorflow as tf
from contextlib import asynccontextmanager
import uvicorn


# --- Lifespan handler (replaces deprecated @app.on_event("startup")) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    try:
        print("🚀 Loading TensorFlow model...")
        model = tf.keras.models.load_model("poster_model_fixed_215.h5")
        print("✅ Model loaded successfully")
    except Exception as e:
        print("❌ Model failed to load:", e)
        model = None

    yield  # App runs while inside this block

    print("🔻 Shutting down app")


# --- Initialize app ---
app = FastAPI(lifespan=lifespan)


# --- Helper function for image preprocessing ---
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


# --- Endpoint for search ---
@app.post("/search")
async def search(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(content={"error": "Model not loaded"}, status_code=500)

    try:
        # Open and preprocess the image
        image = Image.open(file.file).convert("RGB")
        processed = preprocess_image(image)

        # Run inference
        preds = model.predict(processed)
        prediction = np.argmax(preds, axis=1).tolist()

        return {"prediction": prediction}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# --- Root endpoint ---
@app.get("/")
def root():
    return {"message": "Poster recognition API is running!"}


# --- Main entry point (Render needs this to bind the correct port) ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)