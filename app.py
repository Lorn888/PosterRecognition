# app.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import clip
import io
import os

app = FastAPI()

# Force CPU for Render
device = "cpu"

# Load CLIP model and preprocessing
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# Load averaged poster embeddings
data = torch.load("poster_embeddings.pt", map_location=device)
embeddings = data["embeddings"]  # dict: {poster_id: tensor}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and preprocess image
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        # Encode query image
        with torch.no_grad():
            query_emb = model.encode_image(img_tensor)
            query_emb /= query_emb.norm(dim=-1, keepdim=True)

        # Compare to stored embeddings
        best_score = -1
        best_poster = None
        for poster, emb in embeddings.items():
            score = (query_emb @ emb.T).item()
            if score > best_score:
                best_score = score
                best_poster = poster

        return JSONResponse({
            "poster": best_poster,
            "score": round(best_score, 2)
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# Run Uvicorn if executed directly
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=port)