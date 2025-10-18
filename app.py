from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import io

app = FastAPI()

# Lazy-load CLIP and model only when needed
clip_model = None
preprocess = None
embeddings = None
device = "cpu"  # Use CPU — Render free plan has no GPU

@app.on_event("startup")
def startup_event():
    print("🚀 Server started — model will load only when the first request arrives.")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global clip_model, preprocess, embeddings

    # Lazy import CLIP only when needed
    if clip_model is None:
        print("⏳ Loading CLIP model...")
        import clip
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        clip_model.eval()
        print("✅ CLIP model loaded.")

        # Load embeddings file once
        print("⏳ Loading poster embeddings...")
        data = torch.load("poster_embeddings.pt", map_location=device)
        embeddings = data["embeddings"]
        print(f"✅ Loaded {len(embeddings)} embeddings.")

    # Read uploaded image
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        query_emb = clip_model.encode_image(img_tensor)
        query_emb /= query_emb.norm(dim=-1, keepdim=True)

    # Compare against stored embeddings
    best_score = -1
    best_poster = None
    for poster, emb in embeddings.items():
        emb = emb.to(device)
        score = (query_emb @ emb.T).item()
        if score > best_score:
            best_score = score
            best_poster = poster

    return JSONResponse({"poster": best_poster, "score": round(best_score, 3)})


@app.get("/")
def home():
    return {"message": "Poster Recognition API is running 🚀"}