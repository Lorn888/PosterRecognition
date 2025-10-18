import torch
from PIL import Image
import clip

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# Load averaged embeddings
data = torch.load("poster_embeddings.pt")
embeddings = data["embeddings"]
poster_names = data["poster_names"]

# Load query image
query_path = input("Enter image path to test: ").strip()
img = preprocess(Image.open(query_path)).unsqueeze(0).to(device)
with torch.no_grad():
    query_emb = model.encode_image(img)
    query_emb /= query_emb.norm(dim=-1, keepdim=True)

# Find closest poster
best_score = -1
best_poster = None
for poster, emb in embeddings.items():
    emb = emb.to(device)
    score = (query_emb @ emb.T).item()  # cosine similarity
    if score > best_score:
        best_score = score
        best_poster = poster

print(f"🎯 Predicted poster: {best_poster} (score {best_score:.2f})")