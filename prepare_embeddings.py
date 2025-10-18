import os
from PIL import Image
import torch
import clip

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load CLIP model
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# Paths
DATASET_PATH = "dataset"  # your folder of poster folders
EMBEDDINGS_PATH = "poster_embeddings.pt"

embeddings = {}
poster_names = []

for folder in sorted(os.listdir(DATASET_PATH)):
    folder_path = os.path.join(DATASET_PATH, folder)
    if not os.path.isdir(folder_path):
        continue

    # List only image files
    images = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if len(images) < 2:
        print(f"Skipping folder {folder} (not enough images)")
        continue

    folder_embeddings = []
    for img_name in images:
        img_path = os.path.join(folder_path, img_name)
        try:
            img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.encode_image(img)
                emb /= emb.norm(dim=-1, keepdim=True)  # normalize
            folder_embeddings.append(emb)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    if folder_embeddings:
        # Average embeddings for this poster
        avg_emb = torch.stack(folder_embeddings).mean(dim=0)
        avg_emb /= avg_emb.norm(dim=-1, keepdim=True)  # normalize again
        embeddings[folder] = avg_emb.cpu()
        poster_names.append(folder)
        print(f"Saved average embedding for folder {folder}")

# Save embeddings
torch.save({"embeddings": embeddings, "poster_names": poster_names}, EMBEDDINGS_PATH)
print(f"✅ Saved averaged embeddings for {len(embeddings)} posters to {EMBEDDINGS_PATH}")