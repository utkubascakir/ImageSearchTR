import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

image_emb = np.load("NM_project/image_search_project/artifacts/image_emb.npy")
caption_emb = np.load("NM_project/image_search_project/artifacts/caption_emb.npy")

with open("NM_project/image_search_project/artifacts/paths.json", "r", encoding="utf-8") as f:
    paths = json.load(f)

with open("NM_project/image_search_project/artifacts/captions.json", "r", encoding="utf-8") as f:
    captions = json.load(f)

print("[info] image_emb shape:", image_emb.shape)
print("[info] caption_emb shape:", caption_emb.shape)
print("[info] Number of items:", len(paths))

sim_matrix = cosine_similarity(caption_emb, image_emb)

# For each caption, find the most similar image
for i, cap in enumerate(captions):
    sims = sim_matrix[i]
    best_idx = int(np.argmax(sims))
    print(f"\nCaption {i}: {cap}")
    print(f"  Most similar image: {paths[best_idx]} (score={sims[best_idx]:.4f})")