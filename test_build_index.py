import os, json
import numpy as np
import faiss

ART = "artifacts"

# index ve metadata yükle
img_index = faiss.read_index(os.path.join(ART, "images.index"))
cap_index = faiss.read_index(os.path.join(ART, "captions.index"))

with open(os.path.join(ART, "paths.json"), encoding="utf-8") as f:
    paths = json.load(f)
with open(os.path.join(ART, "captions.json"), encoding="utf-8") as f:
    captions = json.load(f)

cap_emb = np.load(os.path.join(ART, "caption_emb.npy")).astype("float32")

print(f"[info] Indexes loaded. images={img_index.ntotal}, captions={cap_index.ntotal}")

# her caption embedding ile image indexte arama yap
for i, cap in enumerate(captions):
    q = cap_emb[i:i+1]  # (1,d)
    scores, ids = img_index.search(q, k=1)
    hit_id = int(ids[0][0])
    print(f"\nQuery caption: {cap}")
    print(f" -> Top image: {paths[hit_id]}  (score={scores[0][0]:.4f})")

# bonus: caption indexte kendi embeddingini arat (kendini bulmalı)
q = cap_emb[0:1]
scores, ids = cap_index.search(q, k=2)
print("\n[test] Caption self-search")
for i, s in zip(ids[0], scores[0]):
    print(f" id={i}, score={s:.4f}, text='{captions[int(i)]}'")
