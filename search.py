import os, json
import numpy as np
import faiss
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor, AutoModel

from encode import ProjectionHead, MultiModalEmbedder

# GLOBAL CACHE
_RT = {
    "device": None,
    "tokenizer": None,
    "img_proc": None,
    "model": None,
    "img_index": None,
    "cap_index": None,
    "paths": None,
    "captions": None,
}



# DEFINE MODEL FOR ONCE
def init(text_model="newmindai/modernbert-base-tr-uncased-allnli-stsb",
         vision_model="facebook/dinov2-base",
         device="auto"):
    if _RT["model"] is not None:
        return
    
    # Set the device
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
        
    print(f"[init] device={dev}")
    tok = AutoTokenizer.from_pretrained(text_model)
    proc = AutoImageProcessor.from_pretrained(vision_model)

    txt_enc = AutoModel.from_pretrained(text_model)
    vis_enc = AutoModel.from_pretrained(vision_model)

    mdl = MultiModalEmbedder(
        text_encoder=txt_enc,
        vision_encoder=vis_enc,
        text_dim=768, image_dim=768, embed_dim=768,
        use_mean_pooling_for_text=True,
        freeze_encoders=True,
    ).to(dev).eval()
    
    _RT.update({"device": dev, "tokenizer": tok, "img_proc": proc, "model": mdl})
    
    
    
# LOAD ARTIFACTS FOR ONCE
def load_indices(artifacts_dir="artifacts",
                 img_index_path="artifacts/images.index",
                 cap_index_path="artifacts/captions.index"):
    if _RT["img_index"] is not None:
        return

    with open(os.path.join(artifacts_dir, "paths.json"), encoding="utf-8") as f:
        _RT["paths"] = json.load(f)
    with open(os.path.join(artifacts_dir, "captions.json"), encoding="utf-8") as f:
        _RT["captions"] = json.load(f)
        
    _RT["img_index"] = faiss.read_index(img_index_path)
    try:
        _RT["cap_index"] = faiss.read_index(cap_index_path)
    except Exception as e:
        print(f"[warn] captions.index yüklenemedi: {e}")
        _RT["cap_index"] = None

    print(f"[ok] indices: images={_RT['img_index'].ntotal}, captions={_RT['cap_index'].ntotal if _RT['cap_index'] else 0}")
    


# TOKENIZE + ENCODE (for user's query) 
def encode_text(query: str) -> np.ndarray:
    tok = _RT["tokenizer"]([query], padding=True, truncation=True, return_tensors="pt").to(_RT["device"])
    q = _RT["model"].encode_text(tok)  
    q = q.detach().cpu().numpy().astype("float32")
    if q.ndim == 1:  # (d,) -> (1,d)
        q = np.expand_dims(q, axis=0)
    return q



# SEARCH
def _minmax(x: np.ndarray) -> np.ndarray:
    """
    Min-max normalize scores to [0,1] so image and caption 
    similarities become comparable for fusion.
    """
    a, b = float(np.min(x)), float(np.max(x))
    if b - a < 1e-6: return np.zeros_like(x)
    return (x - a) / (b - a)

# SEARCH HELPERS
def _filter_faiss(ids: np.ndarray, scores: np.ndarray):
    if (ids < 0).any():
        mask = ids >= 0
        return ids[mask], scores[mask]
    return ids, scores

def search_image_only(q_vec: np.ndarray, topk: int = 12):
    scores, ids = _RT["img_index"].search(q_vec, topk)
    ids, scores = ids[0], scores[0]
    ids, scores = _filter_faiss(ids, scores)

    results = []
    for i, s in zip(ids.tolist(), scores.tolist()):
        results.append({
            "path": _RT["paths"][int(i)],
            "caption": _RT["captions"][int(i)],
            "score": float(s),
            "img_score": float(s),
            "cap_score": None
        })
    return results

def search_hybrid(q_vec: np.ndarray, topk: int = 30, alpha: float = 0.6):
    if _RT["cap_index"] is None:
        return search_image_only(q_vec, min(topk, 12))

    img_s, img_ids = _RT["img_index"].search(q_vec, topk)
    cap_s, cap_ids = _RT["cap_index"].search(q_vec, topk)
    img_s, img_ids = img_s[0], img_ids[0]
    cap_s, cap_ids = cap_s[0], cap_ids[0]

    img_ids, img_s = _filter_faiss(img_ids, img_s)
    cap_ids, cap_s = _filter_faiss(cap_ids, cap_s)

    img_n = _minmax(img_s)
    cap_n = _minmax(cap_s)

    img_map = {int(i): float(s) for i, s in zip(img_ids, img_n)}
    cap_map = {int(i): float(s) for i, s in zip(cap_ids, cap_n)}

    merged = []
    for cid in (set(img_map) | set(cap_map)):
        si = img_map.get(cid, 0.0)
        sc = cap_map.get(cid, 0.0)
        merged.append((cid, alpha*si + (1-alpha)*sc, si, sc))

    merged.sort(key=lambda x: x[1], reverse=True)
    merged = merged[:topk]

    results = []
    for cid, sf, si, sc in merged:
        results.append({
            "path": _RT["paths"][cid],
            "caption": _RT["captions"][cid],
            "score": float(sf),
            "img_score": float(si),
            "cap_score": float(sc)
        })
    return results



# TEST
if __name__ == "__main__":
    init(device="cpu")
    load_indices()
    q = encode_text("kumsalda duran sarı bir köpek")
    print("[fast]", search_image_only(q, topk=5)[:2])
    print("[deep]", search_hybrid(q, topk=5, alpha=0.6)[:2])