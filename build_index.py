import os, json, argparse
import numpy as np
import faiss

def load_artifacts(art_dir: str):
    X_img = np.load(os.path.join(art_dir, "image_emb.npy")).astype("float32")
    X_cap = np.load(os.path.join(art_dir, "caption_emb.npy")).astype("float32")
    with open(os.path.join(art_dir, "paths.json"), encoding="utf-8") as f:
        paths = json.load(f)
    with open(os.path.join(art_dir, "captions.json"), encoding="utf-8") as f:
        caps = json.load(f)
    assert X_img.shape[0] == len(paths), f"rows={X_img.shape[0]} vs paths={len(paths)}"
    assert X_cap.shape[0] == len(caps),  f"rows={X_cap.shape[0]} vs captions={len(caps)}"
    assert X_img.shape[1] == X_cap.shape[1], "embed dims must match"
    return X_img, X_cap, paths, caps

def build_ip_index(X: np.ndarray):
    d = X.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(X)
    return index

def parse_args():
    ap = argparse.ArgumentParser(description="Build FAISS indices for hybrid (image+caption) search")
    ap.add_argument("--artifacts_dir", default="artifacts")
    ap.add_argument("--out_img_index", default="artifacts/images.index")
    ap.add_argument("--out_cap_index", default="artifacts/captions.index")
    return ap.parse_args()

def main():
    args = parse_args()
    print(f"[info] loading artifacts from {args.artifacts_dir}")
    X_img, X_cap, paths, caps = load_artifacts(args.artifacts_dir)

    print(f"[info] building image index: N={X_img.shape[0]}, d={X_img.shape[1]}")
    idx_img = build_ip_index(X_img)
    print(f"[info] building caption index: N={X_cap.shape[0]}, d={X_cap.shape[1]}")
    idx_cap = build_ip_index(X_cap)

    os.makedirs(os.path.dirname(args.out_img_index), exist_ok=True)
    faiss.write_index(idx_img, args.out_img_index)
    faiss.write_index(idx_cap, args.out_cap_index)
    print(f"[ok] saved -> {args.out_img_index} & {args.out_cap_index}")
    print("[tip] keep paths.json and captions.json for ID resolution at query-time.")

if __name__ == "__main__":
    main()