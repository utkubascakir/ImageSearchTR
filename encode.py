import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, math, json, glob, argparse

from tqdm import tqdm
from PIL import Image
from typing import List, Optional
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor


# MODEL CLASS
def masked_mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    lengths = mask.sum(dim=1).clamp(min=1e-6)
    return summed / lengths

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_mult=2.0, p_drop=0.4):
        super().__init__()
        h = int(hidden_mult * out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(h, out_dim),
        )
        self.ln = nn.LayerNorm(out_dim)
        self.use_residual = (in_dim == out_dim)

    def forward(self, x):
        y = self.net(x)
        if self.use_residual:
            y = y + x
        return self.ln(y)

class MultiModalEmbedder(nn.Module):
    def __init__(self, text_encoder, vision_encoder,
                 text_dim=768, image_dim=768, embed_dim=768,
                 temperature_init=1/0.07, use_mean_pooling_for_text=True,
                 freeze_encoders=True):
        super().__init__()
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder
        self.embed_dim = embed_dim
        self.use_mean_pooling_for_text = use_mean_pooling_for_text
        self.text_proj = ProjectionHead(text_dim, embed_dim)
        self.image_proj = ProjectionHead(image_dim, embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(temperature_init), dtype=torch.float))

        for p in self.text_encoder.parameters():   p.requires_grad = not freeze_encoders
        for p in self.vision_encoder.parameters(): p.requires_grad = not freeze_encoders

    def forward_text(self, input_ids, attention_mask):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = masked_mean_pool(out.last_hidden_state, attention_mask) \
                 if self.use_mean_pooling_for_text else out.last_hidden_state[:, 0, :]
        return F.normalize(self.text_proj(pooled), dim=-1)

    def forward_image(self, pixel_values):
        out = self.vision_encoder(pixel_values=pixel_values, return_dict=True)
        cls = out.last_hidden_state[:, 0, :]
        return F.normalize(self.image_proj(cls), dim=-1)

    @torch.no_grad()
    def encode_text(self, inputs: dict):
        ids  = inputs["input_ids"]
        mask = inputs.get("attention_mask")
        if mask is None:
            mask = torch.ones_like(ids)
        return self.forward_text(ids, mask)

    @torch.no_grad()
    def encode_image(self, inputs: dict):
        return self.forward_image(inputs["pixel_values"])



# BUILD THE MODEL
def load_model_and_io(text_model: str, vision_model: str, device: torch.device):
    print("[info] Downloading tokenizer and processor...")
    tokenizer = AutoTokenizer.from_pretrained(text_model)
    img_proc  = AutoImageProcessor.from_pretrained(vision_model)
    print("[ok] Tokenizer and processor are ready.")

    print("[info] Downloading text and image encoders...")
    text_encoder   = AutoModel.from_pretrained(text_model)
    vision_encoder = AutoModel.from_pretrained(vision_model)
    print("[ok] Encoders are ready.")

    print("[info] Building the model...")
    model = MultiModalEmbedder(
        text_encoder=text_encoder,
        vision_encoder=vision_encoder,
        text_dim=768, image_dim=768, embed_dim=768,
        use_mean_pooling_for_text=True,
        freeze_encoders=False
    ).to(device)
    print("[ok] Model is ready.")
    return model, tokenizer, img_proc



# LOAD WEIGHTS
def load_weights(model, device,
                 from_hf: bool = True,
                 hf_repo: str | None = None,
                 hf_filename: str = "pytorch_model.bin",
                 local_pth: str | None = None):
    """
    - from_hf=True  -> If pytorch_model.bin exists locally, load it; otherwise download from HuggingFace Hub.
    - from_hf=False -> Load a local .pth file provided via local_pth (developer usage).
    """
    if from_hf:
        # First check if pytorch_model.bin already exists locally
        if os.path.isfile(hf_filename):
            ckpt_path = hf_filename
            print(f"[info] Local file found: {ckpt_path}")
        else:
            if not hf_repo:
                raise ValueError("If from_hf=True, you must provide hf_repo.")
            print(f"[info] Downloading from HuggingFace Hub: {hf_repo}/{hf_filename}")
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(repo_id=hf_repo, filename=hf_filename)
    else:
        if not local_pth:
            raise ValueError("If from_hf=False, you must provide local_pth (.pth file).")
        ckpt_path = local_pth
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"File not found: {ckpt_path}")

    print(f"[info] Loading weights from: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    # Some checkpoints may wrap weights inside a "state_dict" key
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[ok] Weights loaded. missing={len(missing)} unexpected={len(unexpected)}")



# IMAGE PROCESSING
def list_images(images_dir: str):
    exts = ("*.jpg","*.jpeg","*.png","*.webp","*.bmp")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(images_dir, e)))
    paths = sorted(paths)
    if not paths:
        raise SystemExit(f"[err] No supported image files found inside '{images_dir}'.")
    return paths

def encode_images(model, img_proc, image_paths, device: torch.device, batch_size: int = 32):
    model.eval()
    print(f"[info] Encoding images... (N={len(image_paths)}, batch={batch_size})")
    embs = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = img_proc(imgs, return_tensors="pt").to(device)
        e = model.encode_image(inputs)  
        embs.append(e.cpu().numpy())

    X = np.concatenate(embs, axis=0).astype("float32")
    print("[ok] All image embeddings are generated.")
    return X



# TEXT PROCESSING
def load_captions_jsonl(jsonl_path: str) -> dict[str, str]:
    caps = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = os.path.basename(obj["path"])
            caps[key] = obj["caption"]
    return caps

def align_captions_to_paths(image_paths: list[str], caps_map: dict[str, str]) -> list[str]:
    captions = []
    missing = []
    for p in image_paths:
        fname = os.path.basename(p)
        if fname in caps_map:
            captions.append(caps_map[fname])
        else:
            missing.append(fname)
    if missing:
        raise SystemExit(f"[err] Missing captions for {len(missing)} images: {missing[:5]} ...")
    return captions

def encode_texts(model, tokenizer, texts: list[str], device: torch.device, batch_size: int = 64):
    model.eval()
    print(f"[info] Encoding captions... (N={len(texts)}, batch={batch_size})")
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        e = model.encode_text(inputs)
        embs.append(e.cpu().numpy())
    X = np.concatenate(embs, axis=0).astype("float32")
    print("[ok] All caption embeddings are generated.")
    return X



# SAVE OUTPUTS 
def save_outputs(out_dir: str,
                 image_emb: np.ndarray,
                 image_paths: list[str],
                 caption_emb: np.ndarray,
                 captions: list[str]):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "image_emb.npy"), image_emb)
    np.save(os.path.join(out_dir, "caption_emb.npy"), caption_emb) 
    with open(os.path.join(out_dir, "paths.json"), "w", encoding="utf-8") as f:
        json.dump(image_paths, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "captions.json"), "w", encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False, indent=2)
    print(f"[ok] Saved artifacts to: {out_dir}")



# ARGS
def parse_args():
    ap = argparse.ArgumentParser(description="Simple image & caption embedding exporter")
    ap.add_argument("--images_dir", required=True, help="Directory containing images to encode")
    ap.add_argument("--captions_jsonl", required=True,
                    help="JSONL with {'path': <img or filename>, 'caption': <text>} per line") 
    ap.add_argument("--out_dir", default="artifacts", help="Output directory")
    ap.add_argument("--text_model", default="newmindai/modernbert-base-tr-uncased-allnli-stsb")
    ap.add_argument("--vision_model", default="facebook/dinov2-base")
    # weights
    ap.add_argument("--from_hf", action="store_true", help="Load weights from HF Hub (default: local .pth)")
    ap.add_argument("--hf_repo", default=None)
    ap.add_argument("--hf_filename", default="pytorch_model.bin")
    ap.add_argument("--local_pth", default=None)
    # encode
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    return ap.parse_args()


# MAIN
def main():
    args = parse_args()
    device = torch.device("cuda" if (args.device=="auto" and torch.cuda.is_available()) else
                          ("cuda" if args.device=="cuda" else "cpu"))
    print(f"[info] Device: {device}")

    image_paths = list_images(args.images_dir)
    print(f"[info] Image count: {len(image_paths)}")

    model, tokenizer, img_proc = load_model_and_io(args.text_model, args.vision_model, device)

    if args.from_hf:
        load_weights(model, device, from_hf=True, hf_repo=args.hf_repo, hf_filename=args.hf_filename)
    else:
        load_weights(model, device, from_hf=False, local_pth=args.local_pth)

    # images
    image_emb = encode_images(model, img_proc, image_paths, device, batch_size=args.batch_size)

    # captions
    caps_map = load_captions_jsonl(args.captions_jsonl)
    caps = align_captions_to_paths(image_paths, caps_map)
    caption_emb = encode_texts(model, tokenizer, caps, device, batch_size=64)

    save_outputs(args.out_dir,
                 image_emb=image_emb,
                 image_paths=image_paths,
                 caption_emb=caption_emb,
                 captions=caps)
    print("[done] encode.py finished.")



if __name__ == "__main__":
    main()