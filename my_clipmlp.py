import torch, clip
from PIL import Image
from pathlib import Path
import numpy as np
import torch.nn as nn
import json
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---------- Preparation (run once) ----------
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) Load frozen CLIP
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

# 2) Define *exact* same MLP architecture as during training
class MLPProbe(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=256, num_classes=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x): return self.net(x)

# --- Load class list (saved during training) ---
#   > with open("class_list.json","w") as f: json.dump(classes, f)
with open("class_list.json") as f:
    classes = json.load(f)

probe = MLPProbe(num_classes=len(classes)).to(device)
probe.load_state_dict(torch.load("best_mlp_probe.pth", map_location=device))
probe.eval()

# ---------- Inference function ----------
@torch.inference_mode()
def classify_image_clipmlp(image):
    """Return predicted class name (and logits) for one image."""
    # 1. CLIP preprocessing + encode
    img = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
    feat = clip_model.encode_image(img)
    feat = feat / feat.norm(dim=-1, keepdim=True)       # 1×512

    # 2. Probe forward pass
    logits = probe(feat)                                # 1×C

    # 3. Top-1
    pred_idx = logits.argmax(dim=-1).item()
    pred_cls = classes[pred_idx]
    return pred_cls, feat #, logits.cpu().numpy().squeeze()



def clip_find_top_k_similar_in_category(bank_path: str | os.PathLike, query_feat, k: int = 8):
    bank_path = Path(bank_path)

    # --- resolve the single .npz file --------------------------------------
    if bank_path.is_dir():
        npz_files = [p for p in bank_path.iterdir() if p.suffix == ".npz"]
        if len(npz_files) != 1:
            raise ValueError(
                f"Expected exactly one .npz in {bank_path}, found {len(npz_files)}"
            )
        bank_path = npz_files[0]

    if bank_path.suffix != ".npz":
        raise ValueError("bank_path must point to a .npz file or its parent folder")

    # --- load feature bank --------------------------------------------------
    data   = np.load(bank_path, allow_pickle=False)
    feats  = data["feats"]           # shape (N, D)
    names  = data["names"]           # shape (N,)

    # --- prepare query vector ----------------------------------------------
    q = np.asarray(query_feat).squeeze().reshape(-1)
    q /= np.linalg.norm(q) + 1e-12    # avoid divide-by-zero

    # --- cosine similarities -----------------------------------------------
    F    = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
    sims = F @ q                      # (N,)

    # --- top-k --------------------------------------------------------------
    if k >= len(sims):
        idxs = np.argsort(sims)[::-1]
    else:
        idxs = np.argpartition(-sims, k)[:k]
        idxs = idxs[np.argsort(-sims[idxs])]

    return [(str(names[i]), float(sims[i])) for i in idxs]


def encode_one_text(text: str) -> torch.Tensor:
    # 1) Tokenize expects a list, so wrap your string in a list
    text_tokens = clip.tokenize([text]).to(device)   # shape: [1, L]

    # 2) Encode and normalize
    with torch.no_grad():
        text_feat = clip_model.encode_text(text_tokens)    # shape: [1, 512]
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    # 3) Remove the batch dim → shape: [512]
    return text_feat.squeeze(0)

# ---------- Example usage ----------
if __name__ == "__main__":
    test_img = Image.open("test2.jpg")
    cls, clip_feat = classify_image_clipmlp(test_img)
    print(f"Predicted class: {cls}")