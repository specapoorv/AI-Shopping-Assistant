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


def clip_find_top_k_similar_in_category(feature_dir, query_feat, k=8):
	"""
	Loads all .npy files in feature_dir into a (N, D) array,
	and returns the top-k (filepath, cosine_similarity) pairs
	for query_feat.
	"""
	# --- ensure query_feat is a 1D vector of length D ---
	q = np.asarray(query_feat)
	q = q.squeeze()            # drops any singleton dims
	q = q.reshape(-1)          # force shape (D,)
	q = q / np.linalg.norm(q)  # normalize

	# --- load the feature bank ---
	feats = []
	paths = []
	for fname in os.listdir(feature_dir):
		if not fname.lower().endswith('.npy'):
			continue
		full_path = os.path.join(feature_dir, fname)
		vec = np.load(full_path).squeeze()
		feats.append(vec)
		paths.append(full_path)
	feats = np.stack(feats, axis=0)  # shape (N, D)

	# --- normalize bank and compute cosine sims ---
	F = feats / np.linalg.norm(feats, axis=1, keepdims=True)  # (N, D)
	sims = F.dot(q)                                           # (N,)

	# --- pick top-k ---
	idxs = np.argsort(sims)[-k:][::-1]
	return [(paths[i], float(sims[i])) for i in idxs]

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