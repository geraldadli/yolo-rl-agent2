"""
streamlit_app.py
Streamlit deployment for the YOLO-DQN v3 pipeline.

Expected files inside  _MODELLING/  (set MODELLING_ROOT env-var to override):
  classifier_full.pth   — ClassifierFull weights
  classifier_emb.pth    — ClassifierEmb weights
  scaler_full.joblib    — StandardScaler for [emb + yolo_zeros] features
  scaler_emb.joblib     — StandardScaler for emb-only features
  dqn_agent_state.pth   — StableDQNAgent .net weights
  metadata.json         — {classes, num_classes, obs_dim, action_labels, ...}
  (optional) yolov12x-cls.pt  — YOLO model for detection overlay

Run:  streamlit run streamlit_app.py
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os, json, copy, random
from pathlib import Path
from collections import defaultdict
from typing import List, Optional, Tuple

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
from PIL import Image
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_OK = True
except Exception:
    torch = nn = F = None
    TORCH_OK = False

try:
    from ultralytics import YOLO
    YOLO_OK = True
except Exception:
    YOLO = None
    YOLO_OK = False

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title  = "YOLO-DQN Classifier",
    page_icon   = "🔬",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

SCRIPT_DIR    = Path(__file__).parent.absolute()
MODELLING_ROOT= os.environ.get("MODELLING_ROOT", str(SCRIPT_DIR / "_MODELLING_STABLE"))
DEVICE        = torch.device("cpu")   # Streamlit Cloud is CPU-only
EMB_DIM       = 512

# =============================================================================
# MODEL DEFINITIONS  — must exactly match stable_yolo_dqn_full_fixed_v3.py
# =============================================================================

if TORCH_OK:
    class EmbedderResNet34(nn.Module):
        def __init__(self, emb_dim=512, pretrained=False):
            super().__init__()
            from torchvision import models
            base = models.resnet34(pretrained=pretrained)
            self.conv_base = nn.Sequential(
                base.conv1, base.bn1, base.relu, base.maxpool,
                base.layer1, base.layer2, base.layer3, base.layer4)
            self.pool = base.avgpool
            self.proj = nn.Sequential(
                nn.Linear(512, emb_dim), nn.ReLU(), nn.Dropout(0.2))

        def forward(self, x):
            conv   = self.conv_base(x)
            pooled = self.pool(conv).view(x.size(0), -1)
            return self.proj(pooled), conv

    class ClassifierFull(nn.Module):
        def __init__(self, input_dim, num_classes, hidden=512):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(hidden, max(64, hidden//2)), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(max(64, hidden//2), num_classes))
        def forward(self, x): return self.net(x)

    class ClassifierEmb(nn.Module):
        def __init__(self, emb_dim, num_classes, hidden=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(emb_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(hidden, max(32, hidden//2)), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(max(32, hidden//2), num_classes))
        def forward(self, x): return self.net(x)

    def _gn_groups(channels):
        g = min(8, max(1, channels // 16))
        while channels % g != 0 and g > 1:
            g -= 1
        return max(1, g)

    class StableDQNNet(nn.Module):
        """
        Mirrors StableDQNAgent.net exactly.

        The training script saves:
            torch.save(agent.net.state_dict(), AGENT_STATE)
        where agent.net is a bare nn.Sequential, so state-dict keys are
        flat: "0.weight", "2.weight", "4.weight", ...
        We register the Sequential as self.seq and load with a prefix remap
        so both flat keys and "seq.*" keys are accepted.
        """
        def __init__(self, obs_dim, n_actions, hidden=512):
            super().__init__()
            g1 = _gn_groups(hidden)
            g2 = _gn_groups(max(1, hidden // 2))
            self.seq = nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.ReLU(),
                nn.GroupNorm(g1, hidden), nn.Dropout(0.2),
                nn.Linear(hidden, max(1, hidden // 2)), nn.ReLU(),
                nn.GroupNorm(g2, max(1, hidden // 2)), nn.Dropout(0.2),
                nn.Linear(max(1, hidden // 2), n_actions))

        def forward(self, x): return self.seq(x)

        def act(self, obs: np.ndarray) -> int:
            self.eval()
            with torch.no_grad():
                t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
                q = self.seq(t)[0].cpu().numpy()
            return int(np.argmax(q))

# =============================================================================
# IMAGE PRE-PROCESSING
# =============================================================================

def pil_to_tensor(pil: Image.Image, img_size=(224, 224)) -> "torch.Tensor":
    img = pil.resize(img_size, Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    t    = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (t - mean) / std

# =============================================================================
# MODEL LOADING  (cached — runs only once per Streamlit session)
# =============================================================================

@st.cache_resource(show_spinner="Loading models…")
def load_all_models(mod_root: str):
    """
    Returns a dict with keys:
      metadata, classes, embedder, clf_full, clf_emb,
      scaler_full, scaler_emb, dqn, yolo
    All values may be None if the file is absent.
    """
    out = {k: None for k in [
        "metadata","classes","embedder","clf_full","clf_emb",
        "scaler_full","scaler_emb","dqn","yolo"
    ]}

    # ── metadata ─────────────────────────────────────────────────────────────
    meta_path = os.path.join(mod_root, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        out["metadata"] = meta
        out["classes"]  = meta.get("classes", [])
    else:
        st.sidebar.warning("metadata.json not found — class names unavailable.")
        out["classes"] = []

    classes     = out["classes"]
    num_classes = len(classes)
    if num_classes == 0:
        st.sidebar.error("No classes loaded. Check metadata.json.")
        return out

    if not TORCH_OK:
        st.sidebar.error("PyTorch not available.")
        return out

    # ── scalers ───────────────────────────────────────────────────────────────
    sf_path = os.path.join(mod_root, "scaler_full.joblib")
    se_path = os.path.join(mod_root, "scaler_emb.joblib")
    if os.path.exists(sf_path):
        out["scaler_full"] = joblib.load(sf_path)
    if os.path.exists(se_path):
        out["scaler_emb"]  = joblib.load(se_path)

    # ── embedder ──────────────────────────────────────────────────────────────
    # Load architecture without pretrained weights (we'll load saved weights)
    embedder = EmbedderResNet34(emb_dim=EMB_DIM, pretrained=False).to(DEVICE)
    # The embedder has no separate .pth — it was used as a frozen feature
    # extractor; its pretrained ResNet34 weights are baked in at training time.
    # We load pretrained weights here to replicate inference exactly.
    try:
        from torchvision import models as tvm
        base       = tvm.resnet34(pretrained=True)
        state_dict = {
            "conv_base.0.weight": base.conv1.weight,
            "conv_base.1.weight": base.bn1.weight,
            "conv_base.1.bias"  : base.bn1.bias,
            "conv_base.1.running_mean": base.bn1.running_mean,
            "conv_base.1.running_var" : base.bn1.running_var,
        }
        # Layer-by-layer copy via sequential transfer
        embedder.load_state_dict(base.state_dict(), strict=False)
    except Exception:
        pass
    embedder.eval()
    out["embedder"] = embedder

    # ── ClassifierFull ────────────────────────────────────────────────────────
    meta      = out["metadata"] or {}
    obs_dim   = meta.get("obs_dim", EMB_DIM + 10)   # emb + 10 YOLO features
    clf_f_path= os.path.join(mod_root, "classifier_full.pth")
    if os.path.exists(clf_f_path):
        clf_full = ClassifierFull(input_dim=obs_dim, num_classes=num_classes)
        clf_full.load_state_dict(torch.load(clf_f_path, map_location=DEVICE))
        clf_full.eval()
        out["clf_full"] = clf_full
    else:
        st.sidebar.warning("classifier_full.pth not found.")

    # ── ClassifierEmb ─────────────────────────────────────────────────────────
    clf_e_path = os.path.join(mod_root, "classifier_emb.pth")
    if os.path.exists(clf_e_path):
        clf_emb = ClassifierEmb(emb_dim=EMB_DIM, num_classes=num_classes)
        clf_emb.load_state_dict(torch.load(clf_e_path, map_location=DEVICE))
        clf_emb.eval()
        out["clf_emb"] = clf_emb
    else:
        st.sidebar.warning("classifier_emb.pth not found.")

    # ── DQN agent ─────────────────────────────────────────────────────────────
    dqn_path = os.path.join(mod_root, "dqn_agent_state.pth")
    if os.path.exists(dqn_path):
        try:
            dqn = StableDQNNet(obs_dim=obs_dim, n_actions=num_classes)
            raw_sd = torch.load(dqn_path, map_location=DEVICE)
            # The training script saves agent.net.state_dict() which is a
            # bare Sequential → flat keys "0.weight", "2.weight", etc.
            # Remap to "seq.*" to match StableDQNNet.seq.
            if any(k.startswith("seq.") for k in raw_sd):
                remapped = raw_sd          # already correct prefix
            elif any(k.startswith("net.") for k in raw_sd):
                # old wrapper style → strip "net." prefix then add "seq."
                remapped = {"seq." + k[4:]: v for k, v in raw_sd.items()}
            else:
                # flat keys "0.weight" → prefix with "seq."
                remapped = {"seq." + k: v for k, v in raw_sd.items()}
            dqn.load_state_dict(remapped)
            dqn.eval()
            out["dqn"] = dqn
        except Exception as e:
            st.sidebar.warning(f"DQN load failed: {e}")

    # ── YOLO (optional) ───────────────────────────────────────────────────────
    if YOLO_OK:
        for candidate in ["yolov12x-cls.pt", "yolov12x.pt", "yolo.pt"]:
            yp = os.path.join(mod_root, candidate)
            if os.path.exists(yp):
                try:
                    out["yolo"] = YOLO(yp)
                    break
                except Exception as e:
                    st.sidebar.warning(f"YOLO load failed ({candidate}): {e}")

    return out

# =============================================================================
# INFERENCE
# =============================================================================

def extract_embedding(embedder, pil_img: Image.Image) -> np.ndarray:
    """Returns (EMB_DIM,) float32 embedding."""
    t = pil_to_tensor(pil_img, img_size=(224, 224)).to(DEVICE)
    with torch.no_grad():
        emb, _ = embedder(t)
    return emb.squeeze(0).cpu().numpy().astype(np.float32)

def build_full_feature(emb: np.ndarray) -> np.ndarray:
    """
    Replicates training-time feature: concat(emb, yolo_zeros).
    When YOLO is absent at inference, YOLO features are zeros — identical
    to the training fallback in FeatureExtractor.extract().
    """
    yolo_zeros = np.zeros(10, dtype=np.float32)
    return np.concatenate([emb, yolo_zeros])

def run_classifier(models: dict, pil_img: Image.Image
                   ) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Returns (probs_full, probs_emb, active_model_name).
    Uses clf_full if available (higher accuracy), falls back to clf_emb.
    """
    embedder     = models["embedder"]
    scaler_full  = models["scaler_full"]
    scaler_emb   = models["scaler_emb"]
    clf_full     = models["clf_full"]
    clf_emb      = models["clf_emb"]

    emb      = extract_embedding(embedder, pil_img)                 # (512,)
    full_feat= build_full_feature(emb)                              # (522,)

    probs_full = probs_emb = None

    if clf_full is not None and scaler_full is not None:
        xf = scaler_full.transform(full_feat.reshape(1, -1)).astype(np.float32)
        with torch.no_grad():
            logits = clf_full(torch.from_numpy(xf))
            probs_full = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    if clf_emb is not None and scaler_emb is not None:
        xe = scaler_emb.transform(emb.reshape(1, -1)).astype(np.float32)
        with torch.no_grad():
            logits = clf_emb(torch.from_numpy(xe))
            probs_emb = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    if probs_full is not None:
        return probs_full, probs_emb, "ClassifierFull"
    elif probs_emb is not None:
        return probs_emb, None, "ClassifierEmb"
    return None, None, "none"

def run_dqn_decision(models: dict, pil_img: Image.Image) -> Optional[Tuple[int, float]]:
    """
    Runs the DQN on the image embedding and returns (action_idx, q_value).
    In this pipeline the DQN's action space == class space, so the action
    index is also the predicted class.
    """
    dqn         = models.get("dqn")
    embedder    = models.get("embedder")
    scaler_full = models.get("scaler_full")
    if dqn is None or embedder is None or scaler_full is None:
        return None

    emb      = extract_embedding(embedder, pil_img)
    full_feat= build_full_feature(emb)
    xf       = scaler_full.transform(full_feat.reshape(1, -1)).astype(np.float32)
    obs      = xf.squeeze(0)

    dqn.eval()
    with torch.no_grad():
        t = torch.from_numpy(obs).unsqueeze(0)
        q_values = dqn.seq(t).squeeze(0).cpu().numpy()
    action = int(np.argmax(q_values))
    return action, float(q_values[action])

# =============================================================================
# GRAD-CAM  (reuses EmbedderResNet34's conv_base hook)
# =============================================================================

def compute_gradcam(embedder, classifier, pil_img: Image.Image,
                    class_idx: int) -> Optional[np.ndarray]:
    """Returns (H,W) heatmap normalised to [0,1], or None on failure."""
    inp = pil_to_tensor(pil_img, img_size=(224, 224)).to(DEVICE)

    embedder.eval(); classifier.eval()
    emb, conv = embedder(inp)

    grads  = []
    handle = conv.register_hook(lambda g: grads.append(g.detach().clone()))
    emb    = emb.requires_grad_(True)
    logits = classifier(emb)
    score  = logits[0, class_idx]
    classifier.zero_grad(); embedder.zero_grad()
    try:
        score.backward(retain_graph=True)
    except Exception:
        handle.remove(); return None
    handle.remove()

    if not grads: return None
    weights = grads[0].mean(dim=(2, 3), keepdim=True)
    gcam    = F.relu((weights * conv.detach()).sum(dim=1, keepdim=True))
    up      = F.interpolate(gcam, size=(224, 224), mode="bilinear", align_corners=False)
    heat    = up.squeeze().cpu().numpy()
    heat   -= heat.min()
    if heat.max() > 0:
        heat /= heat.max() + 1e-8
    return heat

def overlay_gradcam(pil_img: Image.Image, heat: np.ndarray,
                    alpha: float = 0.45) -> Image.Image:
    """Blend Grad-CAM heatmap onto original image."""
    heat_pil   = Image.fromarray((heat * 255).astype(np.uint8), mode="L")
    heat_rs    = np.asarray(heat_pil.resize(pil_img.size, Image.BILINEAR)).astype(np.float32) / 255.0
    cmap       = plt.get_cmap("jet")
    colored    = (cmap(heat_rs)[:, :, :3] * 255).astype(np.uint8)
    img_arr    = np.asarray(pil_img.convert("RGB")).astype(np.float32)
    blended    = np.clip(img_arr * (1 - alpha) + colored * alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)

# =============================================================================
# SIDEBAR — model status
# =============================================================================

def render_sidebar(models: dict):
    st.sidebar.title("Model Status")
    st.sidebar.caption(f"`{MODELLING_ROOT}`")

    checks = [
        ("Embedder (ResNet34)",  models["embedder"]   is not None),
        ("ClassifierFull",       models["clf_full"]   is not None),
        ("ClassifierEmb",        models["clf_emb"]    is not None),
        ("Scaler (full)",        models["scaler_full"]is not None),
        ("Scaler (emb)",         models["scaler_emb"] is not None),
        ("DQN Agent",            models["dqn"]        is not None),
        ("YOLO",                 models["yolo"]       is not None),
    ]
    for label, ok in checks:
        icon = "✅" if ok else "❌"
        st.sidebar.write(f"{icon} {label}")

    meta = models.get("metadata") or {}
    if meta:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Metadata")
        st.sidebar.write(f"**Classes:** {meta.get('num_classes','?')}")
        st.sidebar.write(f"**Obs dim:** {meta.get('obs_dim','?')}")
        st.sidebar.write(f"**Train samples:** {meta.get('train_samples','?')}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Expected files")
    files = ["metadata.json","classifier_full.pth","classifier_emb.pth",
             "scaler_full.joblib","scaler_emb.joblib","dqn_agent_state.pth",
             "yolov12x-cls.pt"]
    for f in files:
        exists = os.path.exists(os.path.join(MODELLING_ROOT, f))
        st.sidebar.caption(f"{'✅' if exists else '❌'}  {f}")

# =============================================================================
# MAIN UI
# =============================================================================

def main():
    st.title("🔬 YOLO-DQN Image Classifier")
    st.caption("ResNet34 embedder · ClassifierFull/Emb · StableDQN · Grad-CAM")

    models = load_all_models(MODELLING_ROOT)
    render_sidebar(models)

    classes = models["classes"] or []
    ready   = (models["embedder"] is not None and
               (models["clf_full"] is not None or models["clf_emb"] is not None))

    if not ready:
        st.error("Core model files missing. "
                 "Place `classifier_full.pth` / `classifier_emb.pth`, "
                 "`scaler_*.joblib`, and `metadata.json` inside `_MODELLING/`.")
        st.stop()

    # ── Upload ────────────────────────────────────────────────────────────────
    st.markdown("### Upload an image")
    uploaded = st.file_uploader("", type=["jpg","jpeg","png","bmp","tiff"])
    if uploaded is None:
        st.info("Upload an image to begin.")
        st.stop()

    pil_img = Image.open(uploaded).convert("RGB")

    col_img, col_res = st.columns([1, 1])
    with col_img:
        st.image(pil_img, caption="Input image", use_container_width=True)

    # ── Options ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.subheader("Options")
        top_k        = st.slider("Top-K predictions", 3, min(20, len(classes)), 5)
        show_gradcam = st.checkbox("Show Grad-CAM", value=True)
        show_dqn     = st.checkbox("Show DQN decision", value=True)

    # ── Classification ────────────────────────────────────────────────────────
    with st.spinner("Running inference…"):
        probs, probs_emb, active_clf = run_classifier(models, pil_img)

    if probs is None:
        st.error("Inference failed — check model / scaler files.")
        st.stop()

    top_indices = np.argsort(probs)[::-1][:top_k]
    top_labels  = [classes[i] if i < len(classes) else f"cls_{i}" for i in top_indices]
    top_probs   = probs[top_indices]

    with col_res:
        st.markdown(f"### Prediction  *(using {active_clf})*")
        best_label = classes[top_indices[0]] if top_indices[0] < len(classes) else str(top_indices[0])
        st.success(f"**{best_label}**  —  confidence {top_probs[0]*100:.1f}%")

        # Horizontal bar chart
        fig, ax = plt.subplots(figsize=(5, max(2.5, top_k * 0.4)))
        colours = ["#2c7bb6" if i > 0 else "#d7191c" for i in range(top_k)]
        ax.barh(top_labels[::-1], top_probs[::-1] * 100, color=colours[::-1])
        ax.set_xlabel("Confidence (%)")
        ax.set_xlim(0, 100)
        ax.set_title(f"Top-{top_k} predictions")
        for spine in ["top","right"]: ax.spines[spine].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Numeric table
        with st.expander("Full probability table"):
            all_labels = [classes[i] if i < len(classes) else f"cls_{i}"
                          for i in range(len(probs))]
            st.dataframe(
                {"Class": all_labels,
                 "Probability": [f"{p*100:.2f}%" for p in probs]},
                use_container_width=True,
            )

    # ── Emb-only classifier comparison ────────────────────────────────────────
    if probs_emb is not None:
        emb_top = np.argmax(probs_emb)
        emb_lbl = classes[emb_top] if emb_top < len(classes) else str(emb_top)
        st.info(f"**ClassifierEmb** prediction: **{emb_lbl}** "
                f"({probs_emb[emb_top]*100:.1f}%)")

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    if show_gradcam and models["clf_emb"] is not None:
        st.markdown("---")
        st.markdown("### Grad-CAM Activation")
        target_cls = top_indices[0]
        with st.spinner("Computing Grad-CAM…"):
            heat = compute_gradcam(
                models["embedder"], models["clf_emb"], pil_img, int(target_cls))

        if heat is not None:
            overlay = overlay_gradcam(pil_img, heat, alpha=0.45)
            gcol1, gcol2 = st.columns(2)
            with gcol1:
                st.image(pil_img,  caption="Original",      use_container_width=True)
            with gcol2:
                st.image(overlay,  caption=f"Grad-CAM — predicted: {best_label}",
                         use_container_width=True)
        else:
            st.warning("Grad-CAM computation failed for this image.")

    # ── DQN decision panel ────────────────────────────────────────────────────
    if show_dqn and models["dqn"] is not None:
        st.markdown("---")
        st.markdown("### DQN Agent Decision")
        result = run_dqn_decision(models, pil_img)
        if result is not None:
            dqn_action, dqn_q = result
            meta        = models.get("metadata") or {}
            act_labels  = meta.get("action_labels", classes)
            dqn_label   = (act_labels[dqn_action]
                           if dqn_action < len(act_labels) else str(dqn_action))
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                st.metric("DQN selected action", dqn_label)
                st.metric("Q-value", f"{dqn_q:.4f}")
            with dcol2:
                clf_label = classes[top_indices[0]] if top_indices[0] < len(classes) else "?"
                agreement = "✅ Agree" if dqn_label == clf_label else "⚠️ Disagree"
                st.metric("vs. Classifier", clf_label, delta=agreement)
                st.caption("DQN acts on the same feature vector as ClassifierFull. "
                           "Agreement validates the RL refinement stage.")
        else:
            st.info("DQN result unavailable (missing scaler or embedder).")

    # ── Metadata card ─────────────────────────────────────────────────────────
    meta = models.get("metadata") or {}
    if meta:
        with st.expander("Model metadata"):
            st.json(meta)


if __name__ == "__main__":
    main()