import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import io  # Added for checkpoint loading
import time
from typing import List


# =========================================================
# 1. Page configuration & global constants
# =========================================================

st.set_page_config(
    page_title="Construction Material Temperature Classifier",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Temperature classes & palette (colour‑blind‑friendly)
TEMP_CLASSES: List[str] = ["200", "400", "600", "800"]
TEMP_COLOURS: List[str] = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]  # blue/green/orange/red

# Vision‑Transformer hyper‑parameters
IMG_SIZE = 460  # Input resolution fed to ViT
PATCH_SIZE = 8
EMBED_DIM = 768
ENC_LAYERS = 12
HEADS = 12
NUM_CLASSES = len(TEMP_CLASSES)

# =========================================================
# 2. Header & about section
# =========================================================

st.title("🏗️ ThermoSight : Construction Materials Temperature Classifier 🔬")
with st.container():
    st.markdown("### Advanced thermal response analysis for civil engineering materials")
    with st.expander("ℹ️ About this application"):
        st.write(
            """
            **ThermoSight** uses a Vision Transformer (ViT) to classify construction materials by
            the temperature at which they were imaged.

            Supported temperature classes:
            • **200 °C** – Low thermal exposure  
            • **400 °C** – Medium thermal exposure  
            • **600 °C** – High thermal exposure  
            • **800 °C** – Extreme thermal exposure
            """
        )

# =========================================================
# 3. Vision Transformer definition
# =========================================================

class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim: int, patch_size: int, num_patches: int, dropout: float, in_channels: int):
        super().__init__()
        self.patcher = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.positional_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = self.patcher(x).permute(0, 2, 1)  # (B, N, C)
        x = torch.cat([cls_tokens, x], dim=1)  # prepend CLS
        x = x + self.positional_embeddings[:, : x.size(1), :]
        return self.dropout(x)


class ViT(nn.Module):
    def __init__(
        self,
        img_size: int = IMG_SIZE,
        patch_size: int = PATCH_SIZE,
        embed_dim: int = EMBED_DIM,
        enc_layers: int = ENC_LAYERS,
        n_heads: int = HEADS,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.0,
        in_channels: int = 3,
    ):
        super().__init__()
        n_patches = (img_size // patch_size) ** 2
        self.embed = PatchEmbedding(embed_dim, patch_size, n_patches, dropout, in_channels)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=enc_layers)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.transformer(x)
        return self.mlp_head(x[:, 0])  # CLS token

# =========================================================
# 4. Sidebar – Controls & inputs
# =========================================================

st.sidebar.header("📊 Analysis Control Panel")

# --- Checkpoint selection (upload *or* path) ---
checkpoint_source = st.sidebar.radio(
    "Choose model source",
    options=["Upload .pth file", "Local path"],
    horizontal=True,
)

ckpt_buffer = None
ckpt_path = None

if checkpoint_source == "Upload .pth file":
    ckpt_file = st.sidebar.file_uploader("Upload ViT checkpoint", type=["pth"])
    if ckpt_file is not None:
        ckpt_buffer = ckpt_file.read()  # bytes
        st.sidebar.success("Checkpoint uploaded ✅")
else:
    temp_path = st.sidebar.text_input("Enter absolute path to checkpoint", value="/path/to/model.pth")
    if temp_path and not os.path.exists(temp_path):
        st.sidebar.error("Checkpoint path not found ⚠️")
    else:
        ckpt_path = temp_path if temp_path else None

# --- Image upload ---
st.sidebar.divider()
st.sidebar.subheader("📸 Material Image")
image_file = st.sidebar.file_uploader("Upload material image", type=["jpg", "jpeg", "png", "bmp"])
if image_file is None:
    st.sidebar.info("Upload an image to enable analysis.")

# --- Temperature legend ---
st.sidebar.divider()
st.sidebar.subheader("🌡️ Temperature Legend")
legend_cols = st.sidebar.columns(len(TEMP_CLASSES))
for col, t, c in zip(legend_cols, TEMP_CLASSES, TEMP_COLOURS):
    with col:
        st.color_picker(f"{t}°C", c, disabled=True, key=f"legend_{t}")
        st.caption(f"{t}°C")

# --- Run button ---
st.sidebar.divider()
run = st.sidebar.button("🔬 Analyze Material", type="primary")

# =========================================================
# 5. Pre‑processing utilities & model cache
# =========================================================

@st.cache_resource(show_spinner=False)
def get_transform(img_size: int = IMG_SIZE):
    return transforms.Compose(
        [
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.Lambda(lambda im: im.convert("RGBA").convert("RGB") if im.mode in ("P", "RGBA") else im.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _load_state_dict(buf_or_path, device):
    """Load state dict from bytes or file path."""
    if isinstance(buf_or_path, (bytes, bytearray, memoryview)):
        return torch.load(io.BytesIO(buf_or_path), map_location=device)
    return torch.load(buf_or_path, map_location=device)


@st.cache_resource(show_spinner=False)
def load_model(buf_or_path, device):
    model = ViT().to(device)
    raw_state = _load_state_dict(buf_or_path, device)
    state = raw_state.get("state_dict", raw_state)  # handle lightning checkpoints etc.
    # lightning adds "_orig_mod." prefix when using `torch.compile`; strip it
    cleaned_state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned_state, strict=False)
    model.eval()
    return model

# =========================================================
# 6. Main workflow
# =========================================================

main_col1, main_col2 = st.columns([3, 2])

if run:
    if (ckpt_buffer is None and ckpt_path is None) or image_file is None:
        st.sidebar.error("Please provide **both** a valid checkpoint and an image before running.")
        st.stop()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Animated phase placeholders ---
    phase_box = st.empty()
    prog_bar = st.sidebar.progress(0, text="Preparing…")

    # Phase 1 – load model --------------------------------------------------
    phase_box.write("**Phase 1/3 – Loading model**")
    prog_bar.progress(20)
    ckpt_source = ckpt_buffer if ckpt_buffer is not None else ckpt_path
    model = load_model(ckpt_source, device)
    prog_bar.progress(40)

    # Phase 2 – preprocess image -------------------------------------------
    phase_box.write("**Phase 2/3 – Pre‑processing image**")
    img = Image.open(image_file)
    x = get_transform()(img).unsqueeze(0).to(device)
    prog_bar.progress(60)

    # Phase 3 – inference ---------------------------------------------------
    phase_box.write("**Phase 3/3 – Running inference**")
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    prog_bar.progress(100)
    time.sleep(0.3)
    prog_bar.empty()
    phase_box.empty()

    # -------------------------------------------------- Display results
    pred_idx = torch.argmax(probs).item()
    pred_temp = TEMP_CLASSES[pred_idx]
    confidence = probs[pred_idx].item() * 100

    with main_col1:
        st.image(img, caption="Construction Material Sample", use_column_width=True)

    with main_col2:
        st.success(f"**Predicted Temperature : {pred_temp} °C**")
        st.metric("Confidence", f"{confidence:.1f}%")
        desc = {
            "200": "Low thermal exposure (200 °C) – early‑stage thermal effects.",
            "400": "Medium exposure (400 °C) – moderate thermal degradation.",
            "600": "High exposure (600 °C) – significant thermal damage.",
            "800": "Extreme exposure (800 °C) – severe degradation/phase change.",
        }
        st.info(desc[pred_temp])

    # Bar‑chart of class probabilities -------------------------------------
    st.divider()
    st.subheader("Temperature Class Probability Distribution")
    chart_data = {
        "Temperature": [f"{t} °C" for t in TEMP_CLASSES],
        "Probability (%)": [probs[i].item() * 100 for i in range(NUM_CLASSES)],
    }
    st.bar_chart(chart_data, x="Temperature", y="Probability (%)")

else:
    # ----------------------------- Help / Instructions (collapsed by default)
    with main_col1:
        st.header("🔍 How It Works")
        tab1, tab2 = st.tabs(["Instructions", "Applications"])
        with tab1:
            st.write("### Step‑by‑Step Guide:")
            st.write("1️⃣ Provide a ViT checkpoint (upload or local path).")
            st.write("2️⃣ Upload a construction material image.")
            st.write("3️⃣ Click **Analyze Material** to obtain the predicted temperature class.")
        with tab2:
            st.write("### Example Applications:")
            col_l, col_r = st.columns(2)
            with col_l:
                st.success("🏢 Post‑fire structural assessment")
                st.success("✅ On‑site quality control")
                st.success("🔬 Material‑science research")
            with col_r:
                st.success("🎓 Educational demonstrations")
                st.success("🛡️ Disaster recovery planning")

    with main_col2:
        st.header("📊 Model Details")
        st.info(
            "Vision Transformer with 12‑layer encoder, 768‑d embeddings, "
            "and 8×8 patching on 460×460 images."
        )
        st.write(
            "The model leverages multi‑head self‑attention to capture both local and global "
            "features of thermal patterns, enabling robust classification across diverse "
            "construction materials."
        )

        if "first_visit" not in st.session_state:
            st.session_state.first_visit = True
            st.balloons()
