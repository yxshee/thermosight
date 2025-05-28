import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import time

# ---------------------------------------------------------
# 1. Page Configuration and Theme
# ---------------------------------------------------------
# Set page config for a wide layout and custom title/icon
st.set_page_config(
    page_title="Construction Material Temperature Classifier",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. App Header with Streamlit Components
# ---------------------------------------------------------
# Main title with emojis for visual appeal
st.title("🏗️ Thermosight : Construction Materials Temperature Classifier 🔬")

# Create a colored header with expander for information
with st.container():
    st.markdown("### Advanced thermal response analysis for civil engineering materials")
    
    # Use expander for additional information
    with st.expander("ℹ️ About this application"):
        st.write("""
        This application uses deep learning to classify construction materials based on their thermal signatures.
        Upload an image taken at a specific temperature to determine the material class.
        
        The classifier can identify materials tested at five different temperature ranges:
        - 27°C (room temperature)
        - 200°C (low thermal exposure)
        - 400°C (medium thermal exposure)
        - 600°C (high thermal exposure)
        - 800°C (extreme thermal exposure)
        """)

# ---------------------------------------------------------
# 3. Model Architecture (unchanged)
# ---------------------------------------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels):
        super().__init__()
        self.patcher = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2)
        )
        self.cls_token = nn.Parameter(torch.randn(size=(1, 1, embed_dim)), requires_grad=True)
        self.positional_embeddings = nn.Parameter(torch.randn(size=(1, num_patches + 1, embed_dim)), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = self.patcher(x).permute(0, 2, 1)
        x = torch.cat([cls_token, x], dim=1)

        n_tokens = x.size(1)
        n_orig = self.positional_embeddings.size(1)
        if n_tokens != n_orig:
            cls_pos = self.positional_embeddings[:, :1, :]
            spatial_pos = self.positional_embeddings[:, 1:, :]
            gs_old = int(math.sqrt(spatial_pos.size(1)))
            gs_new = int(math.sqrt(n_tokens - 1))
            spatial_pos = spatial_pos.transpose(1, 2).view(1, -1, gs_old, gs_old)
            new_sp = F.interpolate(spatial_pos, size=(gs_new, gs_new), mode='bilinear', align_corners=False)
            new_sp = new_sp.flatten(2).transpose(1, 2)
            pos_emb = torch.cat((cls_pos, new_sp), dim=1)
        else:
            pos_emb = self.positional_embeddings

        x = x + pos_emb
        x = self.dropout(x)
        return x

class ViT(nn.Module):
    def __init__(self, num_patches, img_size, num_classes, patch_size, embed_dim, num_encoders, num_heads, dropout, activation, in_channels):
        super().__init__()
        self.embeddings_block = PatchEmbedding(embed_dim, patch_size, num_patches, dropout, in_channels)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4*embed_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True
        )
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=num_classes)
        )

    def forward(self, x):
        x = self.embeddings_block(x)
        x = self.encoder_blocks(x)
        x = self.mlp_head(x[:, 0, :])
        return x


# ---------------------------------------------------------
# 4. Sidebar - Using Native Streamlit Components
# ---------------------------------------------------------
# Create a visually appealing sidebar with emoji headers
st.sidebar.title("📊 Analysis Control Panel")

# Add a separator
st.sidebar.divider()

# Model selection section with native components
st.sidebar.subheader("🧠 Model Configuration")
model_file = st.sidebar.file_uploader("Upload Model Checkpoint (.pth)", type=["pth"])

if not model_file:
    st.sidebar.info("Upload the trained model checkpoint to enable material classification.")

# Temperature classes with Streamlit's color display
st.sidebar.divider()
st.sidebar.subheader("🌡️ Temperature Classes")

# Your known classes
classes = ['200', '400', '600', '800']

# Create a visual legend with native components
# Define temperature colors
colors = ['#2962FF', '#00C853', '#FFAB00', '#FF6D00', '#D50000']

# Create a temperature legend using columns
legend_cols = st.sidebar.columns(len(classes))
for i, (col, cls, color) in enumerate(zip(legend_cols, classes, colors)):
    with col:
        st.color_picker(f"{cls}°C", color, disabled=True, key=f"color_{cls}")
        st.write(f"{cls}°C")

st.sidebar.caption("These temperatures represent the thermal conditions at which construction materials were tested.")

# Add another separator
st.sidebar.divider()

# Image upload section
st.sidebar.subheader("📸 Material Image")
uploaded_image = st.sidebar.file_uploader("Upload material image for analysis", type=["jpg", "jpeg", "png","bmp"])

if not uploaded_image:
    st.sidebar.info("Upload an image of construction material for thermal classification.")

# Define test-time transforms
input_size = 128
test_transform = transforms.Compose([
    transforms.Resize(input_size + 10),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------------------------------------------------
# 5. Enhanced Inference & Display with Streamlit Components
# ---------------------------------------------------------
# Use large, visually appealing button
run_col1, run_col2 = st.sidebar.columns([1, 3])
with run_col2:
    run_button = st.button("🔬 Analyze Material", type="primary", use_container_width=True)

# Create 2 columns for the main content
main_col1, main_col2 = st.columns([3, 2])

if run_button:
    if model_file is None:
        st.sidebar.error("⚠️ Please upload a model checkpoint file.")
    elif uploaded_image is None:
        st.sidebar.error("⚠️ Please upload a material image for analysis.")
    else:
        # Show progress with native spinner and progress bar
        with st.status("Analyzing construction material...", expanded=True) as status:
            # Create a progress bar
            progress_bar = st.progress(0)
            
            # Step 1: Load model
            progress_bar.progress(10)
            st.write("🔍 Loading model architecture...")
            time.sleep(0.3)
            
            # Device selection
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Initialize model for 5 classes
            model = ViT(num_patches=16, img_size=input_size, num_classes=len(classes), patch_size=4, embed_dim=128, num_encoders=4, num_heads=4, dropout=0.1, activation='relu', in_channels=3).to(device)
            
            # Step 2: Load weights
            progress_bar.progress(30)
            st.write("⚙️ Loading model parameters...")
            time.sleep(0.3)
            
            # Save the uploaded model to a temporary file and load it
            temp_model_path = "temp_model.pth"
            with open(temp_model_path, "wb") as f:
                f.write(model_file.getbuffer())
            checkpoint = torch.load(temp_model_path, map_location=device)
            model.load_state_dict(checkpoint)
            model.eval()
            os.remove(temp_model_path)
            
            # Step 3: Process image
            progress_bar.progress(60)
            st.write("🖼️ Processing material image...")
            time.sleep(0.3)
            
            # Preprocess the uploaded image
            image = Image.open(uploaded_image).convert("RGB")
            input_tensor = test_transform(image).unsqueeze(0).to(device)
            
            # Step 4: Run inference
            progress_bar.progress(80)
            st.write("🧮 Running material analysis...")
            time.sleep(0.3)
            
            # Perform inference
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                _, predicted = torch.max(outputs, 1)
                predicted_class = classes[predicted.item()]
                confidence = probabilities[predicted].item() * 100
                
            # Complete progress
            progress_bar.progress(100)
            time.sleep(0.2)
            status.update(label="Analysis complete!", state="complete")

        # Display results using Streamlit cards and columns
        with main_col1:
            # Image display with colored border based on temperature
            color_idx = classes.index(predicted_class)
            temperature_color = colors[color_idx]
            
            # Apply a colored border to match the temperature
            st.image(image, caption=f"Construction Material Sample", use_column_width=True)
        
        with main_col2:
            # Create a success message with large text
            st.success(f"Material Temperature Class: {predicted_class}°C")
            
            # Show confidence with a metric and matching color
            st.metric("Analysis Confidence", f"{confidence:.1f}%")
            
            # Add material description based on temperature
            descriptions = {
                '200': "Low thermal exposure (200°C): Shows early-stage thermal effects.",
                '27': "Room temperature (27°C): No thermal exposure effects visible.",
                '400': "Medium thermal exposure (400°C): Moderate thermal degradation visible.",
                '600': "High thermal exposure (600°C): Significant thermal damage present.",
                '800': "Extreme thermal exposure (800°C): Severe thermal degradation and structural changes."
            }
            
            st.info(descriptions[predicted_class])
                
        # Display temperature class distribution below
        st.divider()
        st.subheader("Temperature Class Distribution")
        
        # Create a horizontal bar chart for probabilities
        probs_dict = {f"{cls}°C": probabilities[i].item() * 100 for i, cls in enumerate(classes)}
        
        # Convert to proper format for Streamlit chart
        chart_data = []
        for cls, prob in probs_dict.items():
            chart_data.append({"Temperature": cls, "Probability (%)": prob})
        
        # Use Streamlit's native bar chart
        st.bar_chart(
            chart_data, 
            x="Temperature", 
            y="Probability (%)", 
            color="Temperature"
        )

# ---------------------------------------------------------
# 6. Add Help Section when nothing is running
# ---------------------------------------------------------
if not run_button or (model_file is None or uploaded_image is None):
    # Display help information in the main panel with Streamlit components
    with main_col1:
        st.header("🔍 How It Works")
        
        # Use streamlit tabs for better organization
        tab1, tab2 = st.tabs(["Instructions", "Applications"])
        
        with tab1:
            # Numbered instructions with emojis
            st.write("### Step-by-Step Guide:")
            st.write("1️⃣ Upload a trained model checkpoint (.pth file) in the sidebar")
            st.write("2️⃣ Upload an image of construction material to analyze")
            st.write("3️⃣ Click 'Analyze Material' to identify the temperature class")
            
            # Add example material images
            st.write("### Temperature Classifications:")
            temp_cols = st.columns(5)
            with temp_cols[0]:
                st.write("**27°C**")
                st.caption("Room temperature")
            with temp_cols[1]:
                st.write("**200°C**")
                st.caption("Low exposure")
            with temp_cols[2]:
                st.write("**400°C**")
                st.caption("Medium exposure")
            with temp_cols[3]:
                st.write("**600°C**")
                st.caption("High exposure")
            with temp_cols[4]:
                st.write("**800°C**")
                st.caption("Extreme exposure")
    
        with tab2:
            # Create applications list with emojis using columns
            st.write("### Civil Engineering Applications:")
            
            app_col1, app_col2 = st.columns(2)
            
            with app_col1:
                st.success("🏢 Structural safety assessment")
                st.success("🔥 Post-fire material analysis")
                st.success("✅ Quality control in construction")
            
            with app_col2:
                st.success("🔬 Research in material science")
                st.success("🎓 Civil engineering education")
                st.success("🛡️ Disaster assessment and recovery")

    with main_col2:
        # Add a model demonstration figure or animation
        st.header("📊 Material Analysis")
        
        # Create sample data visualization
        st.write("### How the CNN works")
        st.info("""
        The Construction Materials Classifier uses a Deep Convolutional Neural Network (CNN) with:
        
        - 4 convolutional blocks with increasing filter sizes
        - Batch normalization for stable training
        - Dropout layers to prevent overfitting
        - Adaptive pooling to handle various image sizes
        """)
        
        # Use st.balloons() for a fun effect when the user first visits
        if 'first_visit' not in st.session_state:
            st.session_state.first_visit = True
            st.balloons()
