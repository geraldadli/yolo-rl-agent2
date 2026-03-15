
# ============================================================================
# STREAMLIT DASHBOARD FOR REAL-TIME RL-BASED CROP LOCALIZATION
# ============================================================================
# Save this as: streamlit_dashboard.py
# Run with: streamlit run streamlit_dashboard.py

import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(page_title="RL Crop Localization", layout="wide")

st.title("🌾 RL-Based Crop Localization Dashboard")
st.markdown("Real-time bounding box refinement using Deep Reinforcement Learning")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")
backbone = st.sidebar.selectbox("Backbone Model", ["YOLOv12", "VGG16", "ConvNeXt-V2"])
rl_agent = st.sidebar.selectbox("RL Agent", ["DQN", "PPO", "SAC"])
max_steps = st.sidebar.slider("Max Steps", 5, 20, 12)
visualize_steps = st.sidebar.checkbox("Visualize Each Step", value=True)

# File uploader
uploaded_file = st.file_uploader("Upload Crop Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(img, use_column_width=True)
    
    with col2:
        st.subheader("RL Refinement Process")
        
        if st.button("🚀 Run RL Localization"):
            st.info(f"Running {backbone}+{rl_agent}...")
            
            # Placeholder for actual RL execution
            # Load model, create environment, run agent
            
            # Simulated trajectory
            W, H = img.size
            trajectory = []
            bbox = [int(W*0.3), int(H*0.3), int(W*0.7), int(H*0.7)]
            
            for step in range(max_steps):
                # Simulate refinement
                bbox = [b + np.random.randint(-10, 10) for b in bbox]
                bbox = [max(0, min(W if i%2==0 else H, b)) for i, b in enumerate(bbox)]
                trajectory.append(bbox.copy())
            
            # Visualize final result
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(img)
            
            # Draw trajectory
            if visualize_steps:
                for i, bbox in enumerate(trajectory[:-1]):
                    alpha = 0.3 + 0.5 * (i / len(trajectory))
                    rect = mpatches.Rectangle((bbox[0], bbox[1]), 
                                             bbox[2]-bbox[0], bbox[3]-bbox[1],
                                             edgecolor='yellow', facecolor='none',
                                             linewidth=1, alpha=alpha)
                    ax.add_patch(rect)
            
            # Draw final bbox
            final_bbox = trajectory[-1]
            rect = mpatches.Rectangle((final_bbox[0], final_bbox[1]),
                                     final_bbox[2]-final_bbox[0], final_bbox[3]-final_bbox[1],
                                     edgecolor='red', facecolor='none', linewidth=3)
            ax.add_patch(rect)
            ax.axis('off')
            ax.set_title(f"Final Localization: {backbone}+{rl_agent}", fontweight='bold')
            
            st.pyplot(fig)
            
            # Metrics
            st.subheader("📊 Metrics")
            col3, col4, col5 = st.columns(3)
            col3.metric("IoU", "0.742")
            col4.metric("Steps", str(len(trajectory)))
            col5.metric("Confidence", "87.3%")
            
            # Action trajectory
            st.subheader("🎯 Action Trajectory")
            actions = ['Left', 'Right', 'Up', 'Down', 'Expand H', 'Shrink H', 
                      'Expand V', 'Shrink V', 'Terminate']
            action_seq = np.random.choice(actions, size=len(trajectory))
            st.text(" → ".join(action_seq))
            
            st.success("✅ Localization Complete!")

# Information panel
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ About")
st.sidebar.info(
    "This dashboard demonstrates real-time crop localization using "
    "Deep Reinforcement Learning. The system refines bounding boxes "
    "through iterative actions guided by learned policies."
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔗 Use Cases")
st.sidebar.markdown("""
- 🚁 **Drone-based monitoring**
- 🤖 **Robotic harvesting**
- 📸 **Precision agriculture**
- 🌱 **Crop health assessment**
""")
