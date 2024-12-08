import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
from model import load_model

st.markdown("""
    <style>
    .stApp {
        background-color: #2c3e50;
        color: #ecf0f1;
    }
    
    /* Grouped container styling */
    .stMarkdown, .stUploader, .css-1y4p8pa {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    /* Group elements in sections */
    .element-container {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Section styling */
    .section-container {
        background-color: #34495e;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px !important;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #f39c12 !important;
        color: white !important;
        border: none !important;
        padding: 10px 20px !important;
        border-radius: 5px !important;
    }
    
    .stButton > button:hover {
        background-color: #e67e22 !important;
    }
    
    /* Image styling */
    div[data-testid="stImage"] {
        width: 350px !important;
        margin: auto !important;
    }
    
    div[data-testid="stImage"] img {
        max-width: 350px !important;
        max-height: 250px !important;
        object-fit: contain !important;
        border-radius: 10px !important;
        border: 5px solid #3498db !important;
    }
    </style>
""", unsafe_allow_html=True)

# Model setup (same as before)
@st.cache_resource
def get_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(device=device)
    model.to(device)
    return model, device

model, device = get_model()

# Transform pipeline (same as before)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((128, 128)),
    transforms.ColorJitter(contrast=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def predict_image(image):
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    
    return 'Tumor' if predicted.item() == 1 else 'No Tumor'

# Main app with grouped sections
st.title("Brain Tumor Classification")

# Upload section grouped together
with st.container():
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown("### Upload Image")
    uploaded_file = st.file_uploader("Choose a brain scan image", type=['png', 'jpg', 'jpeg'])
    st.markdown('</div>', unsafe_allow_html=True)

# Sample images section grouped together
with st.container():
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown("### Sample Images for Testing")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Tumor Samples")
        for i in range(1, 4):
            try:
                with open(f'static/sample_images/tumor_{i}.jpg', 'rb') as f:
                    st.download_button(
                        label=f'Sample Tumor Case {i}',
                        data=f,
                        file_name=f'tumor_{i}.jpg',
                        mime='image/jpeg',
                        key=f'tumor_{i}'
                    )
            except FileNotFoundError:
                pass
    
    with col2:
        st.markdown("#### Non-Tumor Samples")
        for i in range(1, 4):
            try:
                with open(f'static/sample_images/no_tumor_{i}.jpg', 'rb') as f:
                    st.download_button(
                        label=f'Sample Normal Case {i}',
                        data=f,
                        file_name=f'no_tumor_{i}.jpg',
                        mime='image/jpeg',
                        key=f'no_tumor_{i}'
                    )
            except FileNotFoundError:
                pass
    st.markdown('</div>', unsafe_allow_html=True)

# Prediction section grouped together
if uploaded_file is not None:
    with st.container():
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        image = Image.open(uploaded_file)
        st.image(image, 
                caption='Uploaded Image',
                use_column_width=False,
                width=350)
        
        prediction = predict_image(image)
        st.markdown("### Prediction Result:")
        st.markdown(f"<h2 style='color: #3498db;'>{prediction}</h2>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)