import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
from model import EnhancedBrainTumorCNN  # your model class

# Page config
st.set_page_config(page_title="Brain Tumor Classification", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #1c1c1c;
        color: #ecf0f1;
    }
    .uploadedFile {
        background-color: #34495e !important;
    }
    .stButton>button {
        background-color: #f39c12;
        color: white;
    }
    .stButton>button:hover {
        background-color: #e67e22;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model = EnhancedBrainTumorCNN()
    model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Image transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Prediction function
def predict_image(image, model):
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)
    return 'Tumor Detected' if predicted.item() == 1 else 'No Tumor Detected'

# Main app
def main():
    st.title("Brain Tumor Classification")
    
    # Load model
    model = load_model()
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a brain scan image", type=['jpg', 'jpeg', 'png'])
    
    # Sample downloads
    st.markdown("### Sample Images for Testing")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Tumor Samples")
        for i in range(1, 4):
            with open(f'static/sample_images/tumor_{i}.jpg', 'rb') as f:
                st.download_button(
                    label=f'Download Tumor Sample {i}',
                    data=f,
                    file_name=f'tumor_{i}.jpg',
                    mime='image/jpeg'
                )
    
    with col2:
        st.markdown("#### Non-Tumor Samples")
        for i in range(1, 4):
            with open(f'static/sample_images/no_tumor_{i}.jpg', 'rb') as f:
                st.download_button(
                    label=f'Download Normal Sample {i}',
                    data=f,
                    file_name=f'no_tumor_{i}.jpg',
                    mime='image/jpeg'
                )
    
    # Make prediction when image is uploaded
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Make prediction
        prediction = predict_image(image, model)
        
        # Display result
        st.markdown("### Prediction Result:")
        st.markdown(f"<h2 style='color: #3498db;'>{prediction}</h2>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
