import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
from model import load_model

# Custom CSS to match your original Flask app exactly
st.markdown("""
    <style>
    /* Global Styles */
    .stApp {
        font-family: 'Arial', sans-serif;
        background-color: #1c1c1c;
        color: #ecf0f1;
    }
    
    /* Container styling */
    .element-container {
        background-color: #2c3e50;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3);
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* Header styling */
    h1 {
        font-size: 2.5em;
        color: #3498db !important;
        margin-bottom: 20px;
    }
    
    /* File uploader styling */
    .stFileUploader {
        padding: 10px;
        background-color: #34495e;
        border: none;
        border-radius: 5px;
        color: white;
        margin: 10px 0;
    }
    
    /* Button styling */
    .stButton > button {
        padding: 10px 20px;
        background-color: #f39c12 !important;
        border: none;
        border-radius: 5px;
        color: white;
        cursor: pointer;
        font-size: 1.1em;
        margin-top: 10px;
        transition: background-color 0.3s, transform 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #e67e22 !important;
        transform: scale(1.05);
    }
    
    /* Results styling */
    h2 {
        color: #e74c3c !important;
    }
    
    .prediction-text {
        font-size: 1.2em;
        color: #3498db;
        font-weight: bold;
    }
    
    /* Image styling */
    .stImage {
        max-width: 500px;
        border-radius: 10px;
        border: 5px solid #3498db;
    }
    
    /* Download buttons styling */
    .download-buttons {
        display: grid;
        gap: 10px;
    }
    
    /* Two column layout */
    .stColumns {
        background-color: #34495e;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    
    /* Sample section headers */
    h3 {
        color: #3498db !important;
        margin-bottom: 15px;
    }
    
    </style>
""", unsafe_allow_html=True)

# Rest of your Streamlit app code remains the same
@st.cache_resource
def get_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(device=device)
    model.to(device)
    return model, device

model, device = get_model()

# Keep the exact same transform
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

# Main app with styled container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.title("Brain Tumor Classification")

uploaded_file = st.file_uploader("Upload a brain scan image", type=['png', 'jpg', 'jpeg'])

# Sample downloads in a styled container
st.markdown('<div class="samples-container">', unsafe_allow_html=True)
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

# Prediction section
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        prediction = predict_image(image)
        
        st.markdown("### Prediction Result:")
        st.markdown(f'<p class="prediction-text">{prediction}</p>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

st.markdown('</div>', unsafe_allow_html=True)
