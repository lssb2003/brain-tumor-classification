import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
from model import load_model

# Page setup
st.set_page_config(page_title="Brain Tumor Classification", layout="centered")

# Ensure upload folder exists
UPLOAD_FOLDER = 'static/uploaded_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load model (keep exactly as in Flask app)
@st.cache_resource
def get_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(device=device)
    model.to(device)
    return model, device

model, device = get_model()

# Keep the exact same transform as your Flask app
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((128, 128)),
    transforms.ColorJitter(contrast=0.5),  # Adjust contrast by a factor of 0.5
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Predict function (kept exactly the same)
def predict_image(image):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    
    return 'Tumor' if predicted.item() == 1 else 'No Tumor'

# Main app
st.title("Brain Tumor Classification")

# File upload section (equivalent to your Flask form)
uploaded_file = st.file_uploader("Upload a brain scan image", type=['png', 'jpg', 'jpeg'])

# Sample downloads section
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
                    mime='image/jpeg'
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
                    mime='image/jpeg'
                )
        except FileNotFoundError:
            pass

# Handle prediction (equivalent to your Flask predict route)
if uploaded_file is not None:
    try:
        # Load and display the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Make prediction
        prediction = predict_image(image)
        
        # Display result
        st.markdown("### Prediction Result:")
        st.markdown(f"<h2 style='color: #3498db;'>{prediction}</h2>", unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")