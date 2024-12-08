import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
from model import load_model

# Basic styling to match dark theme
st.markdown("""
    <style>
    .stApp {
        background-color: #1c1c1c;
        color: #ecf0f1;
    }
    
    .stButton > button {
        background-color: #f39c12;
        color: white;
    }
    
    .stButton > button:hover {
        background-color: #e67e22;
    }
    </style>
""", unsafe_allow_html=True)

# Model setup
@st.cache_resource
def get_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(device=device)
    model.to(device)
    return model, device

model, device = get_model()

# Transform pipeline
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

# Main app
st.title("Brain Tumor Classification")

# File uploader
uploaded_file = st.file_uploader("Upload a brain scan image", type=['png', 'jpg', 'jpeg'])

# Sample images section
st.subheader("Sample Images for Testing")
col1, col2 = st.columns(2)

# Sample images paths
sample_images = {
    'Tumor Samples': [
        ('tumor_1.jpg', 'Sample Tumor Case 1'),
        ('tumor_2.jpg', 'Sample Tumor Case 2'),
        ('tumor_3.jpg', 'Sample Tumor Case 3')
    ],
    'Non-Tumor Samples': [
        ('no_tumor_1.jpg', 'Sample Normal Case 1'),
        ('no_tumor_2.jpg', 'Sample Normal Case 2'),
        ('no_tumor_3.jpg', 'Sample Normal Case 3')
    ]
}

# Display tumor samples
with col1:
    st.markdown("#### Tumor Samples")
    for filename, label in sample_images['Tumor Samples']:
        try:
            with open(os.path.join('static', 'sample_images', filename), 'rb') as f:
                st.download_button(
                    label=label,
                    data=f,
                    file_name=filename,
                    mime='image/jpeg',
                    key=filename
                )
        except FileNotFoundError:
            st.warning(f"Sample {filename} not found")

# Display non-tumor samples
with col2:
    st.markdown("#### Non-Tumor Samples")
    for filename, label in sample_images['Non-Tumor Samples']:
        try:
            with open(os.path.join('static', 'sample_images', filename), 'rb') as f:
                st.download_button(
                    label=label,
                    data=f,
                    file_name=filename,
                    mime='image/jpeg',
                    key=filename
                )
        except FileNotFoundError:
            st.warning(f"Sample {filename} not found")

# Handle prediction
if uploaded_file is not None:
    try:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Make prediction
        prediction = predict_image(image)
        
        # Display result
        st.markdown("### Prediction Result:")
        st.markdown(f"<h2 style='color: #3498db;'>{prediction}</h2>", unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")