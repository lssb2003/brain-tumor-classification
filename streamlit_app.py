import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
from model import load_model

# Styling to match your original Flask frontend with compact image display
st.markdown("""
    <style>
    /* Main background and container */
    .stApp {
        background-color: #2c3e50;
        color: #ecf0f1;
    }
    
    /* Title styling */
    h1 {
        color: #3498db !important;
        font-size: 2.5em;
        margin-bottom: 20px;
    }
    
    /* File uploader styling */
    .stUploadedFileContent {
        background-color: #34495e;
    }
    
    /* Button styling - orange like original */
    .stButton > button {
        background-color: #f39c12 !important;
        color: white !important;
        border: none !important;
        padding: 10px 20px !important;
        border-radius: 5px !important;
        font-size: 1.1em !important;
        transition: background-color 0.3s !important;
    }
    
    .stButton > button:hover {
        background-color: #e67e22 !important;
        transform: scale(1.05);
    }
    
    /* Section headers */
    h2, h3, h4 {
        color: #3498db !important;
        margin-bottom: 15px;
    }
    
    /* Prediction result */
    .prediction-text {
        color: #3498db;
        font-size: 1.2em;
        font-weight: bold;
    }
    
    /* Container styling */
    .element-container {
        background-color: #34495e;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    
    /* Updated image display styling */
    div[data-testid="stImage"] {
        width: 350px !important;   /* Container width */
        margin: auto !important;
    }
    
    div[data-testid="stImage"] img {
        max-width: 350px !important;   /* Image max width */
        max-height: 250px !important;  /* Image max height */
        object-fit: contain !important;
        border-radius: 10px !important;
        border: 5px solid #3498db !important;
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
st.markdown("### Upload Image")
uploaded_file = st.file_uploader("Choose a brain scan image", type=['png', 'jpg', 'jpeg'])

# Sample images section
st.markdown("### Sample Images for Testing")
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
            pass

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
            pass

# Handle prediction
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, 
                caption='Uploaded Image', 
                use_column_width=False,  # Don't use column width
                width=350)  # Smaller fixed width
        
        prediction = predict_image(image)

        
        # Display result
        st.markdown("### Prediction Result:")
        st.markdown(f"<h2 style='color: #3498db;'>{prediction}</h2>", unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")