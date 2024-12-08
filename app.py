#activate venv python -m venv venv

from flask import Flask, request, render_template, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import os
import torch
from torchvision import transforms
from PIL import Image
from model import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploaded_images'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure the folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the model (Ensure that the model is loaded only once when the app starts)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model(device=device)
model.to(device)

# Transform for the uploaded image (including contrast adjustment)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((128, 128)),
    transforms.ColorJitter(contrast=0.5),  # Adjust contrast by a factor of 0.5
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Check if the file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to remove old image if it exists
def remove_old_image():
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)  # Delete the old image
        except Exception as e:
            print(f"Error while deleting file {file_path}: {e}")

# Predict function
def predict_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    
    return 'Tumor' if predicted.item() == 1 else 'No Tumor'

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling image uploads and predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file and allowed_file(file.filename):
        # Remove old image before saving the new one
        remove_old_image()

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Predict the uploaded image
        prediction = predict_image(filepath)
        return render_template('index.html', prediction=prediction, image_path=filename)
    
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
