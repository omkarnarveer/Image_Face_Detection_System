import os
import face_recognition
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# Initialize the Flask app
app = Flask(__name__)

# Path for the model and uploads
MODEL_PATH = 'model/model.pkl'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Distance threshold to classify a face as unknown
FACE_DISTANCE_THRESHOLD = 0.6

# Load the trained model
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    return model_data

# Detect face and predict the name
def recognize_face(image):
    model_data = load_model()
    known_faces = model_data['train_images']
    known_names = model_data['train_labels']
    
    # Get face locations and encodings from the uploaded image
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    if len(face_encodings) == 0:
        return "No faces detected"
    
    predictions = []
    for face_encoding in face_encodings:
        # Compute distances to known faces
        distances = face_recognition.face_distance(known_faces, face_encoding)
        
        # Find the closest match
        min_distance = min(distances)
        
        # If the closest match is below the threshold, we recognize the face, else it's "Unknown"
        if min_distance <= FACE_DISTANCE_THRESHOLD:
            match_index = np.argmin(distances)
            name = known_names[match_index]
        else:
            name = "Unknown"
        
        predictions.append(name)
    
    return predictions

# Route for the main UI
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save uploaded file
    filename = secure_filename(file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(image_path)
    
    # Load the image for prediction
    try:
        image = face_recognition.load_image_file(image_path)
        predictions = recognize_face(image)
        
        # Check if the result is "No faces detected"
        if predictions == "No faces detected":
            return jsonify({'error': 'No faces detected in the image'}), 400
        return jsonify({'result': predictions})
    except Exception as e:
        return jsonify({'error': f"An error occurred while processing the image: {str(e)}"}), 500

if __name__ == '__main__':
    # Ensure the uploads folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    # Run the app
    app.run(debug=True)