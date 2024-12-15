import os
import face_recognition
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Path for dataset and known faces encoding
dataset_path = "dataset/"
known_faces = []
known_names = []

# Distance threshold to classify a face as unknown
FACE_DISTANCE_THRESHOLD = 0.6

# Load and split the dataset into training and testing sets
def load_and_split_data():
    global known_faces, known_names
    all_images = []
    all_labels = []
    
    # Iterate through the dataset to load images and labels
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_path):  # Check if it's a folder
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                try:
                    # Load the image
                    image = face_recognition.load_image_file(image_path)
                    # Find all face encodings in the image
                    encodings = face_recognition.face_encodings(image)
                    if encodings:  # Check if any face encodings were found
                        encoding = encodings[0]  # Take the first encoding
                        all_images.append(encoding)
                        all_labels.append(person_name)
                    else:
                        print(f"No face found in {image_path}")
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
    
    # Split data into training and testing sets
    train_images, test_images, train_labels, test_labels = train_test_split(
        all_images, all_labels, test_size=0.2, random_state=42)
    
    return (train_images, train_labels), (test_images, test_labels)

# Train and save the model
def train_model():
    # Load and split the data into train and test
    (train_images, train_labels), (test_images, test_labels) = load_and_split_data()
    
    # Save the training data (encoded faces)
    model_data = {
        'train_images': train_images,
        'train_labels': train_labels,
        'test_images': test_images,
        'test_labels': test_labels
    }
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print("Model trained and saved successfully!")
    
    # Evaluate on the test set
    evaluate_model(test_images, test_labels)

# Evaluate the model accuracy on the test set
def evaluate_model(test_images, test_labels):
    # Load the model to get known faces and names
    with open('model/model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    train_images = model_data['train_images']
    train_labels = model_data['train_labels']
    
    correct_predictions = 0
    total_predictions = len(test_images)

    for i, test_encoding in enumerate(test_images):
        # Compute distances to known faces in the training set
        distances = face_recognition.face_distance(train_images, test_encoding)
        
        # Find the closest match
        min_distance = min(distances)
        
        # If the closest match is below the threshold, we recognize the face, else it's "Unknown"
        if min_distance <= FACE_DISTANCE_THRESHOLD:
            match_index = np.argmin(distances)
            predicted_label = train_labels[match_index]
        else:
            predicted_label = "Unknown"
        
        if predicted_label == test_labels[i]:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_predictions * 100
    print(f"Model Accuracy on Test Set: {accuracy:.2f}%")

if __name__ == '__main__':
    # Train the model and evaluate on the test set
    train_model()