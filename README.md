# Image_Face_Detection_System
This project uses Flask to create a web application that allows users to upload images or videos, detect faces within the media, and recognize known persons from a pre-trained dataset of facial encodings. The system supports uploading both images and videos, processes them, and displays the detected faces with names.

# Features
Face Detection: Detect faces from uploaded images or videos.
Face Recognition: Match detected faces with pre-trained known faces and display the name.
Upload Options: Users can upload images or videos.
Output Display: The processed image or video is displayed with recognized faces.

# Requirements
Before running the project, make sure you have the following installed:
Python 3.x
Flask
face_recognition
OpenCV
Werkzeug
NumPy

# Installation
# 1. Clone the Repository
Clone the repository to your local machine:
git clone <repository-url>

# 2. Install Dependencies
Navigate to the project folder and install the necessary dependencies:
 example: cd "C:\Users\Omkar\OneDrive\Documents\Image Face Detection System"
pip install -r requirements.txt

# 3. Set Up the Project
Ensure the following directory structure:

Image Face Detection System/
├── app.py
├── templates/
│   └── index.html
├── model/  (Processed or trained images model is saved here)
├── dataset/      (Add known faces for recognition here)
├── train_model.py
├── uploads/ ( All runtime files uploaded from ui are stored here)
└── requirements.txt
dataset/: Store images of people you want to recognize. Each person should have a separate folder with their name.
model/: The system will store processed images in this folder.

# 4. Add Known Faces
To train the system, add images of known people in the dataset/ folder. For example:

dataset/
├── John Doe/
│   └── john1.jpg
│   └── john2.jpg
├── Jane Smith/
│   └── jane1.jpg
│   └── jane2.jpg

# 5. Running the Application
To run the application, execute:
Step 1. python train_model.py
Step 2. python app.py
This will start a Flask web server on http://127.0.0.1:5000/.

# Usage
1. Accessing the Application
Navigate to http://127.0.0.1:5000/ in your web browser.

# 2. Uploading an Image
Click the Upload an image
Select an image from your computer.
The system will process the file, detect faces, and try to recognize them based on pre-trained data.

# 3. Viewing Processed Output
Once the processing is done, the detected faces will be displayed. If a face is recognized, it will display the name of the person. If the face is unknown, it will display "Unknown".

# 4. Supported File Formats
Images: PNG, JPG, JPEG

# File Structure
# app.py
This is the main file that contains the Flask web server. It handles file uploads, face detection, and recognition.

# train_model.py
This file contains the logic for encoding faces from images, saving them, and loading the pre-trained face encodings. It also includes functions to match detected faces with known ones.

# templates/index.html
The HTML template used by Flask to render the front end. It provides a file upload form and displays the processed media (image or video) after processing.

# Future Enhancements
# 1. Real-time Face Recognition: 
Implement real-time face recognition in webcam or live video.
# 2. Error Handling: 
Improve error handling, especially for edge cases (e.g., no faces detected).
# 3. More File Formats: 
Support additional file formats for images and videos.
# 4. Training System: Add functionality to dynamically add new known faces to the system.
