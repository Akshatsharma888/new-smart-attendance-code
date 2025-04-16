from flask import Flask, request, jsonify
import os
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import face_recognition
from pathlib import Path
from werkzeug.utils import secure_filename
import base64

from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Directory where the known faces are located
image_directory = Path("named_images")

# Initialize known_faces dictionary
known_faces = {}

# Load known faces
def load_known_faces():
    global known_faces
    known_faces = {}
    image_files = [f for f in image_directory.iterdir() if f.suffix in ['.png', '.jpg', '.jpeg']]
    for image_file in image_files:
        name = image_file.stem  # Name of the person is the filename without the extension
        print(f"Loading image: {image_file}")
        face_encodings = face_recognition.face_encodings(face_recognition.load_image_file(str(image_file)))
        if face_encodings:
            known_faces[name] = face_encodings[0]
            print(f"Face encoding loaded for: {name}")
        else:
            print(f"No face encoding found for: {name}")

# Initialize DataFrame to store attendance
attendance = pd.DataFrame(columns=['Name', 'Time', 'Status'])

@app.route('/train', methods=['POST'])
def train():
    name = request.form.get('name')
    enrollment_number = request.form.get('enrollment_number')
    if not name or not enrollment_number:
        return jsonify({"error": "Name and enrollment_number are required."}), 400

    # Check if the post request has the file part
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request."}), 400
    file = request.files['image']

    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    # Convert image to base64
    image_data = base64.b64encode(file.read()).decode('utf-8')

    # Save the image with the enrollment number as filename
    image_path = os.path.join(image_directory, f"{enrollment_number}.jpg")
    with open(image_path, 'wb') as f:
        f.write(base64.b64decode(image_data))

    # Load known faces after adding the new face
    load_known_faces()
    return jsonify({"message": "Face trained successfully."}), 200

@app.route('/identify-students', methods=['POST'])
def mark_attendance():
    global attendance
    if 'image' not in request.files:
        return jsonify({"error": "Image file is required."}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        image_path = image_directory / filename
        file.save(image_path)
        # Load the image
        image = face_recognition.load_image_file(image_path)
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        recognized_faces = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            for enrollment, known_face_encoding in known_faces.items():
                match = face_recognition.compare_faces([known_face_encoding], face_encoding)
                if match[0]:
                    # If a match was found in known_face_encodings, mark attendance
                    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    new_record = {'Name': enrollment, 'Time': time, 'Status': 'Present'}
                    attendance = attendance._append(new_record, ignore_index=True)
                    recognized_faces.append({"Name": enrollment, "Time": time})
        if recognized_faces:
            return jsonify({"message": "Attendance marked successfully.", "recognized_faces": recognized_faces}), 200
        else:
            return jsonify({"error": "No faces recognized."}), 400
    else:
        return jsonify({"error": "File type not allowed."}), 400
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == '_main_':
    # Load known faces before starting the server
    load_known_faces()
    app.run(debug=True)
