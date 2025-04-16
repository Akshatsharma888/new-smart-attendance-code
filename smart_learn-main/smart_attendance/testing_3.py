'''main code for smart attendance. it is most imp code never delete it.'''
import os
import pickle
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import face_recognition
from pathlib import Path

# Directory where the known faces are located
image_directory = Path("named_images")

# Get a list of all image files in the directory
image_files = [f for f in image_directory.iterdir() if f.suffix in ['.png', '.jpg', '.jpeg']]

# Initialize known_faces dictionary
known_faces = {}

# File to store known face encodings
encodings_file = Path("face_encodings.pkl")

# Load known faces
if encodings_file.exists():
    with open(encodings_file, 'rb') as f:
        known_faces = pickle.load(f)
else:
    for image_file in image_files:
        name = image_file.stem  # Name of the person is the filename without the extension
        print(f"Loading image: {image_file}")
        image = face_recognition.load_image_file(str(image_file))
        # Get face locations first
        face_locations = face_recognition.face_locations(image)
        if face_locations:
            # Get the face encoding using the first detected face
            face_encoding = face_recognition.face_encodings(image, [face_locations[0]], num_jitters=1)[0]
            known_faces[name] = face_encoding
            print(f"Face encoding loaded for: {name}")
        else:
            print(f"No face encoding found for: {name}")
    # Save the face encodings for future use
    with open(encodings_file, 'wb') as f:
        pickle.dump(known_faces, f)

known_face_encodings = list(known_faces.values())
known_face_names = list(known_faces.keys())

# Initialize DataFrame to store attendance
attendance = pd.DataFrame(columns=['Name', 'Time', 'Status'])

# Add all known faces to the attendance DataFrame with status 'Absent'
for name in known_face_names:
    attendance = pd.concat([attendance, pd.DataFrame([{'Name': name, 'Time': None, 'Status': 'Absent'}])], ignore_index=True)

# Path to the group photo
group_photo_path = Path(r"C:\smart_learn-main\smart_attendance\gr9009.jpg")  # replace with your actual filename and extension

# Load the group photo
image = cv2.imread(str(group_photo_path))

if image is None:
    print(f"Could not open or find the image: {group_photo_path}")
else:
    print("Group photo loaded successfully.")
    # Convert the image from BGR to RGB (face_recognition uses RGB)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize image to 1/4 size for faster face detection
    small_frame = cv2.resize(rgb_image, (0, 0), fx=0.25, fy=0.25)
    
    # Find face locations in the group photo
    face_locations = face_recognition.face_locations(small_frame)
    print(f"Found {len(face_locations)} face(s) in the group photo.")
    
    # Process each face separately
    for i, face_location in enumerate(face_locations):
        top, right, bottom, left = face_location
        
        # Scale back up face locations since we detected faces in a scaled image
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        # Extract the face from the original image
        face_image = rgb_image[top:bottom, left:right]
        
        # Save the face to a temporary file
        temp_face_file = f"temp_face_{i}.jpg"
        cv2.imwrite(temp_face_file, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
        
        # Load the face image and get its encoding
        try:
            temp_face = face_recognition.load_image_file(temp_face_file)
            temp_face_locations = face_recognition.face_locations(temp_face)
            
            if temp_face_locations:
                # Get face encoding from the temp face image
                temp_face_encoding = face_recognition.face_encodings(temp_face, [temp_face_locations[0]])[0]
                
                # Compare with known faces
                matches = face_recognition.compare_faces(known_face_encodings, temp_face_encoding, tolerance=0.6)
                name = "Unknown"
                
                if True in matches:
                    face_distances = face_recognition.face_distance(known_face_encodings, temp_face_encoding)
                    best_match_index = np.argmin(face_distances)
                    name = known_face_names[best_match_index]
                
                # Mark the individual as present
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                if name != "Unknown":
                    attendance.loc[attendance['Name'] == name, ['Time', 'Status']] = [current_time, 'Present']
                print(f"Face #{i+1} recognized as {name}.")
            else:
                print(f"No face detected in extracted face #{i+1}")
        except Exception as e:
            print(f"Error processing face #{i+1}: {e}")
        
        # Remove temporary file
        if os.path.exists(temp_face_file):
            os.remove(temp_face_file)

# Save the attendance to an Excel file
now = datetime.now()
attendance_file_name = now.strftime("%Y-%m-%d_%H-%M-%S") + ".xlsx"
attendance.to_excel(attendance_file_name, index=False)
print(f"Attendance saved to {attendance_file_name}.")