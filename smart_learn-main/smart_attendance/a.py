# # Load known faces
# if encodings_file.exists():
#     with open(encodings_file, 'rb') as f:
#         known_faces = pickle.load(f)
# else:
#     for image_file in image_files:
#         name = image_file.stem.split('_')[0]  # Assume the name is before the first underscore
#         print(f"Loading image: {image_file}")
#         face_encodings = face_recognition.face_encodings(face_recognition.load_image_file(str(image_file)))
#         if face_encodings:
#             if name in known_faces:
#                 known_faces[name].append(face_encodings[0])
#             else:
#                 known_faces[name] = [face_encodings[0]]
#             print(f"Face encoding loaded for: {name}")
#         else:
#             print(f"No face encoding found for: {name}")
#     # Save the face encodings for future use
#     with open(encodings_file, 'wb') as f:
#         pickle.dump(known_faces, f)