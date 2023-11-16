import cv2
import pickle
import face_recognition
import numpy as np
import firebase_admin
from firebase_admin import credentials, db, storage
import pyttsx3
import tkinter as tk
from tkinter import simpledialog
import os
import uuid

# Initialize Firebase Admin SDK with credentials and specify the storage bucket name
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://face-recognition-system-ebb56-default-rtdb.firebaseio.com/",
    'storageBucket': "face-recognition-system-ebb56.appspot.com"
})
bucket = storage.bucket(app=firebase_admin.get_app())
imgPath = 'images'
# Initialize text-to-speech engine
engine = pyttsx3.init()


def speak(text):
    engine.say(text)
    engine.runAndWait()


def load_encodings(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except EOFError:
        print(f"Error: EOFError occurred while loading encodings from {file_path}")
        return None
    except Exception as e:
        print(f"Error loading encodings: {e}")
        return None


def save_encodings(file_path, encodings_data):
    with open(file_path, 'wb') as file:
        pickle.dump(encodings_data, file)


def get_client_info(client_id):
    ref = db.reference(f'Clients/{client_id}')
    client_info = ref.get()
    return client_info


def save_client_info(client_id, client_info):
    ref = db.reference(f'Clients/{client_id}')
    ref.set(client_info)


def upload_image_to_firebase(client_id, img_path):
    destination_path = f'images/{client_id}/{os.path.basename(img_path)}'
    blob = bucket.blob(destination_path)
    blob.upload_from_filename(img_path)


def get_client_image(client_id):
    blob = bucket.get_blob(f'images/{client_id}/{client_id}.jpeg')
    if blob:
        array = np.frombuffer(blob.download_as_string(), np.uint8)
        return cv2.imdecode(array, cv2.IMREAD_COLOR)
    return None


def display_text(img, text, position, font=cv2.FONT_HERSHEY_COMPLEX_SMALL, scale=1, color=(0, 0, 0), thickness=1):
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    offset = (img.shape[1] - w) // 2
    cv2.putText(img, str(text), (position[0] + offset, position[1]), font, scale, color, thickness)


def show_register_dialog():
    root = tk.Tk()
    root.withdraw()

    name = simpledialog.askstring("User Registration", "Enter your name:")
    if name is None:  
        root.destroy()
        return None

    domain = simpledialog.askstring("User Registration", "Enter your domain:")
    if domain is None:
        root.destroy()
        return None

    age = simpledialog.askstring("User Registration", "Enter your age:")
    if age is None:
        root.destroy()
        return None

    year = simpledialog.askstring("User Registration", "Enter your year:")
    if year is None:
        root.destroy()
        return None

    root.destroy()

    return {
        'name': name,
        'Domain': domain,
        'Age': age,
        'year': year
    }

# Face Recognition System

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

encodings_file = 'encoding.p'
encodings_data = load_encodings(encodings_file)

if encodings_data is not None:
    encodingListKnown, clientId = encodings_data
    print("Encodings loaded successfully.")
else:
  
    encodingListKnown = []
    clientId = []

while True:
    success, img = cap.read()

    img_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

    face_cur_frame = face_recognition.face_locations(img_small)
    encode_cur_frame = face_recognition.face_encodings(img_small, face_cur_frame)

    if len(face_cur_frame) > 0:
        for encode_face, face_location in zip(encode_cur_frame, face_cur_frame):
            matches = face_recognition.compare_faces(encodingListKnown, encode_face)
            face_distance = face_recognition.face_distance(encodingListKnown, encode_face)

            if len(face_distance) > 0: 
                match_index = np.argmin(face_distance)

                if matches[match_index]:
                    client_id = clientId[match_index]

                    client_info = get_client_info(client_id)

                    if client_info:
                        img_clients = get_client_image(client_id)

                        display_text(img, client_info.get('name', 'Unknown'), (50, 50))
                        display_text(img, client_id, (50, 100), color=(100, 100, 100))
                        display_text(img, client_info.get('Domain', 'Unknown'), (50, 150), color=(100, 100, 100))

                        if isinstance(img_clients, np.ndarray) and img_clients.size > 0:
                            img_clients_resized = cv2.resize(img_clients, (590, 280))
                            img[200:200 + img_clients_resized.shape[0], 50:50 + img_clients_resized.shape[1]] = img_clients_resized
                    else:
                        print("Client information not found.")
                else:
                    new_client_id = str(uuid.uuid4())

                    client_info = show_register_dialog()

                    if client_info is not None: 
                        save_client_info(new_client_id, client_info)
                        img_path = f'images/{new_client_id}.jpeg'
                        cv2.imwrite(img_path, img)
                        upload_image_to_firebase(new_client_id, img_path)
                        speak("Registration successful. Welcome to the system.")
                        encodingListKnown.append(encode_face)
                        clientId.append(new_client_id)
                        save_encodings(encodings_file, [encodingListKnown, clientId])
            else:
                # No existing encodings, treat as a new user
                new_client_id = str(uuid.uuid4())

                client_info = show_register_dialog()

                if client_info is not None: 
                    save_client_info(new_client_id, client_info)
                    img_path = f'images/{new_client_id}.jpeg'
                    cv2.imwrite(img_path, img)
                    upload_image_to_firebase(new_client_id, img_path)
                    speak("Registration successful. Welcome to the system.")
                    encodingListKnown.append(encode_face)
                    clientId.append(new_client_id)
                    save_encodings(encodings_file, [encodingListKnown, clientId])

    cv2.imshow("Face Recognition", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
