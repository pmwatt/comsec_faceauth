from flask import Flask, render_template, redirect, url_for, request
import cv2
import numpy as np
import mysql.connector
import os
from dotenv import load_dotenv
import torch
from scipy import spatial
from facenet_pytorch import InceptionResnetV1

app = Flask(__name__)

# MySQL connection details
db = mysql.connector.connect(
    host=os.getenv('host'),
    user=os.getenv('username'),
    password=os.getenv('password'),
    database=os.getenv('database'),
)

# Load the pre-trained FaceNet model
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

# Function to capture and store facial data
def capture_face():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if ret:
            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the grayscale frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                # Normalize the face image and convert it to a numpy array
                x, y, w, h = faces[0]
                face_img = frame[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (160, 160))
                face_data = np.array(face_img, dtype=np.float32)
                face_data = np.transpose(face_data, (2, 0, 1))  # Reorder the channels to match the expected input
                face_data = np.expand_dims(face_data, axis=0)  # Add batch dimension

                # Convert the NumPy array to a PyTorch tensor
                face_data = torch.from_numpy(face_data)

                # Use FaceNet to extract facial features
                face_embeddings = facenet_model(face_data)[0].detach().numpy()

                cap.release()
                cv2.destroyAllWindows()
                return face_embeddings

            # Display the camera feed
            cv2.imshow('Capture Face', frame)

            # Wait for the user to press 'Esc' to exit
            if cv2.waitKey(1) & 0xFF == 27:
                cap.release()
                cv2.destroyAllWindows()
                return None


# Function to register a new user
def register_user(username, face_embeddings):
    cursor = db.cursor()

    try:
        # Insert the username and face embeddings into the database
        sql = "INSERT INTO users (username, face_embeddings) VALUES (%s, %s)"
        values = (username, face_embeddings.tobytes())
        cursor.execute(sql, values)
        db.commit()

        return True
    except mysql.connector.Error as error:
        print(f"Error: {error}")
        db.rollback()
        return False

# Function to authenticate the user
def authenticate_user(username, face_embeddings):
    cursor = db.cursor()

    # Check if the username exists in the database
    cursor.execute("SELECT face_embeddings FROM users WHERE username = %s", (username,))
    result = cursor.fetchone()

    if result:
        # Convert the stored byte string back to a NumPy array
        stored_face_embeddings = np.frombuffer(result[0], dtype=np.float32)

        # Use cosine similarity to compare the captured face embeddings with the stored face embeddings
        similarity = 1 - spatial.distance.cosine(face_embeddings, stored_face_embeddings)
        print('similarity: ', similarity)

        if similarity > 0.8:
            return True

    return False

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        face_embeddings = capture_face()

        if face_embeddings is not None:
            if authenticate_user(username, face_embeddings):
                return redirect(url_for('home'))
            else:
                return render_template('login.html', error='Invalid login facial authentication')
        else:
            return render_template('login.html', error='Unable to capture face data')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']

        # Check if the username already exists
        cursor = db.cursor()
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        result = cursor.fetchone()

        if result:
            return render_template('register.html', error='Username already exists')

        face_embeddings = capture_face()

        if face_embeddings is not None:
            if register_user(username, face_embeddings):
                return redirect(url_for('login'))
            else:
                return render_template('register.html', error='Registration failed')
        else:
            return render_template('register.html', error='Unable to capture face data')
    return render_template('register.html')

@app.route('/home')
def home():
    return render_template('home.html')

if __name__ == '__main__':
    load_dotenv(override=True)
    app.run(debug=True)