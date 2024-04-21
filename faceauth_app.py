from flask import Flask, render_template, redirect, url_for, request
import cv2
import numpy as np
import mysql.connector
import os
from dotenv import load_dotenv
import torch
from scipy.spatial.distance import cosine
from facenet_pytorch import InceptionResnetV1
from mtcnn import MTCNN
from pprint import pprint

app = Flask(__name__)

# Load environment variables
load_dotenv()

# MySQL connection details
db = mysql.connector.connect(
    host=os.environ.get('host'),
    user=os.environ.get('username'),
    password=os.environ.get('password'),
    database=os.environ.get('database'),
)

# Load the pre-trained FaceNet model
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

# Initialize MTCNN for face detection and alignment
detector = MTCNN()

# Function to preprocess the face image
def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (160, 160))
    face_img = np.transpose(np.array(face_img, dtype=np.float32), (2, 0, 1))
    face_img = (face_img - face_img.mean()) / face_img.std()  # Normalize pixel values
    face_img = np.expand_dims(face_img, axis=0)
    face_img = torch.from_numpy(face_img)
    return face_img

# Function to capture and store facial data
def capture_face():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if ret:
            faces = detector.detect_faces(frame)

            if faces:
                x, y, w, h = faces[0]['box']
                face_img = frame[y:y+h, x:x+w]
                face_data = preprocess_face(face_img)

                face_embeddings = facenet_model(face_data)[0].detach().numpy()
                pprint(face_embeddings)
                cap.release()
                cv2.destroyAllWindows()
                return face_embeddings

            cv2.imshow('Capture Face', frame)

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
        similarity = 1 - cosine(face_embeddings, stored_face_embeddings)
        print('similarity: ', similarity)

        if similarity > 0.95:
            return True

    return False


############################################


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