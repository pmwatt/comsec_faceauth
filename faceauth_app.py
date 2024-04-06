import cv2
import pickle
import numpy as np
import mysql.connector
import os
from dotenv import load_dotenv

# MySQL connection details
db = mysql.connector.connect(
    host=os.getenv('host'),
    user=os.getenv('username'),
    password=os.getenv('password'),
    database=os.getenv('database'),
)

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
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (100, 100))
                face_data = np.array(face_img, dtype=np.float32).flatten()

                cap.release()
                cv2.destroyAllWindows()
                return face_data

            # Display the camera feed
            cv2.imshow('Capture Face', frame)

            # Wait for the user to press 'Enter' to capture the face
            if cv2.waitKey(1) & 0xFF == 13:
                cap.release()
                cv2.destroyAllWindows()
                return face_data

            # Wait for the user to press 'Esc' to exit
            if cv2.waitKey(1) & 0xFF == 27:
                cap.release()
                cv2.destroyAllWindows()
                return None


# Function to register a new user
def register_user(username, face_data):
    cursor = db.cursor()

    try:
        # Convert the face data to a byte string using pickle
        print(face_data)
        face_data_bytes = pickle.dumps(face_data)

        # Insert the username and face data into the database
        sql = "INSERT INTO users (username, face_data) VALUES (%s, %s)"
        values = (username, face_data_bytes)
        cursor.execute(sql, values)
        db.commit()

        print("Registration successful!")
    except mysql.connector.Error as error:
        print(f"Error: {error}")
        db.rollback()

# Function to authenticate the user
def authenticate_user(username, face_data):
    cursor = db.cursor()

    # Check if the username exists in the database
    cursor.execute("SELECT face_data FROM users WHERE username = %s", (username,))
    result = cursor.fetchone()

    if result:
        # Convert the stored byte string back to a NumPy array
        stored_face_data = pickle.loads(result[0])
        print(stored_face_data)

        # Compare the captured face data with the stored face data
        diff = cv2.norm(face_data, stored_face_data, cv2.NORM_L2)
        print(diff)

        if diff < 20000:
            return True

    return False

# Main function
def main():
    print("Welcome to the facial authentication system!")
    print("Please choose an option:")
    print("1. Login")
    print("2. Register")

    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        username = input("Enter your username: ")
        face_data = capture_face()

        if face_data is not None:
            if authenticate_user(username, face_data):
                print("Authentication successful!")
            else:
                print("Authentication failed.")
        else:
            print("Unable to capture face data.")
    elif choice == '2':
        username = input("Enter a new username: ")
        face_data = capture_face()

        if face_data is not None:
            register_user(username, face_data)
        else:
            print("Unable to capture face data.")
    else:
        print("Invalid choice. Please try again.")

if __name__ == "__main__":
    load_dotenv(override=True)
    main()