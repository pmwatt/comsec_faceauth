CREATE DATABASE IF NOT EXISTS comsec_faceauth;

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    face_embeddings BLOB NOT NULL
);

-- for testing
SELECT * FROM users;