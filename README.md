# Facial Authentication using OpenCV
Made using VGGFace2 pretrained datasets and Inception Resnet V1 model ([see more](https://github.com/timesler/facenet-pytorch)). Frontend UI made using Flask, and database made using MySQL. We set the similarity threshold between the embeddings stored in the database and the face detected during login to 0.8, but feel free to adjust.

## Screenshots
Figure 1: Login Page and invalid login
![ภาพ](https://github.com/user-attachments/assets/98e906c1-0556-42d3-af9c-d7035df20283)
![ภาพ](https://github.com/user-attachments/assets/14886209-8311-4a1e-92cd-2ec32904a269)

Figure 2: Register Page to store new facial embeddings
![ภาพ](https://github.com/user-attachments/assets/91db0b40-ffa9-429b-8e3e-6856f033d455)

Figure 3: Home page upon successful login
![ภาพ](https://github.com/user-attachments/assets/46d34e4f-9384-4b30-a0ec-6d8b60c6c849)

## Setup
- Use MySQL workbench to run `database.sql`. This should create a database and an empty users table.
- Set up MySQL workbench user privileges for SELECT/INSERT/UPDATE/DELETE as the following
   - Toolbar > Server > User and Privileges > Add Account
   - Set up **Login Name** and **Password** (copy or memorize them too)
   - Schema Privileges > Add Entry... > Select schema `comsec_faceauth` > OK
   - Object Rights > toggle SELECT, INSERT, UPDATE, DELETE
   - Apply
 - Then create a new `.env` file in the project's root
   - Copy the **Login Name** and **Password** (that you filled into the MySQL workbench) into `.env`'s `username` and `password` variables. For example, `.env` should have something like these:

```
host="localhost"
database="comsec_faceauth"
username="pokemonmysterydungeon"
password="ExplorersOfTime"
```

 - Install dependencies
   - `pip install -r requirements.txt`
   - `pip install "numpy<2.0" tensorflow mysql-connector-python`
 - Run `python faceauth-app.py`
