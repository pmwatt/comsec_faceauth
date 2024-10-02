# Facial Authentication using OpenCV

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
     - `username="prachnachai"`
     - `password="no"`
 - Install dependencies
   - `pip install -r requirements.txt`
 - Run `python faceauth-app.py`
