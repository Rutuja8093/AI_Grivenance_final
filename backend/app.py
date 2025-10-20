import os
import sqlite3
from flask import Flask, render_template, request, redirect, url_for
from flask_cors import CORS
from nlp import predict_category, extract_text_from_image, transcribe_audio
import pandas as pd  # For CSV export

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

DB_PATH = "data/complaints.db"
CSV_PATH = "data/all_complaints.csv"

# -------------------------------
# Initialize Database
# -------------------------------
def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS complaints (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        village_name TEXT,
        text TEXT,
        category TEXT,
        type TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()
    print("✅ Database initialized")

# -------------------------------
# Export to CSV
# -------------------------------
def update_csv():
    """Export all complaints to CSV."""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM complaints", conn)
        os.makedirs("data", exist_ok=True)
        df.to_csv(CSV_PATH, index=False)
        print("✅ CSV updated")
    finally:
        conn.close()

# Initialize DB once at import (safe)
init_db()

# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def home():
    return redirect("/login")

# -------- Login Page --------
@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        role = request.form.get("role", "user")
        if role == "user":
            return redirect(url_for("complaint_page", username=username))
        else:
            return redirect(url_for("admin_dashboard"))
    return render_template("login.html")

# -------- Complaint Submission Page --------
@app.route("/complaint/<username>", methods=["GET","POST"])
def complaint_page(username):
    message = ""
    if request.method == "POST":
        village_name = request.form.get("village_name", "")
        ctype = request.form.get("ctype", "text")
        text = ""

        # Extract complaint text based on type
        if ctype == "text":
            text = request.form.get("complaint_text", "")
        elif ctype in ["image", "audio"]:
            file = request.files.get("file")
            if file:
                # keep filename safe in production; here keep simple
                file_path = os.path.join("data", file.filename)
                file.save(file_path)
                if ctype == "image":
                    text = extract_text_from_image(file_path)
                else:
                    text = transcribe_audio(file_path)

        # Predict category
        try:
            category = predict_category(text)
        except Exception as e:
            category = "Unknown"
            print("Error predicting category:", e)

        # Insert into DB
        conn = sqlite3.connect(DB_PATH)
        try:
            conn.execute(
                "INSERT INTO complaints (username, village_name, text, category, type) VALUES (?,?,?,?,?)",
                (username, village_name, text, category, ctype)
            )
            conn.commit()
        finally:
            conn.close()

        # Update CSV immediately
        update_csv()

        message = f"✅ Complaint submitted! Predicted category: {category}"

    return render_template("complaint.html", username=username, message=message)

# -------- Admin Dashboard --------
@app.route("/admin")
def admin_dashboard():
    conn = sqlite3.connect(DB_PATH)
    try:
        complaints = conn.execute("SELECT * FROM complaints ORDER BY timestamp DESC").fetchall()
    finally:
        conn.close()
    return render_template("admin_dashboard.html", complaints=complaints)

# -------------------------------
# Run App (Render uses PORT env var)
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
