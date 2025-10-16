import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# ----- AI Libraries -----
import easyocr
reader = easyocr.Reader(['en'])  # For handwritten complaints

import whisper
model_whisper = whisper.load_model("base")  # For audio complaints

# ----- Paths -----
DATA_PATH = "data/complaints.csv"
MODEL_PATH = "models/grievance_model.pkl"

# ----- 1. Train AI model -----
def train_model():
    """
    Train a complaint category classifier using your dataset.
    Expects 'Complaint_Text' as input and 'Category' as target.
    Saves trained model to models/grievance_model.pkl
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found. Add your dataset first.")

    df = pd.read_csv(DATA_PATH)

    # Check required columns exist
    required_cols = ["Complaint_Text", "Category"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Dataset missing required column: {col}")

    X = df["Complaint_Text"]
    y = df["Category"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pipeline: TF-IDF + Logistic Regression
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)
    print(f"✅ Model trained on {len(X_train)} samples.")

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

# ----- 2. Predict category -----
def predict_category(text):
    """
    Predict category for a given complaint text.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_model() first.")
    
    model = joblib.load(MODEL_PATH)
    return model.predict([text])[0]

# ----- 3. Extract text from image -----
def extract_text_from_image(image_path):
    """
    Use EasyOCR to extract text from handwritten complaint image.
    Returns concatenated text.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    result = reader.readtext(image_path, detail=0)
    extracted_text = " ".join(result)
    return extracted_text

# ----- 4. Transcribe audio -----
def transcribe_audio(audio_path):
    """
    Use Whisper to transcribe voice complaint to text.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    result = model_whisper.transcribe(audio_path)
    return result.get("text", "")

# ----- 5. Optional: test functions -----
if __name__ == "__main__":
    # Train model if needed
    train_model()
    # Test text prediction
    sample_text = "Internet not working in my village"
    print("Predicted category:", predict_category(sample_text))
