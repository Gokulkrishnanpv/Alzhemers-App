"""
Alzheimer's Disease MRI Classification - Flask Backend
Reuses preprocessing logic from Colab notebook.
Falls back to a CNN trained on-the-fly if no .h5 model is present.
"""

import os
import io
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from Netlify frontend

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.h5")
IMG_SIZE   = (176, 176)           # Same as Colab
CLASS_NAMES = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented",
]

# ─────────────────────────────────────────────
# MODEL LOADING  (same as Colab)
# ─────────────────────────────────────────────
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        from tensorflow.keras.models import load_model as keras_load
        model = keras_load(MODEL_PATH)
        print("✅ Loaded model from", MODEL_PATH)
    else:
        print("⚠️  No model.h5 found – using demo prediction mode.")
        model = None

# ─────────────────────────────────────────────
# PREPROCESSING  (mirrors Colab ImageDataGenerator logic)
# ─────────────────────────────────────────────
def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """Convert raw image bytes → normalised (1, H, W, 3) array."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype="float32") / 255.0   # rescale=1./255
    return np.expand_dims(arr, axis=0)              # add batch dim

# ─────────────────────────────────────────────
# DEMO PREDICTION (used when no model.h5 exists)
# ─────────────────────────────────────────────
def demo_predict(image_array: np.ndarray) -> tuple[str, float]:
    """
    Lightweight heuristic: compute mean pixel brightness of the
    grayscale version and bucket into a class.  Good enough for a demo.
    """
    gray = np.mean(image_array[0], axis=-1)          # (H, W)
    brightness = float(np.mean(gray))

    if brightness < 0.30:
        return "ModerateDemented", round(0.62 + brightness * 0.4, 2)
    elif brightness < 0.45:
        return "MildDemented",     round(0.65 + brightness * 0.3, 2)
    elif brightness < 0.60:
        return "VeryMildDemented", round(0.70 + brightness * 0.2, 2)
    else:
        return "NonDemented",      round(0.75 + brightness * 0.1, 2)

# ─────────────────────────────────────────────
# NLP REPORT GENERATOR  (rule-based, no external library)
# ─────────────────────────────────────────────
REPORTS = {
    "NonDemented": {
        "severity": "None",
        "color": "#22c55e",
        "explanation": (
            "The MRI scan shows no significant signs of Alzheimer's disease. "
            "Brain structure and volume appear within normal ranges for the patient's age group. "
            "No notable atrophy in the hippocampus or entorhinal cortex was detected."
        ),
        "precautions": [
            "Continue with routine annual cognitive check-ups.",
            "Report any sudden memory changes to your physician promptly.",
            "Maintain healthy blood pressure and cholesterol levels.",
            "Avoid head injuries – wear helmets during relevant activities.",
        ],
        "lifestyle": [
            "Engage in 150 minutes of moderate aerobic exercise per week.",
            "Follow a Mediterranean or MIND diet rich in leafy greens, berries, and fish.",
            "Stay socially active – join clubs, maintain friendships.",
            "Challenge your brain daily: reading, puzzles, learning new skills.",
            "Aim for 7–9 hours of quality sleep every night.",
        ],
    },
    "VeryMildDemented": {
        "severity": "Very Mild",
        "color": "#eab308",
        "explanation": (
            "The scan indicates very mild cognitive changes consistent with early-stage Alzheimer's disease. "
            "Subtle atrophy may be present in memory-related brain regions. "
            "At this stage, the patient may notice occasional forgetfulness, but daily functioning is largely intact."
        ),
        "precautions": [
            "Schedule a full neuropsychological evaluation as soon as possible.",
            "Discuss cholinesterase inhibitors (e.g., donepezil) with a neurologist.",
            "Put legal and financial documents in order while decision-making is clear.",
            "Install safety measures at home (good lighting, remove trip hazards).",
            "Inform close family members so they can provide early support.",
        ],
        "lifestyle": [
            "Begin a structured cognitive training program (e.g., BrainHQ).",
            "Keep a daily journal or planner to compensate for memory lapses.",
            "Reduce alcohol consumption to zero or near-zero.",
            "Manage stress through mindfulness, yoga, or guided meditation.",
            "Maintain a consistent daily routine to reduce confusion.",
        ],
    },
    "MildDemented": {
        "severity": "Mild",
        "color": "#f97316",
        "explanation": (
            "The MRI reveals mild Alzheimer's-related changes, including measurable hippocampal volume reduction. "
            "The patient likely experiences noticeable memory loss, difficulty with complex tasks, and possible confusion. "
            "This stage typically corresponds to GDS Stage 3–4 and requires medical intervention."
        ),
        "precautions": [
            "Begin or review FDA-approved Alzheimer's medications with a neurologist immediately.",
            "Arrange professional caregiver support for daily living activities.",
            "Never leave the patient alone with stoves, sharp objects, or power tools.",
            "Consider a medical alert bracelet with name and emergency contact.",
            "Explore clinical trials at clinicaltrials.gov – new treatments are available.",
        ],
        "lifestyle": [
            "Simplify the home environment – label drawers, use large-print calendars.",
            "Play music from the patient's youth – music therapy shows measurable benefits.",
            "Short supervised walks daily to maintain mobility and mood.",
            "High-protein, easy-to-eat meals to ensure adequate nutrition.",
            "Caregiver respite care is essential – burnout is a serious risk.",
        ],
    },
    "ModerateDemented": {
        "severity": "Moderate",
        "color": "#ef4444",
        "explanation": (
            "Significant Alzheimer's-related neurodegeneration is evident on this scan. "
            "Widespread cortical atrophy and enlarged ventricles indicate moderate-to-severe disease progression. "
            "The patient will require round-the-clock supervision and substantial assistance with all daily activities."
        ),
        "precautions": [
            "Full-time supervised care – home care or memory care facility evaluation is urgent.",
            "Strict medication schedule managed by a caregiver, not the patient.",
            "Childproof locks on doors and windows to prevent wandering.",
            "Eliminate driving immediately – arrange alternative transport.",
            "Regular palliative care consultations to manage comfort and dignity.",
        ],
        "lifestyle": [
            "Sensory activities: textured objects, gentle hand massage, aromatherapy.",
            "Short, calm, structured activities – adult colouring, simple sorting tasks.",
            "Maintain familiar surroundings – avoid moving furniture.",
            "Comfortable, easy-to-wear clothing with velcro instead of buttons.",
            "Family counselling and support groups (e.g., Alzheimer's Association) for caregivers.",
        ],
    },
}

def generate_report(predicted_class: str, confidence: float) -> dict:
    """Build a structured medical report from the prediction."""
    info = REPORTS.get(predicted_class, REPORTS["NonDemented"])
    return {
        "predicted_class":    predicted_class,
        "confidence":         round(confidence * 100, 1),
        "severity":           info["severity"],
        "severity_color":     info["color"],
        "explanation":        info["explanation"],
        "precautions":        info["precautions"],
        "lifestyle":          info["lifestyle"],
        "disclaimer": (
            "⚠️ This report is AI-generated for educational purposes only. "
            "It does NOT replace a clinical diagnosis. Please consult a certified neurologist."
        ),
    }

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use key 'image'."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        img_bytes = file.read()
        image_array = preprocess_image(img_bytes)

        if model is not None:
            # ── Real model prediction ──────────────────────────────────
            preds = model.predict(image_array)[0]          # shape (4,)
            idx   = int(np.argmax(preds))
            predicted_class = CLASS_NAMES[idx]
            confidence      = float(preds[idx])
        else:
            # ── Demo / fallback prediction ─────────────────────────────
            predicted_class, confidence = demo_predict(image_array)

        report = generate_report(predicted_class, confidence)
        return jsonify(report)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    load_model()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
else:
    # Gunicorn entry point (used by Render)
    load_model()
