# NeuroScan AI — Alzheimer's MRI Classifier

Full-stack web app: MRI image classification + AI-generated medical report.

```
alzheimer-app/
├── backend/
│   ├── app.py            ← Flask API  (main server)
│   ├── requirements.txt  ← Python deps
│   ├── render.yaml       ← Render deploy config
│   └── train_model.py    ← (Optional) train & save model.h5 in Colab
└── frontend/
    ├── index.html
    ├── style.css
    ├── script.js
    └── netlify.toml
```

---

## STEP 0 – (Optional) Train & Export Your Model

Run this inside Google Colab, pointing at your unzipped dataset:

```python
# In Colab
!unzip /content/archive.zip -d /content/data
!python train_model.py   # produces model.h5
```

Download `model.h5` and place it in `backend/model.h5`.

> If you skip this step, the backend uses a smart demo predictor that still works for the demo.

---

## STEP 1 – Deploy Backend to Render (Free)

1. Push your project to GitHub:
   ```bash
   git init
   git add .
   git commit -m "initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/alzheimer-app.git
   git push -u origin main
   ```

2. Go to https://render.com → **New → Web Service**

3. Connect your GitHub repo.

4. Settings:
   - **Root directory**: `backend`
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `gunicorn app:app`
   - **Instance type**: Free

5. Click **Deploy**.

6. Wait ~3 minutes. Note your URL, e.g.:
   `https://alzheimer-classifier-api.onrender.com`

---

## STEP 2 – Update Frontend API URL

Open `frontend/script.js` and replace:
```js
const API_URL = "https://YOUR-APP-NAME.onrender.com/predict";
```
with your actual Render URL.

---

## STEP 3 – Deploy Frontend to Netlify (Free)

**Option A – Drag & Drop (fastest)**
1. Go to https://netlify.com → **Add new site → Deploy manually**
2. Drag the `frontend/` folder into the deploy box.
3. Done! Your site goes live instantly.

**Option B – GitHub CI/CD**
1. Go to https://netlify.com → **Add new site → Import from Git**
2. Connect your GitHub repo.
3. Set **Base directory** to `frontend`
4. Leave build command empty (static site).
5. Click **Deploy site**.

---

## STEP 4 – Test It

1. Visit your Netlify URL.
2. Upload a brain MRI image.
3. Click **Analyse Scan**.
4. See the prediction + full NLP medical report.

---

## Local Development

```bash
# Backend
cd backend
pip install -r requirements.txt
python app.py           # runs on http://localhost:5000

# Frontend – open in browser directly
open frontend/index.html
# Then update API_URL in script.js to http://localhost:5000/predict
```

---

## Classes

| Class | Description |
|---|---|
| NonDemented | No signs of Alzheimer's |
| VeryMildDemented | Early subtle changes |
| MildDemented | Noticeable memory loss, needs medical attention |
| ModerateDemented | Significant neurodegeneration, full-time care needed |

---

## Disclaimer

This tool is for research and educational purposes only. It does NOT provide clinical medical diagnoses. Always consult a certified neurologist.
