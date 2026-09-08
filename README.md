# ODI Player Tier Predictor

A machine learning API that predicts an ODI (One Day International) cricket batsman's performance tier — **Weak**, **Average**, **Great**, or **Elite** — based on their career batting statistics. Built end-to-end: data cleaning, exploratory analysis, model training, and deployment as a live web app.

🔗 **Live demo:** [odi-tier-api-production.up.railway.app/form](https://odi-tier-api-production.up.railway.app/form)
📄 **API docs:** [odi-tier-api-production.up.railway.app/docs](https://odi-tier-api-production.up.railway.app/docs)

---

## Overview

This project classifies ODI batsmen into performance tiers using a Random Forest classifier trained on career stats (matches, innings, runs, strike rate, centuries, etc.). It was built as a first classification project, following a full pipeline from raw data to a publicly deployed prediction tool.

### Tier definitions
Based on career batting average (Ave):

| Tier | Criteria |
|---------|------------------|
| Elite | Ave ≥ 40 |
| Great | 30 ≤ Ave < 40 |
| Average | 18 ≤ Ave < 30 |
| Weak | Ave < 18 |

---

## Tech stack

- **Python** — pandas, numpy, scikit-learn
- **Model** — Random Forest Classifier (`class_weight='balanced'`)
- **API** — FastAPI
- **Frontend** — plain HTML/CSS/JS form (no framework), served directly by FastAPI
- **Deployment** — Railway (hosting), GitHub (version control)
- **Development environment** — Google Colab

---

## Project pipeline

1. **Data cleaning** — handled missing values (`-` placeholders for non-batters), parsed "not out" high scores (e.g. `150*`), filtered out players with fewer than 10 innings to avoid noise from small samples.
2. **Exploratory Data Analysis (EDA)** — visualized distributions of average and strike rate, correlation heatmap between features, and class balance across tiers.
3. **Feature engineering** — created the `Tier` target label from the `Ave` column; engineered an `HS_not_out` flag from the raw `HS` field.
4. **Model training** — compared Logistic Regression (with feature scaling) against a Random Forest Classifier (`class_weight='balanced'` to handle tier imbalance).
5. **Evaluation** — accuracy, precision/recall/F1 per class, and confusion matrix. Random Forest selected as the final model (~82% weighted accuracy).
6. **Deployment** — model and label encoder serialized with `pickle`, served via a FastAPI `/predict` endpoint, with a simple HTML form UI at `/form` for non-technical use. Hosted on Railway, connected to this GitHub repo for automatic redeployment on push.

---

## Model performance

| Tier | Precision | Recall | F1-score |
|---------|-----------|--------|----------|
| Weak | 0.90 | 0.88 | 0.89 |
| Average | 0.78 | 0.89 | 0.83 |
| Great | 0.72 | 0.63 | 0.68 |
| Elite | 0.78 | 0.44 | 0.56 |

**Overall accuracy: ~82%**

Elite is the hardest tier to predict correctly, mainly because it's the smallest class in the dataset (few truly elite batsmen relative to the rest) and sits right next to the "Great" tier, so misclassifications tend to happen between neighboring tiers rather than randomly.

---

## API usage

### `POST /predict`

Send a JSON body with a player's career stats:

```json
{
  "Mat": 200,
  "Inns": 195,
  "NO": 20,
  "Runs": 8500,
  "HS": 150,
  "BF": 9000,
  "SR": 94.4,
  "Hundreds": 25,
  "Fifties": 45,
  "Ducks": 8,
  "HS_not_out": 1
}
```

Response:

```json
{
  "predicted_tier": "Great"
}
```

### `GET /form`

A simple HTML form for entering stats and getting a prediction directly in the browser — no API knowledge required.

### `GET /`

Health check endpoint, confirms the API is running.

---

## Running locally

```bash
git clone https://github.com/rafaqatkhattak5/odi-tier-api.git
cd odi-tier-api
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then visit `http://localhost:8000/form` or `http://localhost:8000/docs`.

---

## Dataset

ODI batting statistics for ~2,500 international players, including matches, innings, runs, high score, average, strike rate, centuries, half-centuries, and ducks.

---

## Future improvements

- Address Elite-tier class imbalance further (e.g. SMOTE oversampling)
- Add feature importance visualization to the app
- Expand to bowling stats for an all-rounder tier system
- Add authentication/rate-limiting for public API use

---

## Author

Built by [Rafaqat Ullah](https://github.com/rafaqatkhattak5) 
