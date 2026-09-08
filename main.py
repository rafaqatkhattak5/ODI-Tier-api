from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI(title="ODI Player Tier Predictor")

with open('odi_tier_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

class PlayerStats(BaseModel):
    Mat: float
    Inns: float
    NO: float
    Runs: float
    HS: float
    BF: float
    SR: float
    Hundreds: float
    Fifties: float
    Ducks: float
    HS_not_out: int

@app.get("/")
def home():
    return {"message": "ODI Player Tier Prediction API is running"}

@app.post("/predict")
def predict_tier(stats: PlayerStats):
    features = np.array([[
        stats.Mat, stats.Inns, stats.NO, stats.Runs, stats.HS,
        stats.BF, stats.SR, stats.Hundreds, stats.Fifties,
        stats.Ducks, stats.HS_not_out
    ]])

    pred_encoded = model.predict(features)[0]
    pred_tier = le.inverse_transform([pred_encoded])[0]

    return {"predicted_tier": pred_tier}

@app.get("/form", response_class=HTMLResponse)
def form():
    return """
    <html>
    <head><title>ODI Player Tier Predictor</title></head>
    <body style="font-family:sans-serif; max-width:500px; margin:40px auto;">
        <h2>ODI Player Tier Predictor</h2>
        <form id="predictForm">
            Matches: <input type="number" id="Mat" required><br><br>
            Innings: <input type="number" id="Inns" required><br><br>
            Not Outs: <input type="number" id="NO" required><br><br>
            Runs: <input type="number" id="Runs" required><br><br>
            Highest Score: <input type="number" id="HS" required><br><br>
            Balls Faced: <input type="number" id="BF" required><br><br>
            Strike Rate: <input type="number" step="0.01" id="SR" required><br><br>
            Hundreds: <input type="number" id="Hundreds" required><br><br>
            Fifties: <input type="number" id="Fifties" required><br><br>
            Ducks: <input type="number" id="Ducks" required><br><br>
            High Score Not Out (1=yes, 0=no): <input type="number" id="HS_not_out" required><br><br>
            <button type="submit">Predict Tier</button>
        </form>
        <h3 id="result"></h3>

        <script>
        document.getElementById('predictForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const data = {
                Mat: parseFloat(document.getElementById('Mat').value),
                Inns: parseFloat(document.getElementById('Inns').value),
                NO: parseFloat(document.getElementById('NO').value),
                Runs: parseFloat(document.getElementById('Runs').value),
                HS: parseFloat(document.getElementById('HS').value),
                BF: parseFloat(document.getElementById('BF').value),
                SR: parseFloat(document.getElementById('SR').value),
                Hundreds: parseFloat(document.getElementById('Hundreds').value),
                Fifties: parseFloat(document.getElementById('Fifties').value),
                Ducks: parseFloat(document.getElementById('Ducks').value),
                HS_not_out: parseInt(document.getElementById('HS_not_out').value)
            };
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(data)
            });
            const result = await response.json();
            document.getElementById('result').innerText = 'Predicted Tier: ' + result.predicted_tier;
        });
        </script>
    </body>
    </html>
    """
