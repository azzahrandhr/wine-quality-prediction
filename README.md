# Wine Quality Prediction

> My first vibe-coded project — built with curiosity, Claude, and a lot of wine data.

A machine learning project that predicts the quality of red and white wines based on their physicochemical characteristics. Includes a full exploratory data analysis notebook and an interactive Streamlit web app for wine sellers to get instant quality predictions.

---

## What this project does

Given measurable chemical properties of a wine sample (acidity, sugar, alcohol content, etc.), in this project I:

1. **Explores the data** — visualises distributions, correlations, and outliers across 6,497 wine samples
2. **Engineers new features** — creates ratios and interaction terms that amplify signals in the data
3. **Trains and compares models** — Logistic Regression, Random Forest, and Gradient Boosting
4. **Serves predictions via a web app** — wine sellers can input characteristics and instantly see a quality score (3–9) with confidence probability

---

## Dataset

- **Source:** [UCI Machine Learning Repository — Wine Quality](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **Size:** 6,497 samples (4,898 white + 1,599 red)
- **Features:** 11 physicochemical measurements (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free/total sulfur dioxide, density, pH, sulphates, alcohol)
- **Target:** Quality score (integer, 3–9)

No download needed — the app and notebook load the data directly from the UCI URL.

---

## Project structure

```
wine-quality-prediction/
├── wine-data-exploration.ipynb   # Full EDA, feature engineering, model training & evaluation
├── app.py                        # Streamlit web app for wine quality prediction
└── requirements.txt              # Python dependencies
```

---

## Notebook walkthrough (`wine-data-exploration.ipynb`)

| Section | What it covers |
|---|---|
| EDA 1 — Data Quality | Missing values, data types, duplicate row check |
| EDA 2 — Correlation | Heatmap of all feature correlations |
| EDA 3 — Distributions | KDE plots of every feature split by wine type |
| EDA 4 — Outliers | IQR-based outlier counts, z-scored boxplots |
| Feature Engineering | 3 new features: SO2 ratio, acid ratio, density × alcohol |
| Model Building | Logistic Regression (L2), Random Forest, Gradient Boosting |
| Evaluation | Confusion matrices, classification reports, ROC curves, feature importances |

---

## Streamlit app (`app.py`)

A clean, interactive web app built for wine sellers:

- Select **wine type** (Red / White) from a dropdown — sliders auto-fill with typical values for that type
- Adjust **11 physicochemical sliders** to describe the wine sample
- See the predicted **quality score**, **confidence %**, and **quality band** (Below Average / Average / Premium)
- View a **probability bar chart** across all quality classes (3–9)

### Run it locally

```bash
# 1. Clone the repo
git clone https://github.com/nadhirahendra/wine-quality-prediction.git
cd wine-quality-prediction

# 2. Create and activate a virtual environment (or conda env)
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

The app will train a Gradient Boosting model on startup (takes ~30 seconds the first time), then cache it for the rest of the session.

---

## Model results

| Model | CV Accuracy | Test Accuracy | ROC-AUC |
|---|---|---|---|
| Logistic Regression (L2) | ~97.5% | ~97.5% | ~0.997 |
| Random Forest | ~99.4% | ~99.5% | ~0.999 |
| **Gradient Boosting** | ~99.3% | ~99.5% | ~0.999 |

Gradient Boosting was selected for the app — it produces well-calibrated probabilities across all quality classes, making the confidence scores meaningful.

---

## What I learned building this

This was my first time vibe-coding a project from scratch with an AI pair programmer. A few things that stood out:

- **EDA always surprises you** — 1,177 duplicate rows and heavily skewed features like residual sugar were not obvious until plotted
- **Feature engineering compounds value** — the SO2 ratio and acid ratio features improved both interpretability and model accuracy
- **Scaling matters for distance-based models** — KNN accuracy jumped from 96.5% to 98.6% after adding StandardScaler
- **Tree models are robust** — Random Forest and Gradient Boosting hit 99.5% accuracy without any scaling, handling outliers naturally
- **Building an app changes how you think about the model** — you stop caring about the coefficients and start caring about whether the prediction makes intuitive sense to a real user

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
```

---

## License

MIT
