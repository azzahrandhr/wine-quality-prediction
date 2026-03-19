import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #fdf6f0; }
    [data-testid="stSidebar"] { background-color: #2c1a2e; }
    [data-testid="stSidebar"] label { color: #f0d9c8 !important; font-size: 0.85rem; }
    [data-testid="stSidebar"] h2 { color: #e8c5a0; }
    [data-testid="stSidebar"] p { color: #c9a88a; }
    .result-card {
        background: #fff;
        border-left: 6px solid #8B0000;
        padding: 1.2rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .quality-score { font-size: 3rem; font-weight: 700; color: #8B0000; }
    .confidence-badge { font-size: 1.1rem; color: #555; margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)

# ── MODEL TRAINING (cached once per session) ──────────────────────────────────
@st.cache_resource(show_spinner="Training model on UCI wine data...")
def load_and_train():
    red = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "wine-quality/winequality-red.csv", sep=";"
    )
    white = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "wine-quality/winequality-white.csv", sep=";"
    )
    red["winetype"] = "Red"
    white["winetype"] = "White"
    df = pd.concat([red, white], ignore_index=True)

    feature_cols = [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide",
        "density", "pH", "sulphates", "alcohol",
    ]

    X = df[feature_cols]
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42
    )
    model.fit(X_train, y_train)
    test_acc = (model.predict(X_test) == y_test).mean()

    # Per-type medians for slider defaults
    medians = df.groupby("winetype")[feature_cols].median()
    return model, feature_cols, model.classes_, test_acc, medians

model, FEATURE_COLS, CLASSES, TEST_ACC, MEDIANS = load_and_train()

# ── SLIDER CONFIG: (min, max, step) ──────────────────────────────────────────
SLIDER_CFG = {
    "fixed acidity":        (3.8,   15.9,  0.1),
    "volatile acidity":     (0.08,   1.58, 0.01),
    "citric acid":          (0.0,    1.66, 0.01),
    "residual sugar":       (0.6,   65.8,  0.1),
    "chlorides":            (0.009,  0.611, 0.001),
    "free sulfur dioxide":  (1.0,  289.0,  1.0),
    "total sulfur dioxide": (6.0,  440.0,  1.0),
    "density":              (0.987, 1.039, 0.0001),
    "pH":                   (2.72,   4.01, 0.01),
    "sulphates":            (0.22,   2.0,  0.01),
    "alcohol":              (8.0,   14.9,  0.1),
}

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍷 Wine Input")

    wine_type = st.selectbox("Wine Type", ["Red", "White"])
    st.markdown("---")
    st.markdown("**Physicochemical Characteristics**")

    defaults = MEDIANS.loc[wine_type]

    inputs = {}
    for feat in FEATURE_COLS:
        lo, hi, step = SLIDER_CFG[feat]
        default = float(np.clip(defaults[feat], lo, hi))
        inputs[feat] = st.slider(
            feat.title(), min_value=lo, max_value=hi, value=default, step=step
        )

    st.markdown("---")
    st.caption(f"Model test accuracy: **{TEST_ACC:.1%}**")
    st.caption("Trained on UCI red + white wine dataset (6,497 samples)")

# ── PREDICTION ────────────────────────────────────────────────────────────────
X_input = pd.DataFrame([{f: inputs[f] for f in FEATURE_COLS}])
predicted_quality = model.predict(X_input)[0]
proba = model.predict_proba(X_input)[0]
confidence = proba.max() * 100

if predicted_quality <= 4:
    band, band_color = "Below Average", "#c0392b"
elif predicted_quality <= 6:
    band, band_color = "Average", "#e67e22"
else:
    band, band_color = "Premium", "#27ae60"

# ── MAIN PANEL ────────────────────────────────────────────────────────────────
st.markdown("# 🍷 Wine Quality Predictor")
st.markdown(
    "Select wine type and adjust the characteristics on the left. "
    "The model predicts a **quality score (3–9)** instantly."
)
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown(f"""
    <div class="result-card">
      <div style="font-size:0.85rem;color:#999;text-transform:uppercase;letter-spacing:.05em">
        Predicted Quality
      </div>
      <div class="quality-score">{predicted_quality} / 9</div>
      <div class="confidence-badge">Confidence: {confidence:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-card">
      <div style="font-size:0.85rem;color:#999;text-transform:uppercase;">Quality Band</div>
      <div style="font-size:1.6rem;font-weight:700;color:{band_color};">{band}</div>
      <div style="font-size:0.9rem;color:#888;margin-top:.3rem">
        {wine_type} Wine
      </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("#### Probability Distribution Across Quality Classes")

    bar_colors = ["#8B0000" if c == predicted_quality else "#d4a0a0" for c in CLASSES]

    fig, ax = plt.subplots(figsize=(8, 3.8))
    fig.patch.set_facecolor("#fdf6f0")
    ax.set_facecolor("#fdf6f0")

    bars = ax.bar([str(c) for c in CLASSES], proba * 100,
                  color=bar_colors, edgecolor="white", linewidth=0.8)

    for bar, pct in zip(bars, proba * 100):
        if pct > 1.0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.4, f"{pct:.1f}%",
                    ha="center", va="bottom", fontsize=8, color="#333")

    ax.set_xlabel("Quality Score", fontsize=10)
    ax.set_ylabel("Probability (%)", fontsize=10)
    ax.set_title(f"Quality Probabilities — {wine_type} Wine", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(proba * 100) * 1.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    st.pyplot(fig)

# ── EXPANDER: input table ─────────────────────────────────────────────────────
with st.expander("View entered feature values"):
    display_df = pd.DataFrame({
        "Feature": list(inputs.keys()),
        "Value": list(inputs.values()),
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)
