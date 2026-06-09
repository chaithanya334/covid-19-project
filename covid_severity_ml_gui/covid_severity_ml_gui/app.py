import os
import joblib
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="COVID-19 Severity Classifier", page_icon="🩺", layout="wide")

# preserve your original header/branding exactly
with open("static/brand.svg","r",encoding="utf-8") as f:
    brand_svg = f.read()

st.markdown(f"""
<style>
.header {{
  display:flex; align-items:center; gap:16px; padding:16px 20px;
  background: linear-gradient(90deg, #ff7a7a, #ffb36b);
  color:white; border-radius:18px; box-shadow: 0 8px 24px rgba(0,0,0,.15);
}}
.brand {{display:flex; align-items:center; gap:10px}}
.brand svg {{height:40px}}
.tag {{font-size:12px; opacity:.9; border:1px solid rgba(255,255,255,.5); padding:2px 8px; border-radius:999px}}
.card {{
  background: white; padding: 18px; border-radius: 16px;
  box-shadow: 0 10px 20px rgba(0,0,0,.06);
  border:1px solid rgba(0,0,0,.06);
}}
.metric {{ font-size: 32px; font-weight: 800; letter-spacing: -0.5px; }}
.help {{ color:#666 }}
</style>
<div class="header">
  <div class="brand">{brand_svg}<span style="font-weight:800;font-size:20px">COVID-19 Severity Classifier</span></div>
  <span class="tag">Demo • Random Forest</span>
</div>
""", unsafe_allow_html=True)

# Paths & filenames
DATA_PATH = "data"
MODEL_PATH = "models"
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

DATA_FILE = os.path.join(DATA_PATH, "covid_reduced_100.csv")
MODEL_FILE = os.path.join(MODEL_PATH, "covid_rf_reduced_5feat.pkl")

# -------------------------
# CHOSEN FEATURES (exact order used for train/load/predict)
# -------------------------
FEATURES = ["age", "comorbidity_count", "spo2", "resp_rate", "crp"]

# -------------------------
# Sidebar: only the five features shown
# -------------------------
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=55, step=1)
com = st.sidebar.number_input("Comorbidity Count", min_value=0, max_value=10, value=1, step=1)
spo2 = st.sidebar.slider("SpO₂ (%)", min_value=70.0, max_value=100.0, value=94.0, step=0.1)
rr = st.sidebar.slider("Respiratory Rate (breaths/min)", min_value=8.0, max_value=50.0, value=20.0, step=0.1)
crp = st.sidebar.slider("CRP (mg/L)", min_value=0.0, max_value=300.0, value=35.0, step=0.1)

# map for building input vector (keys must match model FEATURES names)
input_values_map = {
    "age": age,
    "comorbidity_count": com,
    "spo2": float(spo2),
    "resp_rate": float(rr),
    "crp": float(crp)
}

# -------------------------
# Synthetic dataset generator (100 samples) matching the 5 features
# -------------------------
def generate_synthetic_dataset_5(n=100, seed=42):
    rng = np.random.RandomState(seed)
    ages = rng.randint(18, 90, size=n)
    coms = rng.poisson(1.0, size=n)
    spo2 = np.clip(rng.normal(95, 3, size=n).astype(float) - (coms * 0.6) - (ages - 50)/120.0, 70.0, 100.0)
    resp_rate = np.clip(rng.normal(18, 4, size=n) + (coms * 1.5) + (100 - spo2)/8.0, 8.0, 50.0)
    crp = np.clip(rng.normal(20, 25, size=n) + coms * 10 + (100 - spo2) * 1.2, 0.0, 300.0)

    # scoring heuristic to assign severity
    score = (
        (ages >= 65).astype(int) * 1.5 +
        (coms >= 2).astype(int) * 1.8 +
        ((spo2 < 94) & (spo2 >= 90)).astype(int) * 1.2 +
        (spo2 < 90).astype(int) * 3.0 +
        (resp_rate >= 25).astype(int) * 1.5 +
        (crp >= 50).astype(int) * 1.2
    )
    labels = []
    for s in score:
        if s >= 4.0:
            labels.append("Severe")
        elif s >= 2.0:
            labels.append("Moderate")
        else:
            labels.append("Mild")

    df = pd.DataFrame({
        "age": ages,
        "comorbidity_count": coms,
        "spo2": np.round(spo2, 1),
        "resp_rate": np.round(resp_rate, 1),
        "crp": np.round(crp, 1),
        "severity": labels
    })
    return 

# -------------------------
# Train/save model (expects FEATURES order above)
# -------------------------
def train_and_save_model_5(df, model_path=MODEL_FILE):
    X = df[FEATURES].values
    y = df["severity"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    payload = {
        "model": clf,
        "features": FEATURES,
        "classes": list(clf.classes_)
    }
    joblib.dump(payload, model_path)
    preds = clf.predict(X_test)
    cm = confusion_matrix(y_test, preds, labels=payload["classes"])
    report = classification_report(y_test, preds, output_dict=True)
    return payload, cm, report, (X_test, y_test, preds)

# -------------------------
# Robust model loader
# -------------------------
def load_model_now(path=MODEL_FILE):
    if not os.path.exists(path):
        return None, f"Model file not found at: {path}"
    try:
        payload = joblib.load(path)
    except Exception as e:
        return None, f"joblib.load failed: {e}"
    if not isinstance(payload, dict):
        return None, "Loaded model is not a dict (expected keys 'model','features','classes')."
    req = ("model","features","classes")
    if not all(k in payload for k in req):
        return None, f"Payload missing required keys. Present: {list(payload.keys())}"
    return payload, None

# -------------------------
# App layout: controls & dataset/model info
# -------------------------
left, right = st.columns([1, 1.4])

with left:
    st.header("Controls")
    if not os.path.exists(DATA_FILE):
        if st.button("Generate synthetic dataset (2000 samples)"):
            df = generate_synthetic_dataset_5(2000, seed=123)
            df.to_csv(DATA_FILE, index=False)
            st.success(f"Saved synthetic dataset -> {DATA_FILE}")
    else:
        st.info(f"Dataset found: `{DATA_FILE}` ({os.path.getsize(DATA_FILE)} bytes)")

    if not os.path.exists(MODEL_FILE):
        if st.button("Train model on dataset"):
            if not os.path.exists(DATA_FILE):
                st.error("No dataset found. Please generate dataset first.")
            else:
                df = pd.read_csv(DATA_FILE)
                payload, cm, report, test_data = train_and_save_model_5(df)
                st.success("Model trained and saved.")
                st.write("Confusion matrix (test split):")
                st.write(cm)
                st.write("Classification report (test split):")
                st.write(pd.DataFrame(report).transpose())
    else:
        st.success(f"Model available: `{MODEL_FILE}`")

    st.markdown("---")
    st.markdown("### Predict (5 inputs)")
    if st.button("Predict"):
        payload, err = load_model_now()
        if err is not None:
            st.error(f"No trained model found. {err}")
        else:
            model = payload["model"]
            model_features = payload["features"]
            CLASSES = payload["classes"]

            # build input vector according to model_features
            try:
                x_row = [input_values_map[f] for f in model_features]
                x = np.array([x_row], dtype=float)
            except KeyError as ke:
                st.error(f"Model expects features {model_features} but app doesn't have: {ke}")
                st.stop()
            except Exception as e:
                st.error(f"Failed building input vector: {e}")
                st.stop()

            try:
                proba = model.predict_proba(x)[0]
            except Exception as e:
                st.error(f"Model predict failed: {e}")
            else:
                idx = int(np.argmax(proba))
                label = CLASSES[idx]
                conf = float(proba[idx])
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.write("**Predicted severity:**")
                st.markdown(f"<div style='font-size:28px; font-weight:800'>{label}</div>", unsafe_allow_html=True)
                st.caption(f"Confidence: {conf:.2%}")
                st.markdown('</div>')

                # Horizontal probability bar
                fig, ax = plt.subplots(figsize=(6,2))
                y_pos = np.arange(len(CLASSES))
                ax.barh(y_pos, proba, align='center')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(CLASSES)
                ax.set_xlim(0,1)
                ax.set_xlabel("Probability")
                ax.invert_yaxis()
                for i, v in enumerate(proba):
                    ax.text(v + 0.01, i, f"{v:.2%}", va='center')
                st.pyplot(fig)

with right:
    st.header("Dataset & Model")
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        st.markdown("**Dataset preview (first 8 rows)**")
        st.dataframe(df.head(8))
        st.markdown(f"Rows: {len(df)}")
        st.download_button("Download dataset (CSV)", df.to_csv(index=False), file_name="covid_reduced_100.csv", mime="text/csv")
    else:
        st.info("No dataset yet. Click 'Generate synthetic dataset' on the left.")

    st.markdown("---")
    payload, err = load_model_now()
    if payload is None:
        st.info("No trained model available. Train the model to see feature importances and evaluation.")
    else:
        model = payload["model"]
        model_features = payload["features"]
        st.markdown("### Feature Importance")
        imp = getattr(model, "feature_importances_", None)
        perm_done = False
        if os.path.exists(DATA_FILE):
            try:
                df_all = pd.read_csv(DATA_FILE)
                if all(f in df_all.columns for f in model_features) and "severity" in df_all.columns:
                    X_all = df_all[model_features].values
                    y_all = df_all["severity"].values
                    with st.spinner("Computing permutation importance..."):
                        p_imp = permutation_importance(model, X_all, y_all, n_repeats=12, random_state=42, n_jobs=-1)
                    perm_means = p_imp.importances_mean
                    perm_stds = p_imp.importances_std
                    perm_done = True
                else:
                    st.info("Dataset found but doesn't contain the model's feature columns; falling back to model importances.")
            except Exception as e:
                st.warning(f"Permutation importance failed: {e}. Falling back to model importances.")

        if perm_done:
            df_imp = pd.DataFrame({
                "feature": model_features,
                "perm_mean": perm_means,
                "perm_std": perm_stds
            }).sort_values("perm_mean", ascending=False).reset_index(drop=True)
            base_vals = df_imp["perm_mean"].values
            fig = plt.figure(figsize=(6,3))
            y_pos = np.arange(len(df_imp))
            plt.barh(y_pos, df_imp["perm_mean"], xerr=df_imp["perm_std"], align='center')
            plt.yticks(y_pos, df_imp["feature"])
            plt.gca().invert_yaxis()
            plt.xlabel("Permutation importance (mean)")
            plt.title("Permutation Feature Importance")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            if imp is None:
                st.info("Model does not expose feature_importances_.")
                df_imp = pd.DataFrame({"feature": model_features, "imp": [0.0]*len(model_features)})
                base_vals = df_imp["imp"].values
            else:
                norm_imp = imp / (imp.sum() if imp.sum() > 0 else 1.0)
                df_imp = pd.DataFrame({"feature": model_features, "imp": norm_imp}).sort_values("imp", ascending=False).reset_index(drop=True)
                base_vals = df_imp["imp"].values
                fig = plt.figure(figsize=(6,3))
                y_pos = np.arange(len(df_imp))
                plt.barh(y_pos, df_imp[df_imp.columns[1]], align='center')
                plt.yticks(y_pos, df_imp["feature"])
                plt.gca().invert_yaxis()
                plt.xlabel("Normalized importance")
                plt.title("Random Forest Feature Importances (normalized)")
                plt.tight_layout()
                st.pyplot(fig)

        # cumulative importance
        cum = (base_vals / base_vals.sum()).cumsum() if base_vals.sum() > 0 else np.zeros_like(base_vals)
        fig2 = plt.figure(figsize=(5,2.5))
        plt.plot(range(1, len(cum)+1), cum, marker='o')
        plt.xticks(range(1, len(cum)+1), df_imp["feature"], rotation=45, ha='right')
        plt.ylabel("Cumulative importance")
        plt.title("Cumulative Feature Importance")
        plt.ylim(0,1.02)
        plt.tight_layout()
        st.pyplot(fig2)

        # lollipop / stem (no use_line_collection)
        fig3 = plt.figure(figsize=(6,3))
        x_vals = np.arange(len(df_imp))
        plt.stem(x_vals, base_vals, basefmt=" ")
        plt.xticks(x_vals, df_imp["feature"], rotation=45, ha='right')
        plt.ylabel("Importance")
        plt.title("Lollipop view — Feature importance")
        plt.tight_layout()
        st.pyplot(fig3)

        # table + download
        if perm_done:
            df_out = df_imp.rename(columns={"perm_mean": "importance_mean", "perm_std": "importance_std"})
        else:
            df_out = df_imp.rename(columns={df_imp.columns[1]: "importance"})
        st.write("Importance table (top → bottom):")
        st.dataframe(df_out)
        st.download_button("Download importances CSV", df_out.to_csv(index=False).encode("utf-8"), file_name="feature_importances_5feat.csv", mime="text/csv")

st.markdown("---")
st.markdown("**Disclaimer:** Educational demo only. Not for clinical use.")            






