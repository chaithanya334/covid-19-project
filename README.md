# COVID-19-Severity-Classifier

## Recommended Structure and Order

### 1. Project Title / Headline
A concise, descriptive name for the project.

🩺 **COVID-19 Patient Severity Classification (ML + Streamlit GUI)**
A ready-to-run web app that predicts COVID-19 patient severity (Mild / Moderate / Severe / Critical) from clinical features, using a pretrained Random Forest model with an interactive Streamlit interface.

### 2. Short Description / Purpose
1–2 sentences explaining what the app does and why it exists.

This project is an end-to-end machine learning application that classifies COVID-19 patient severity based on tabular clinical data. It includes a pretrained Random Forest model, a Streamlit UI for live predictions with charts, and a training script so the model can be retrained on custom data — built as an educational demo of a full ML deployment workflow.

> **Disclaimer:** Educational demo only. Not for clinical use.

### 3. Tech Stack
List the key technologies used to build the app.

The app was built using the following tools and technologies:<br>
• 🐍 Python (3.8–3.11) – Core language for data processing and model development.<br>
• 🌐 Streamlit – Powers the interactive UI (`app.py`) with sliders, inputs, and live charts.<br>
• 🌲 Scikit-learn (Random Forest) – Classification model pretrained on synthetic clinical data.<br>
• 🐼 Pandas & NumPy – Data handling and feature preprocessing.<br>
• 📊 Matplotlib – Renders probability and feature importance charts.<br>
• 📁 File Format – `.pkl` for the trained model, `.csv` for data, `.svg` for branding.

**Project Structure:**
```
covid_severity_ml_gui/
├── app.py                 # Streamlit UI
├── train_model.py         # Train from CSV or synthetic data
├── model_card.md          # Model details & limitations
├── requirements.txt
├── models/
│   └── covid_rf.pkl       # Pretrained RandomForest model
├── data/
│   └── schema.csv         # Column names & example row
└── static/
    └── brand.svg          # Logo
```

**Quick Start:**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

### 4. Data Source
More info on where the data comes from and how it's structured.

The model was pretrained on **synthetic clinical data** generated to mimic real-world COVID-19 severity patterns. The full schema includes 14 clinical input features: **age, sex, comorbidity_count, spo2, respiratory_rate, heart_rate, temperature_c, systolic_bp, diastolic_bp, crp_mg_l, d_dimer_ng_ml, lymphocyte_pct, neutrophil_pct,** and **chest_xray_score**, each mapped to a severity label (Mild / Moderate / Severe / Critical).

The app also supports retraining on a custom dataset:
```bash
python train_model.py --csv data/your_data.csv --model models/covid_rf.pkl
```
A custom CSV must include all input feature columns above plus a `severity` target column.

*Note: The demo version shown in the app screenshots uses a reduced 5-feature model (age, comorbidity_count, spo2, respiratory_rate, crp) trained on a smaller 100-row dataset with 3 severity classes (Mild/Moderate/Severe) — a lightweight variant of the full pipeline described above.*

### 5. Features / Highlights
The best project explanation format.

**• Business Problem**
During a health crisis like COVID-19, clinicians and health workers need a fast way to gauge how severe a patient's condition might be based on vital signs and lab values — without waiting on manual scoring or specialist review.

Key questions such as:
- How severe is this patient's case likely to be, given their vitals?
- Which clinical factors matter most in predicting severity?
- How confident is the model in its prediction?

… are difficult to answer quickly without a structured tool.

**• Goal of the Project**
To deliver an interactive app that:
- Lets a user input a patient's clinical values through sliders and steppers.
- Instantly predicts severity level with an associated confidence score.
- Explains which features drive the model's decisions, for transparency.
- Supports retraining on new data via a dedicated training script.

**• Walkthrough of Key Features**
- **Controls Panel (Left Sidebar)**
  Sliders and steppers let users input clinical values — e.g., Age 55, Comorbidity Count 1, SpO₂ 94%, Respiratory Rate 20, CRP 35.
- **Dataset & Model Panel**
  Displays a live preview of the training dataset (first 8 rows) and confirms the loaded model file, giving transparency into what data and model are powering the prediction.
- **Predict Panel**
  A single "Predict" button runs the inputs through the Random Forest model and returns the predicted severity — e.g., **Mild**, with **60% confidence** — alongside a probability breakdown (Mild 60.00%, Moderate 28.50%, Severe 11.50%).
- **Feature Importance – Permutation Importance (Bar Chart)**
  Ranks input features by how much they impact model predictions. CRP and SpO₂ emerge as the most influential features, followed by age, respiratory rate, and comorbidity count.
- **Cumulative Feature Importance (Line Chart)**
  Shows how quickly the top features account for the model's total predictive power — the top two features (CRP, SpO₂) alone capture roughly 80% of cumulative importance.

**• Impact & Insights**
- **Faster Triage:** Gives a quick, explainable severity estimate from a handful of easily measurable vitals.
- **Model Transparency:** The feature importance panel shows CRP (inflammation marker) and SpO₂ (oxygen saturation) as the strongest predictors, aligning with known clinical risk factors.
- **Educational Value:** Demonstrates a full ML deployment workflow — data → training → pretrained model → interactive prediction UI — packaged as a runnable app.
- **Extensibility:** The full pipeline supports 14 clinical features and 4 severity classes, and can be retrained on custom data via `train_model.py`.

### 6. Screenshots / Demos
Show what the app looks like.

**App Interface – Controls, Prediction & Dataset Preview**
![COVID-19 Severity Classifier - Controls and Prediction](https://github.com/chaithanya334/COVID-19-Severity-Classifier/blob/main/covid_19_project.jpg)

**Feature Importance – Permutation & Cumulative Importance**
![COVID-19 Severity Classifier - Feature Importance](https://github.com/chaithanya334/COVID-19-Severity-Classifier/blob/main/covid_19_project_2.jpg)
