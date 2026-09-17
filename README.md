# Travel Safety Advisor

An interactive machine-learning application that estimates whether a travel destination may be **safe or unsafe** based on location, outbreak context, environmental conditions, travel date, and traveler health-risk indicators.

The project includes a reproducible training script and a Streamlit interface that uses the exact same preprocessing artifacts created during training.

> **Important:** This repository is an educational prototype. The included CSV contains synthetic, rule-generated labels and must not be used for medical, public-health, or travel decisions.

## Features

- Location selection by Indian state/UT and district
- Travel-date input with week, month, and year features
- Outbreak context including reported cases and coordinates
- Environmental inputs for temperature, precipitation, leaf-area index, and AQI
- Optional live weather lookup through OpenWeatherMap
- Manual weather fallback when the API is unavailable
- Health-risk checklist covering respiratory, cardiovascular, neurological, and other conditions
- Random Forest safety classification with a probability-based safety score
- Feature-importance chart explaining the strongest model factors
- Saved encoders, imputer, scaler, model, dropdown options, and evaluation metrics
- Cross-platform paths using `pathlib`

## Application Preview

Run the application locally with the command below:

```bash
streamlit run app.py
```

The interface collects the model inputs, applies the saved preprocessing pipeline, and displays a safety prediction plus a 0-100 safety score.

## Tech Stack

- Python 3.9+
- Streamlit for the web interface
- pandas and NumPy for data preparation
- scikit-learn for preprocessing, training, and evaluation
- Random Forest for classification
- joblib for model artifact persistence
- Plotly for the safety gauge and feature visualization
- Requests and python-dotenv for optional weather integration

## Project Structure

```text
Travel-Safety-Advisor-main/
|-- app.py                         # Streamlit application
|-- train_model.py                 # Training and evaluation pipeline
|-- health_crisis_with_safety.csv  # Training dataset
|-- requirements.txt               # Python dependencies
|-- saved/
    |-- random_forest_model.joblib # Trained classifier
    |-- imputer.joblib              # Median-value imputer
    |-- scaler.joblib               # StandardScaler
    |-- encoders.joblib             # Categorical encoders
    |-- feature_names.joblib        # Training feature order
    |-- state_options.joblib        # State/UT dropdown values
    |-- district_by_state.joblib    # District dropdown mapping
    |-- metrics.txt                 # Held-out evaluation report
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/travel-safety-advisor.git
cd travel-safety-advisor
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Optional Weather API Setup

The app works without an API key by using manual temperature input. To enable live weather retrieval, create a `.env` file in the project root:

```env
OPENWEATHER_API_KEY=your_openweathermap_api_key
```

Do not commit `.env` or any secret API key to GitHub. Add `.env` to `.gitignore` before publishing the project.

## Run the Project

The saved model artifacts are included, so the app can be launched immediately after installation:

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## Retrain the Model

Retrain whenever the CSV changes or you want to regenerate the saved artifacts:

```bash
python train_model.py
```

The training pipeline:

1. Loads `health_crisis_with_safety.csv`.
2. Removes target-leaking and unavailable columns such as `cluster`, `safe_or_unsafe`, `Deaths`, and `Disease`.
3. Converts the `Cases` column to numeric values.
4. Encodes `week_of_outbreak`, `state_ut`, and `district`, including an explicit unknown category.
5. Creates a stratified 80/20 train-test split.
6. Fits a median imputer and standard scaler on the training split only.
7. Trains a class-balanced Random Forest classifier.
8. Reports accuracy, classification metrics, ROC-AUC, and feature importance.
9. Refits the classifier on all available rows and saves the deployment artifacts under `saved/`.

## Dataset and Model Inputs

The model uses location, time, environmental, outbreak, and health-condition signals:

- Location: `state_ut`, `district`, `Latitude`, `Longitude`
- Time: `week_of_outbreak`, `day`, `mon`, `year`
- Outbreak/environment: `Cases`, `preci`, `LAI`, `Temp`, `AQI`
- Health indicators: `heat_stroke`, `hypothermia`, `migraine`, `allergies`, `dehydration`, `skin_disease`, `sunburn_risk`, `respiratory_infection`, `joint_pain`, `stroke_risk`, `vertigo_risk`, `asthma`, `low_bp`, `cardiac_arrest`, `back_pain`, `paralysis_risk`, and `pressure_problem`

The target is `safe_or_unsafe_binary`, where `1` represents safe and `0` represents unsafe.

## Evaluation Snapshot

The checked-in training run reports the following on a stratified 20% holdout set of 1,797 rows:

| Metric | Score |
|---|---:|
| Accuracy | 1.000 |
| ROC-AUC | 1.000 |
| Unsafe precision/recall/F1 | 1.00 / 1.00 / 1.00 |
| Safe precision/recall/F1 | 1.00 / 1.00 / 1.00 |

These results should **not** be interpreted as real-world model performance. The dataset labels were generated from deterministic relationships between the input variables, so a model can reproduce the labeling rule unusually well. The strongest recorded factors include `pressure_problem`, `asthma`, `skin_disease`, `preci`, and `Temp`. Real deployment would require independently collected, representative, and clinically validated outcome data.

## Responsible Use

- This tool is not a medical diagnosis system or an official travel advisory.
- Do not use its output as a substitute for a doctor, local health authority, or government travel guidance.
- The model may reflect dataset bias and synthetic labeling assumptions.
- Never enter personally identifiable health information into a public deployment.
- Replace the sample dataset and revalidate the complete pipeline before any serious use.

## Publishing to GitHub

From the project directory:

```bash
git init
git add .
git commit -m "Initial Travel Safety Advisor project"
git branch -M main
git remote add origin https://github.com/<your-username>/travel-safety-advisor.git
git push -u origin main
```

Before the first push, verify that secrets and local environments are ignored. A suitable `.gitignore` should include:

```gitignore
.venv/
__pycache__/
*.py[cod]
.env
.streamlit/
```

## License

Add the license you want to use before publishing. For an open-source project, MIT is a common permissive option, but the final choice should match the project's ownership and intended use.
