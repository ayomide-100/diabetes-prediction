# 🩺 Diabetes Prediction System using Machine Learning

A machine learning project that classifies and predicts whether a patient is diabetic or non-diabetic based on diagnostic measurements. The project implements comprehensive Exploratory Data Analysis (EDA), feature preprocessing, and compares **Support Vector Machines (SVM)** and **Logistic Regression** classifiers.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset Summary](#-dataset-summary)
- [Project Structure](#-project-structure)
- [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
- [Model Training & Evaluation](#-model-training--evaluation)
- [Saved Models](#-saved-models)
- [Installation & Setup](#-installation--setup)
- [How to Use & Run](#-how-to-use--run)
- [Future Enhancements](#-future-enhancements)

---

## 🔬 Project Overview
Diabetes mellitus is a chronic disease requiring early detection to prevent long-term metabolic complications. This project uses clinical and diagnostic variables to build classification models:
- **Linear Support Vector Classifier (`SVC(kernel='linear')`)**
- **Logistic Regression Classifier**

Both models are trained and evaluated with standard scaling and stratified validation, with trained model artifacts serialized for downstream inference.

---

## 📊 Dataset Summary
The project utilizes the **Pima Indians Diabetes Dataset** located in [`data/diabetes.csv`](data/diabetes.csv):
- **Total Records:** 768 instances
- **Total Features:** 8 numerical predictor features + 1 binary target (`Outcome`)
- **Class Distribution:**
  - `0 (Non-Diabetic)`: 500 patients (65.1%)
  - `1 (Diabetic)`: 268 patients (34.9%)

### Feature Description:
| Feature | Description |
| :--- | :--- |
| **`Pregnancies`** | Number of times pregnant |
| **`Glucose`** | Plasma glucose concentration a 2 hours in an oral glucose tolerance test |
| **`BloodPressure`** | Diastolic blood pressure ($mm\ Hg$) |
| **`SkinThickness`** | Triceps skin fold thickness ($mm$) |
| **`Insulin`** | 2-Hour serum insulin ($\mu U/ml$) |
| **`BMI`** | Body mass index ($weight\ in\ kg / (height\ in\ m)^2$) |
| **`DiabetesPedigreeFunction`** | Diabetes pedigree function (genetic score) |
| **`Age`** | Patient age (years) |
| **`Outcome`** | Target class variable (`0` = Non-Diabetic, `1` = Diabetic) |

---

## 📁 Project Structure

```bash
diabetes-prediction-svm/
├── data/
│   └── diabetes.csv                 # Raw dataset
├── models/                          # Serialized trained models & scalers
│   ├── svm_model.joblib
│   ├── logistic_regression_model.joblib
│   └── scaler.joblib
├── notebooks/
│   └── modelling.ipynb              # Main workflow notebook (EDA, training, evaluation, export)
├── src/
│   ├── __init__.py                  # Package initializer
│   └── eda_functions.py             # Modular reusable EDA and visualization suite
├── .gitignore                       # Git ignore file for python/jupyter
├── requirements.txt                 # Project dependencies
└── Readme.md                        # Documentation and guide
```

---

## 🔍 Exploratory Data Analysis (EDA)
Modularized within [`src/eda_functions.py`](src/eda_functions.py), the EDA workflow covers:
1. **Missing & Zero Value Detection:** Identifies physiologically impossible zeros in features like `Insulin` (48.7%), `SkinThickness` (29.6%), `BloodPressure` (4.7%), `BMI` (1.4%), and `Glucose` (0.7%).
2. **Distribution & Outlier Analysis:** Boxplots and IQR analysis across all 8 features.
3. **Correlation Heatmap:** Highlights `Glucose` ($r \approx 0.47$) and `BMI` ($r \approx 0.29$) as having the strongest positive correlation with the target outcome.
4. **Stratified Feature Comparisons:** Visualizing feature shifts between diabetic and non-diabetic cohorts.

---

## ⚙️ Model Training & Evaluation

### 1. Data Preprocessing
- **Feature Scaling:** Applied `StandardScaler` to normalize features with vastly different scales (e.g., `Insulin` vs. `DiabetesPedigreeFunction`).
- **Stratified Split:** Split into 75% Training (576 samples) and 25% Testing (192 samples) maintaining class proportions.

### 2. Performance Comparison

| Metric | Linear SVM (`SVC`) | Logistic Regression | Best Performer |
| :--- | :---: | :---: | :--- |
| **Train Accuracy** | 78.99% | 79.51% | Comparable fit |
| **Test Accuracy** | 70.83% | **73.44%** | **Logistic Regression (+2.61%)** |
| **10-Fold CV Accuracy** | 77.22% | — | Linear SVM (stable generalization) |
| **Precision (Diabetic - Class 1)** | 60.00% | **65.00%** | **Logistic Regression** (fewer false positives: 19 vs 22) |
| **Recall / Sensitivity (Diabetic)** | 49.25% | **52.24%** | **Logistic Regression** (detected 35 vs 33 cases) |
| **F1-Score (Diabetic - Class 1)** | 0.54 | **0.58** | **Logistic Regression (+0.04)** |
| **Macro F1-Score** | 0.66 | **0.69** | **Logistic Regression** |

---

## 💾 Saved Models
Trained estimators and preprocessing pipelines are saved in the `models/` directory using `joblib`:
- `models/svm_model.joblib`: Trained Support Vector Classifier
- `models/logistic_regression_model.joblib`: Trained Logistic Regression Classifier
- `models/scaler.joblib`: Fitted `StandardScaler` for preprocessing new input data

### Loading a Model for Inference:
```python
import joblib
import numpy as np

# Load the scaler and model
scaler = joblib.load('models/scaler.joblib')
model = joblib.load('models/logistic_regression_model.joblib')

# Example input patient features
# [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]
raw_features = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
scaled_features = scaler.transform(raw_features)

prediction = model.predict(scaled_features)
print("Diabetic" if prediction[0] == 1 else "Non-Diabetic")
```

---

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<YOUR_GITHUB_USERNAME>/diabetes-prediction-svm.git
   cd diabetes-prediction-svm
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 How to Use & Run

1. Launch Jupyter Notebook or VS Code Jupyter extension:
   ```bash
   jupyter notebook
   ```
2. Open [`notebooks/modelling.ipynb`](notebooks/modelling.ipynb).
3. Run all cells sequentially to execute:
   - Data loading and analysis
   - Exploratory visualizations
   - Standardization and train/test splitting
   - SVM and Logistic Regression training
   - Model evaluation & export to `models/`

---

## 🔮 Future Enhancements
- **Missing Value Imputation:** Implement KNN or Iterative Imputer for zero values in `Insulin` and `SkinThickness`.
- **Non-Linear Kernels & Hyperparameter Tuning:** Evaluate Radial Basis Function (`rbf`) and Polynomial SVM kernels with `GridSearchCV`.
- **Ensemble Techniques:** Benchmark against Random Forests, Gradient Boosting, and XGBoost.
- **Web App / API Deployment:** Wrap the exported model in a Streamlit dashboard or FastAPI endpoint for real-time predictions.
