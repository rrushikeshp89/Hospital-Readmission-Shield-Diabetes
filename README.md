# 🛡️ 30-Day Hospital Readmission Shield (Diabetes)

### A Machine Learning "Early Warning System" to Reduce Hospital Readmissions

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Executive Summary
Under the **Hospital Readmissions Reduction Program (HRRP)**, hospitals face severe penalties for high 30-day readmission rates. This project is a predictive engine designed to identify high-risk diabetic patients *before* they are discharged, allowing medical teams to prioritize interventions.

Unlike standard tutorials, this project implements a **Production-Grade Pipeline** featuring automated data quality tests, class imbalance handling, and an interactive dashboard for end-users.

---

## 🚀 Key Features & Engineering
* **Imbalanced Class Handling:** Achieved **55% Recall** on the minority class (Readmitted <30 days) using XGBoost's `scale_pos_weight` parameter (Target balance was 9:91).
* **Advanced Feature Engineering:**
    * **Service Utilization:** Aggregated historical ER, Inpatient, and Outpatient visits to capture "frequent flyer" behavior.
    * **ICD-9 Grouping:** Mapped 700+ complex medical codes into 9 high-level clinical categories (Circulatory, Respiratory, etc.).
* **Automated Quality Assurance (QA):** Implemented a "Gatekeeper" script that runs automated tests (Data Leakage checks, Probability range validation) before model deployment.
* **Explainable AI (XAI):** Integrated **SHAP (SHapley Additive exPlanations)** to provide clinician-friendly reasons for every risk score.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Modeling:** XGBoost (Gradient Boosting)
* **Evaluation:** Recall, AUC-ROC, Confusion Matrix
* **Explainability:** SHAP
* **Deployment:** Streamlit (Web App)

---

## 📊 Model Performance
The model was trained on the **UCI Diabetes 130-US Hospitals Dataset** (~70k clean records).

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **AUC-ROC** | **0.65** | Strong predictive power for this specific messy clinical dataset (State-of-the-Art is ~0.66). |
| **Recall (<30 days)** | **0.55** | Successfully flags more than half of all high-risk patients. |
| **Precision** | **0.14** | Intentionally prioritized Recall over Precision to minimize "False Negatives" (Missing a sick patient). |

---

## 📸 Dashboard Demo
*The Streamlit application allows doctors to input patient vitals and get an instant Risk Score.*

<img width="1919" height="933" alt="image" src="https://github.com/user-attachments/assets/0056ce59-88c0-4964-8ec2-f8e8ef5795a5" />

<img width="1919" height="931" alt="image" src="https://github.com/user-attachments/assets/2d79cdce-699b-4e74-b328-36f58d2324e1" />


**Top Predictors Found:**
1.  **Prior Service Utilization** (History of visits)
2.  **Discharge Location** (Sent to Skilled Nursing Facilities)
3.  **Age Group** (Older patients >70).

---

## 💻 Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/rrushikeshp89/readmission_shield.git](https://github.com/YOUR_USERNAME/readmission_shield.git)
cd readmission_shield
```
### 2. Install the Dependencies
```bash
pip install pandas numpy xgboost scikit-learn streamlit joblib shap
```
### 3. Run the Dashboard
```bash
streamlit run app.py
```

## 📂 Project Structure
```plaintext
readmission_shield/
│
├── data/                  # Raw dataset (GitIgnored if large)
├── notebooks/             # Jupyter Notebooks for EDA and Training
├── src/                   # Python scripts for cleaning and engineering
├── app.py                 # Streamlit Application (Frontend)
├── xgb_readmission_model.json  # Trained Model Artifact
├── feature_names.pkl      # Feature column mapping
└── README.md              # Project Documentation
```

## Streamlit Deployed App 
```bash
https://hospital-readmission-shield-diabetes-mxh3cfy4tg5movmukaphtm.streamlit.app/
```

## 🤝 Future Improvements
- **Hyperparameter Tuning:** Implement GridSearchCV or Optuna to optimize model parameters and improve AUC-ROC performance.
- **Dockerization:** Containerize the application to enable consistent environments and simplify cloud deployment.
- **API Development:** Expose the trained model via a FastAPI endpoint for seamless integration with Electronic Health Record (EHR) systems.

**Author**
**Rushikesh Panchal** Data Science Student | Software Testing Enthusiast https://www.linkedin.com/in/rushikesh-panchal

                        
