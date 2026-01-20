# Credit Risk Prediction System – End-to-End ML Project

## 📌 Project Overview
This project predicts whether a loan applicant is **High Risk** or **Low Risk** using real-world financial and demographic data. The system simulates an **industry-grade ML workflow** from preprocessing to deployment.

---

## 📊 Dataset
- Features include:
  - Age, Income, Employment Type, Loan Amount, Loan Duration, Credit Score
  - Previous Defaults, Collateral, Dependents, Marital Status
- Target variable: `Risk` (High / Low)
- Data stored in: `data/credit_risk_data.csv`

---

## ⚙️ Workflow
1. Data loading & understanding
2. Data preprocessing & feature engineering
3. Exploratory Data Analysis (EDA)
4. Model building (Logistic Regression, Random Forest, Gradient Boosting)
5. Hyperparameter tuning & imbalance handling (SMOTE)
6. Model evaluation (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
7. Deployment using Streamlit

---

## 🤖 Models Implemented
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

**Random Forest** selected for deployment due to high accuracy and robustness.

---

## 📈 Key Insights
- High debt-to-income and total obligation ratios correlate with **High Risk**.
- Employment type, credit score, and previous defaults are strong predictors.
- Feature engineering and proper encoding improve model performance.

---

## 🚀 Deployment
- Deployed using **Streamlit**.
- Users can enter applicant details to get **High Risk / Low Risk** prediction.
- Input features include demographics, income, loan info, collateral, dependents, and marital status.

**Live App Link:** *(Add after deployment)*

---

## ▶️ How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
