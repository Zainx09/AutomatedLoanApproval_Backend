# 🧠 Backend - Loan Approval System (ML + API)

This backend powers a smart loan application system using **Machine Learning** models and a secure API-based architecture. It handles prediction, data storage, and admin controls efficiently.

---

## 🔧 Technologies Used

- ⚙️ **Python (Flask)** — Lightweight web framework
- 🤖 **Machine Learning** — Logistic Regression & Random Forest
- 🛢️ **MySQL** — SQL database for storing applications
- 📂 **Joblib** — For model serialization
- 🔐 **Flask-CORS** — Handles cross-origin frontend requests

---

## 🧠 ML Models Overview

### 1️⃣ **Loan Approval Prediction**  
- **Type:** Classification (0 = No, 1 = Yes)  
- **Model:** Logistic Regression  
- **Input:** Income, Debt, Credit Score, etc.  
- **Training:** 80/20 split  
- **File:** `loan_approval_step1_logistic_model.joblib`  
- **Reason Chosen:** Fast, highly accurate (~95%), interpretable

### 2️⃣ **Loan Amount & Interest Rate Prediction**  
- **Type:** Regression  
- **Model:** Random Forest Regressor  
- **Input:** Income, Credit History, Loan Type, etc.  
- **Output:** `approved_amount`, `interest_rate`  
- **File:** `loan_approval_step2_randomforest_model.joblib`  
- **Reason Chosen:** High accuracy with fast prediction time

---

## 🔗 API Endpoints

| Method | Endpoint                                | Description                                           |
|--------|-----------------------------------------|-------------------------------------------------------|
| POST   | `/predict`                              | Predict loan approval based on user input            |
| GET    | `/predict/<applicant_id>`               | Fetch prediction result for a specific user |
| GET    | `/api/credit_details/<applicant_id>`    | Get credit-related details of a user                 |
| PUT    | `/api/application`                      | Update an existing application                       |
| POST   | `/api/application`                      | Submit a new loan application                        |
| GET    | `/api/application/<applicant_id>`       | Fetch full application by applicant ID               |
| POST   | `/api/calculate-loan`                   | Predict approved loan amount and interest rate       |
| GET    | `/admin/applications`                   | Admin: Get list of all submitted applications        |
| GET    | `/admin/users`                          | Admin: Get list of all users                         |


---


## ✅ Key Features

- 🤖 Machine Learning-driven decision making  
- 🔐 Clean RESTful API structure  
- 📦 Scalable folder structure for future microservices  
- ⚙️ Efficient MySQL integration  

---

## 📦 Installation

```bash
pip install -r requirements.txt
python app.py


