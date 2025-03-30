import joblib
import pandas as pd
from .denial_reasons import get_denial_reasons, DENIAL_MESSAGES  # Relative import from same package (extras)
# from ..models import (  # Relative import from parent app directory's models
#     loan_approval_step1_logistic_model, 
#     loan_approval_scaler, 
#     loan_approval_imputer_X,
#     loan_terms_step2_rf_model, 
#     loan_terms_scaler, 
#     loan_terms_imputer_X, 
# )


# # Use imported models directly
# approval_model = loan_approval_step1_logistic_model
# approval_scaler = loan_approval_scaler
# approval_imputer_X = loan_approval_imputer_X
# terms_model = loan_terms_step2_rf_model
# terms_scaler = loan_terms_scaler
# terms_imputer_X = loan_terms_imputer_X

import os

# Get the absolute path to the models directory relative to get_prediction.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Load models and preprocessors
approval_model = joblib.load(os.path.join(MODELS_DIR, 'loan_approval_step1_logistic_model.joblib'))
approval_scaler = joblib.load(os.path.join(MODELS_DIR, 'loan_approval_scaler.joblib'))
approval_imputer_X = joblib.load(os.path.join(MODELS_DIR, 'loan_approval_imputer_X.joblib'))
terms_model = joblib.load(os.path.join(MODELS_DIR, 'loan_terms_step2_rf_model.joblib'))
terms_scaler = joblib.load(os.path.join(MODELS_DIR, 'loan_terms_scaler.joblib'))
terms_imputer_X = joblib.load(os.path.join(MODELS_DIR, 'loan_terms_imputer_X.joblib'))


# Define expected columns
expected_columns = [
    "credit_score", "annual_income", "self_reported_debt", "self_reported_expenses",
    "requested_amount", "age", "province", "employment_status", "months_employed",
    "total_credit_limit", "credit_utilization", "num_open_accounts",
    "num_credit_inquiries", "payment_history", "estimated_debt",
    "total_monthly_debt", "DTI"
]

# Shared prediction function
def get_prediction(data):
    try:
        # Ensure data is a mutable dictionary
        data = data.copy()

        # Set default values for optional fields if missing
        defaults = {
            "credit_score": 0,
            "annual_income": 1,  # Avoid division by zero
            "self_reported_debt": 0,
            "self_reported_expenses": 0,
            "requested_amount": 0,
            "age": 0,
            "province": 0,
            "employment_status": 0,
            "months_employed": 0,
            "total_credit_limit": 0,
            "credit_utilization": 0,
            "num_open_accounts": 0,
            "num_credit_inquiries": 0,
            "payment_history": 0
        }
        for key, default in defaults.items():
            data[key] = float(data.get(key, default))

        # Calculate derived fields with fallback values
        data["estimated_debt"] = data["total_credit_limit"] * (data["credit_utilization"] / 100) * 0.03
        data["total_monthly_debt"] = data["self_reported_debt"] + data["estimated_debt"]
        
        # Calculate DTI, handle division by zero
        monthly_income = data["annual_income"] / 12
        if monthly_income == 0:
            data["DTI"] = float("inf")
        else:
            data["DTI"] = (
                (data["self_reported_debt"] + data["estimated_debt"] + (data["requested_amount"] * 0.03))
                / monthly_income
            ) * 100
        if (data["credit_score"] >= 700 and 
            data["DTI"] <= 35 and 
            data["credit_utilization"] <= 50 and 
            data["payment_history"] < 3):
            data["payment_history"] += 1
        # Convert to DataFrame, fill missing fields with NaN for imputer
        df = pd.DataFrame([data])
        df = df.reindex(columns=expected_columns, fill_value=pd.NA)  # Ensure all expected columns

        # Step 1: Predict approval
        approval_data_imputed = pd.DataFrame(approval_imputer_X.transform(df), columns=df.columns)
        approval_data_scaled = approval_scaler.transform(approval_data_imputed)
        approval_pred = approval_model.predict(approval_data_scaled)[0]
        approval_prob = approval_model.predict_proba(approval_data_scaled)[0][1]

        # Prepare response, handle pd.NA safely
        def safe_float(value):
            if pd.isna(value) or value == float("inf"):
                return 0.0
            return float(value)

        if approval_pred == 1:
            # Step 2: Predict amount and rate if approved
            terms_data_imputed = pd.DataFrame(terms_imputer_X.transform(df), columns=df.columns)
            terms_data_scaled = terms_scaler.transform(terms_data_imputed)
            terms_pred = terms_model.predict(terms_data_scaled)[0]
            
            # Get adjusted amount from function
            predicted_amount = float(terms_pred[0])
            adjusted_amount = adjust_approved_amount(predicted_amount, data)
            
            # Set approved_amount to the smaller of adjusted_amount or requested_amount
            approved_amount = min(data["requested_amount"], adjusted_amount)
    
            # Set approved_amount to the smaller of requested_amount or predicted amount
            # approved_amount = min(data["requested_amount"], float(terms_pred[0]))
            
            response = {
                "approval_status": int(approval_pred),
                "approval_probability": float(approval_prob),
                "approved_amount": approved_amount,
                "interest_rate": float(terms_pred[1]),
                "DTI": safe_float(data["DTI"])  # Use safe_float to handle pd.NA or inf
            }
        else:
            # Get denial reasons
            denial_reason_keys = get_denial_reasons(data)
            denial_reasons = [DENIAL_MESSAGES.get(key, DENIAL_MESSAGES["default"]) for key in denial_reason_keys] if denial_reason_keys else [DENIAL_MESSAGES["default"]]
            response = {
                "approval_status": int(approval_pred),
                "approval_probability": float(approval_prob),
                "approved_amount": 0.0,
                "interest_rate": 0.0,
                "DTI": safe_float(data["DTI"]),  # Use safe_float to handle pd.NA or inf
                "denial_reasons": denial_reasons
            }

        return response

    except ZeroDivisionError:
        return {
            "error": "Division by zero occurred, check annual_income",
            "approval_status": 0,
            "approval_probability": 0.0,
            "approved_amount": 0.0,
            "interest_rate": 0.0,
            "DTI": 0.0,
            "denial_reasons": ["Invalid income data"]
        }
    except Exception as e:
        return {
            "error": f"Prediction failed: {str(e)}",
            "approval_status": 0,
            "approval_probability": 0.0,
            "approved_amount": 0.0,
            "interest_rate": 0.0,
            "DTI": 0.0,
            "denial_reasons": ["Internal server error"]
        }
        
        
def adjust_approved_amount(predicted_amount, data):
    # Base Limit
    if data["credit_score"] >= 660:
        base_limit = 0.5 * data["annual_income"]
    elif 500 <= data["credit_score"] <= 659:
        base_limit = 0.25 * data["annual_income"]
    else:
        base_limit = 0.1 * data["annual_income"]

    # DTI Adjustment
    if data["DTI"] <= 30:
        estimated_amount = base_limit
    elif 30 < data["DTI"] <= 40:
        estimated_amount = base_limit * 0.75  # Reduce by 25%
    else:
        estimated_amount = base_limit * 0.5   # Reduce by 50%

    # Employment Bonus
    if data["employment_status"] == 1 and data["months_employed"] >= 12:
        estimated_amount *= 1.10  # +10% bonus

    # Payment Penalty (assuming payment_history > 60 indicates "Late >60")
    if data["payment_history"] > 60:
        estimated_amount *= 0.5   # -50% penalty

    # Credit Score Cap
    if data["credit_score"] >= 750:
        estimated_amount = min(estimated_amount, 25000)
    elif 660 <= data["credit_score"] <= 749:
        estimated_amount = min(estimated_amount, 15000)
    elif 500 <= data["credit_score"] <= 659:
        estimated_amount = min(estimated_amount, 10000)
    else:
        estimated_amount = min(estimated_amount, 5000)

    # Ensure estimated_amount stays under 25,000
    estimated_amount = min(estimated_amount, 25000)

    # Adjust predicted_amount if deviation > ±2000
    deviation = predicted_amount - estimated_amount
    if abs(deviation) > 2000:
        if deviation > 0:
            adjusted_amount = estimated_amount + 2000  # Cap at estimated + 2000
        else:
            adjusted_amount = estimated_amount - 2000  # Floor at estimated - 2000
        adjusted_amount = max(0, min(adjusted_amount, 25000))  # Keep within 0–25,000
    else:
        adjusted_amount = predicted_amount  # Keep original if within ±2000

    return adjusted_amount
# # Shared prediction function
# def get_prediction(data):
#     try:
#         # Ensure data is a mutable dictionary
#         data = data.copy()

#         # Set default values for optional fields if missing
#         defaults = {
            
#             "credit_score": 0,
#             "annual_income": 1, # Avoid division by zero
#             "self_reported_debt": 0,
#             "self_reported_expenses": 0,
#             "requested_amount": 0,
#             "age": 0,
#             "province": 0,
#             "employment_status": 0,
#             "months_employed": 0,
#             "total_credit_limit": 0,
#             "credit_utilization": 0,
#             "num_open_accounts": 0,
#             "num_credit_inquiries": 0,
#             "payment_history": 0
#         }
#         for key, default in defaults.items():
#             data[key] = float(data.get(key, default))

#         # Calculate derived fields with fallback values
#         data["estimated_debt"] = data["total_credit_limit"] * (data["credit_utilization"] / 100) * 0.03
#         data["total_monthly_debt"] = data["self_reported_debt"] + data["estimated_debt"]
        
#         # Calculate DTI, handle division by zero
#         monthly_income = data["annual_income"] / 12
#         if monthly_income == 0:
#             data["DTI"] = float("inf")
#         else:
#             data["DTI"] = (
#                 (data["self_reported_debt"] + data["estimated_debt"] + (data["requested_amount"] * 0.03))
#                 / monthly_income
#             ) * 100

#         # Convert to DataFrame, fill missing fields with NaN for imputer
#         df = pd.DataFrame([data])
#         df = df.reindex(columns=expected_columns, fill_value=pd.NA)  # Ensure all expected columns

#         # Step 1: Predict approval
#         approval_data_imputed = pd.DataFrame(approval_imputer_X.transform(df), columns=df.columns)
#         approval_data_scaled = approval_scaler.transform(approval_data_imputed)
#         approval_pred = approval_model.predict(approval_data_scaled)[0]
#         approval_prob = approval_model.predict_proba(approval_data_scaled)[0][1]

#         # Prepare response, handle pd.NA safely
#         def safe_float(value):
#             if pd.isna(value) or value == float("inf"):
#                 return 0.0
#             return float(value)

#         if approval_pred == 1:
#             # Step 2: Predict amount and rate if approved
#             terms_data_imputed = pd.DataFrame(terms_imputer_X.transform(df), columns=df.columns)
#             terms_data_scaled = terms_scaler.transform(terms_data_imputed)
#             terms_pred = terms_model.predict(terms_data_scaled)[0]
#             response = {
#                 "approval_status": int(approval_pred),
#                 "approval_probability": float(approval_prob),
#                 "approved_amount": float(terms_pred[0]),
#                 "interest_rate": float(terms_pred[1]),
#                 "DTI": safe_float(data["DTI"])  # Use safe_float to handle pd.NA or inf
#             }
#         else:
#             # Get denial reasons
#             denial_reason_keys = get_denial_reasons(data)
#             denial_reasons = [DENIAL_MESSAGES.get(key, DENIAL_MESSAGES["default"]) for key in denial_reason_keys] if denial_reason_keys else [DENIAL_MESSAGES["default"]]
#             response = {
#                 "approval_status": int(approval_pred),
#                 "approval_probability": float(approval_prob),
#                 "approved_amount": 0.0,
#                 "interest_rate": 0.0,
#                 "DTI": safe_float(data["DTI"]),  # Use safe_float to handle pd.NA or inf
#                 "denial_reasons": denial_reasons
#             }

#         return response

#     except ZeroDivisionError:
#         return {
#             "error": "Division by zero occurred, check annual_income",
#             "approval_status": 0,
#             "approval_probability": 0.0,
#             "approved_amount": 0.0,
#             "interest_rate": 0.0,
#             "DTI": 0.0,
#             "denial_reasons": ["Invalid income data"]
#         }
#     except Exception as e:
#         return {
#             "error": f"Prediction failed: {str(e)}",
#             "approval_status": 0,
#             "approval_probability": 0.0,
#             "approved_amount": 0.0,
#             "interest_rate": 0.0,
#             "DTI": 0.0,
#             "denial_reasons": ["Internal server error"]
#         }
    