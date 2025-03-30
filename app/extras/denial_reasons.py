# denial_reasons.py

def get_denial_reasons(data):
    """
    Evaluate input data and return a list of denial reason keys based on rules.
    """
    reasons = []

    # Credit Score Checks
    if data["credit_score"] < 500:
        reasons.append("credit_score_below_500")
    elif data["credit_score"] < 660:
        reasons.append("credit_score_below_660")

    # DTI Checks
    if data["DTI"] > 50:
        reasons.append("dti_above_50")
    elif data["DTI"] > 40:
        reasons.append("dti_above_40")

    # Payment History Checks
    if data["payment_history"] == 0:
        reasons.append("payment_history_late_60")
    elif data["payment_history"] != 3:
        reasons.append("payment_history_not_perfect")

    # Credit Utilization Checks
    if data["credit_utilization"] > 80:
        reasons.append("credit_util_above_80")
    elif data["credit_utilization"] > 50:
        reasons.append("credit_util_above_50")

    return reasons

# Dictionary of denial reasons with user-friendly messages
DENIAL_MESSAGES = {
    "credit_score_below_500": "Credit score is too low (below 500). Consider improving your credit score.",
    "credit_score_below_660": "Credit score is below the preferred threshold (660). A higher score may help.",
    "dti_above_50": "Debt-to-Income ratio is too high (above 50%). Reduce your debt or increase income.",
    "dti_above_40": "Debt-to-Income ratio exceeds 40%. Lowering your debt could improve approval chances.",
    "payment_history_late_60": "Payment history shows late payments over 60 days. Maintain on-time payments.",
    "payment_history_not_perfect": "Payment history is not perfect. Consistent on-time payments may help.",
    "credit_util_above_80": "Credit utilization is too high (above 80%). Reduce your credit usage.",
    "credit_util_above_50": "High credit utilization (above 50%) may have impacted approval.",
    "default": "Your application did not meet the model’s approval criteria. Review your financial profile."
}