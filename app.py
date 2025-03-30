from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib
import io
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from flask_mysqldb import MySQL
from app.extras.denial_reasons import get_denial_reasons, DENIAL_MESSAGES  # Relative import
from app.extras.get_prediction import get_prediction
from datetime import datetime  # Import the datetime class

app = Flask(__name__)
CORS(app)

# MySQL Configuration
app.config["MYSQL_HOST"] = "localhost"        # Change if using remote DB
app.config["MYSQL_USER"] = "root"             # Your MySQL username
app.config["MYSQL_PASSWORD"] = "zain1234" # Your MySQL password
app.config["MYSQL_DB"] = "loan_approval_db"       # Your MySQL database name

mysql = MySQL(app)


# Load models and preprocessors
approval_model = joblib.load('app/models/loan_approval_step1_logistic_model.joblib')
approval_scaler = joblib.load('app/models/loan_approval_scaler.joblib')
approval_imputer_X = joblib.load('app/models/loan_approval_imputer_X.joblib')
terms_model = joblib.load('app/models/loan_terms_step2_rf_model.joblib')
terms_scaler = joblib.load('app/models/loan_terms_scaler.joblib')
terms_imputer_X = joblib.load('app/models/loan_terms_imputer_X.joblib')

# Define expected columns
expected_columns = [
    "credit_score", "annual_income", "self_reported_debt", "self_reported_expenses",
    "requested_amount", "age", "province", "employment_status", "months_employed",
    "total_credit_limit", "credit_utilization", "num_open_accounts",
    "num_credit_inquiries", "payment_history", "estimated_debt",
    "total_monthly_debt", "DTI"
]

@app.route('/api/application', methods=['PUT'])
def update_application_status():
    try:
        # Get JSON data from frontend
        data = request.get_json()

        # Ensure required fields are provided
        if "application_id" not in data or "status" not in data:
            return jsonify({"error": "application_id and status are required"}), 400

        application_id = data["application_id"]
        status = data["status"]

        # Update status in loan_applications table
        cursor = mysql.connection.cursor()
        query = """
            UPDATE loan_applications
            SET status = %s, updated_at = %s
            WHERE application_id = %s
        """
        cursor.execute(query, (status, datetime.utcnow().isoformat(), application_id))
        mysql.connection.commit()
        cursor.close()

        return jsonify({"message": "Application status updated successfully"}), 200

    except Exception as e:
        return jsonify({"error": f"Failed to update application status: {str(e)}"}), 500

@app.route('/api/application', methods=['POST'])
def save_application():
    try:
        # Get JSON data from frontend
        data = request.get_json()

        # Ensure applicant_id is provided
        if "applicant_id" not in data:
            return jsonify({"error": "applicant_id is required"}), 400

        applicant_id = data.get("applicant_id")
        approved_amount = data.get("approved_amount")
        approved = data.get("approved", 0)  # Default to 0 if not provided
        interest_rate = data.get("interest_rate")
        dti = data.get("dti")
        status = data.get("status", "Pending")  # Default to "Pending" if not provided
        created_at = data.get("created_at", datetime.utcnow().isoformat())  # Default to now
        updated_at = data.get("updated_at", datetime.utcnow().isoformat())  # Default to now
        approved_at = datetime.utcnow().isoformat() if (status == "Approved" and data.get("approved", 0) == 1) else data.get("approved_at")
        
        rejected_reason = data.get("rejected_reason")
        admin_notes = data.get("admin_notes")

        # Insert into loan_applications table
        cursor = mysql.connection.cursor()
        query = """
            INSERT INTO loan_applications (
                applicant_id, approved_amount, approved, interest_rate, dti, status,
                created_at, updated_at, approved_at, rejected_reason, admin_notes
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (
            applicant_id, approved_amount, approved, interest_rate, dti, status,
            created_at, updated_at, approved_at, rejected_reason, admin_notes
        ))
        mysql.connection.commit()
        cursor.close()

        return jsonify({"msg": "Your application has been submitted successfully"}), 201

    except Exception as e:
        return jsonify({"error": f"Failed to save application: {str(e)}"}), 500

@app.route('/api/application/<applicant_id>', methods=['GET'])
def get_application(applicant_id):
    try:
        cursor = mysql.connection.cursor()

        # Fetch application details from loan_applications table
        app_query = """
            SELECT 
                application_id, applicant_id, approved_amount, approved, interest_rate,
                dti, status, created_at, updated_at, approved_at, rejected_reason, admin_notes
            FROM loan_applications
            WHERE applicant_id = %s
            LIMIT 1
        """
        cursor.execute(app_query, (applicant_id,))
        app_record = cursor.fetchone()

        # Fetch user details from users table regardless of app_record
        user_query = """
            SELECT 
                first_name, last_name, email, address, phone_number
            FROM users
            WHERE applicant_id = %s
            LIMIT 1
        """
        cursor.execute(user_query, (applicant_id,))
        user_record = cursor.fetchone()
        cursor.close()

        # If no application record exists, return found = 0 with user_info
        if not app_record:
            return jsonify({
                "found": 0,
                "user_info": {
                    "first_name": user_record[0] if user_record else None,
                    "last_name": user_record[1] if user_record else None,
                    "email": user_record[2] if user_record else None,
                    "address": user_record[3] if user_record else None,
                    "phone_number": user_record[4] if user_record else None
                } if user_record else None
            }), 200

        # If application record exists, return full response with user_info
        response = {
            "found": 1,
            "application_id": app_record[0],
            "applicant_id": app_record[1],
            "approved_amount": float(app_record[2]) if app_record[2] else None,
            "approved": app_record[3],
            "interest_rate": float(app_record[4]) if app_record[4] else None,
            "dti": float(app_record[5]) if app_record[5] else None,
            "status": app_record[6],
            "created_at": app_record[7].isoformat(),
            "updated_at": app_record[8].isoformat(),
            "approved_at": app_record[9].isoformat() if app_record[9] else None,
            "rejected_reason": app_record[10],
            "admin_notes": app_record[11],
            "user_info": {
                "first_name": user_record[0] if user_record else None,
                "last_name": user_record[1] if user_record else None,
                "email": user_record[2] if user_record else None,
                "address": user_record[3] if user_record else None,
                "phone_number": user_record[4] if user_record else None
            } if user_record else None
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"found": 0, "error": str(e)}), 500
    
# @app.route('/api/application/<applicant_id>', methods=['GET'])
# def get_application(applicant_id):
#     try:
#         cursor = mysql.connection.cursor()
#         query = """
#             SELECT 
#                 application_id, applicant_id, approved_amount, approved, interest_rate,
#                 dti, status, created_at, updated_at, approved_at, rejected_reason, admin_notes
#             FROM loan_applications
#             WHERE applicant_id = %s
#             LIMIT 1
#         """
#         cursor.execute(query, (applicant_id,))
#         record = cursor.fetchone()
#         cursor.close()

#         if not record:
#             return jsonify({"found": 0}), 200

#         response = {
#             "found": 1,
#             "application_id": record[0],
#             "applicant_id": record[1],
#             "approved_amount": float(record[2]) if record[2] else None,
#             "approved": record[3],
#             "interest_rate": float(record[4]) if record[4] else None,
#             "dti": float(record[5]) if record[5] else None,
#             "status": record[6],
#             "created_at": record[7].isoformat(),
#             "updated_at": record[8].isoformat(),
#             "approved_at": record[9].isoformat() if record[9] else None,
#             "rejected_reason": record[10],
#             "admin_notes": record[11]
#         }
#         return jsonify(response), 200

#     except Exception as e:
#         return jsonify({"found": 0, "error": str(e)}), 500
    
@app.route('/api/calculate-loan', methods=['POST'])
def calculate_loan():
    data = request.get_json()
    amount = data.get('amount')
    rate = data.get('rate')
    months = data.get('months')

    # Validate inputs
    if not all([amount, rate, months]) or amount <= 0 or rate <= 0 or months <= 0:
        return jsonify({"error": "Invalid input values"}), 400

    # Convert annual rate to monthly and percentage to decimal
    monthly_rate = rate / 100 / 12
    emi = (amount * monthly_rate * (1 + monthly_rate) ** months) / (((1 + monthly_rate) ** months) - 1)
    total_payment = emi * months
    total_interest = total_payment - amount

    return jsonify({
        "monthlyPayment": round(emi, 2),
        "totalInterest": round(total_interest, 2),
        "totalPayment": round(total_payment, 2)
    })
    
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get JSON data from frontend
#         data = request.get_json()

#         # Ensure applicant_id is provided
#         if "applicant_id" not in data:
#             return jsonify({"error": "applicant_id is required"}), 400

#         applicant_id = data["applicant_id"]
        
#         # Define field types for conversion
#         field_types = {
#             "credit_score": int,
#             "annual_income": float,
#             "self_reported_debt": float,
#             "self_reported_expenses": float,
#             "requested_amount": float,
#             "age": int,
#             "province": int,
#             "employment_status": int,
#             "months_employed": int,
#             "total_credit_limit": float,
#             "credit_utilization": float,
#             "num_open_accounts": int,
#             "num_credit_inquiries": int,
#             "payment_history": int
#         }
        
#         if applicant_id == 9999:
            
#             # Convert incoming data to numeric types
#             incoming_data = {}
#             for field, field_type in field_types.items():
#                 if field in data:
#                     try:
#                         incoming_data[field] = field_type(data[field])
#                     except (ValueError, TypeError):
#                         return jsonify({"error": f"Invalid value for {field}: must be numeric"}), 400
                    
#         else:
#             # Fetch record from credit_details table
#             cursor = mysql.connection.cursor()
#             query = """
#                 SELECT 
#                     credit_score, annual_income, self_reported_debt, self_reported_expenses,
#                     requested_amount, age, province, employment_status, months_employed,
#                     total_credit_limit, credit_utilization, num_open_accounts, num_credit_inquiries,
#                     payment_history
#                 FROM credit_details
#                 WHERE applicant_id = %s
#                 ORDER BY id DESC
#                 LIMIT 1
#             """
#             cursor.execute(query, (applicant_id,))
#             saved_record = cursor.fetchone()
#             cursor.close()

#             if not saved_record:
#                 return jsonify({"error": f"No record found for applicant_id {applicant_id}"}), 404

#             # Map saved record to dictionary
#             saved_data = {
#                 "credit_score": saved_record[0],
#                 "annual_income": float(saved_record[1]),
#                 "self_reported_debt": float(saved_record[2]),
#                 "self_reported_expenses": float(saved_record[3]),
#                 "requested_amount": float(saved_record[4]),
#                 "age": saved_record[5],
#                 "province": saved_record[6],
#                 "employment_status": saved_record[7],
#                 "months_employed": saved_record[8],
#                 "total_credit_limit": float(saved_record[9]),
#                 "credit_utilization": float(saved_record[10]),
#                 "num_open_accounts": saved_record[11],
#                 "num_credit_inquiries": saved_record[12],
#                 "payment_history": saved_record[13]
#             }

#             # Convert incoming data to numeric types
#             incoming_data = {}
#             for field, field_type in field_types.items():
#                 if field in data:
#                     try:
#                         incoming_data[field] = field_type(data[field])
#                     except (ValueError, TypeError):
#                         return jsonify({"error": f"Invalid value for {field}: must be numeric"}), 400

#             # Compare incoming data with saved record
#             mismatched_fields = []
#             for field in incoming_data:
#                 if field in saved_data and incoming_data[field] != saved_data[field]:
#                     mismatched_fields.append({
#                         "field": field,
#                         "provided_value": incoming_data[field],
#                         "saved_value": saved_data[field]
#                     })

#             if mismatched_fields:
#                 return jsonify({
#                     "approval_status" : 0,
#                     "mismatch": 1,
#                     "msg": (
#                         "The provided information does not match our records. "
#                         "Please ensure the details are correct or contact your branch representative for manual approval."
#                     ),
#                     "mismatchFields": [
#                         {
#                             "field": m["field"],
#                             "provided": m["provided_value"],
#                             "expected": m["saved_value"]
#                         } for m in mismatched_fields
#                     ]
#                 }), 200

#         # If all matches, proceed with prediction
#         response = get_prediction(incoming_data)
#         return jsonify(response)

#     except Exception as e:
#         return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from frontend
        data = request.get_json()

        # Ensure applicant_id is provided
        if "applicant_id" not in data:
            return jsonify({"error": "applicant_id is required"}), 400

        applicant_id = data["applicant_id"]
        requested_amount = data["requested_amount"]
        
        # Check if an application already exists in loan_applications table
        cursor = mysql.connection.cursor()
        check_query = """
            SELECT application_id 
            FROM loan_applications 
            WHERE applicant_id = %s 
            LIMIT 1
        """
        cursor.execute(check_query, (applicant_id,))
        existing_application = cursor.fetchone()
        cursor.close()

        # If a record is found, return {"found": 1}
        if existing_application:
            return jsonify({"found": 1}), 200

        # If no existing application, proceed with the current predict flow
        # Define field types for conversion
        field_types = {
            "credit_score": int,
            "annual_income": float,
            "self_reported_debt": float,
            "self_reported_expenses": float,
            "requested_amount": float,
            "age": int,
            "province": int,
            "employment_status": int,
            "months_employed": int,
            "total_credit_limit": float,
            "credit_utilization": float,
            "num_open_accounts": int,
            "num_credit_inquiries": int,
            "payment_history": int
        }
        
        if applicant_id == 999999:
            # Convert incoming data to numeric types
            incoming_data = {}
            for field, field_type in field_types.items():
                if field in data:
                    try:
                        incoming_data[field] = field_type(data[field])
                    except (ValueError, TypeError):
                        return jsonify({"error": f"Invalid value for {field}: must be numeric"}), 400
                    
        else:
            # Fetch record from credit_details table
            cursor = mysql.connection.cursor()
            query = """
                SELECT 
                    credit_score, annual_income, self_reported_debt, self_reported_expenses,
                    requested_amount, age, province, employment_status, months_employed,
                    total_credit_limit, credit_utilization, num_open_accounts, num_credit_inquiries,
                    payment_history
                FROM credit_details
                WHERE applicant_id = %s
                ORDER BY id DESC
                LIMIT 1
            """
            cursor.execute(query, (applicant_id,))
            saved_record = cursor.fetchone()
            cursor.close()

            if not saved_record:
                return jsonify({"error": f"No record found for applicant ID {applicant_id}"}), 404

            # Map saved record to dictionary
            saved_data = {
                "credit_score": saved_record[0],
                "annual_income": float(saved_record[1]),
                "self_reported_debt": float(saved_record[2]),
                "self_reported_expenses": float(saved_record[3]),
                "requested_amount": float(requested_amount),
                "age": saved_record[5],
                "province": saved_record[6],
                "employment_status": saved_record[7],
                "months_employed": saved_record[8],
                "total_credit_limit": float(saved_record[9]),
                "credit_utilization": float(saved_record[10]),
                "num_open_accounts": saved_record[11],
                "num_credit_inquiries": saved_record[12],
                "payment_history": saved_record[13]
            }

            # Convert incoming data to numeric types
            incoming_data = {}
            for field, field_type in field_types.items():
                if field in data:
                    try:
                        incoming_data[field] = field_type(data[field])
                    except (ValueError, TypeError):
                        return jsonify({"error": f"Invalid value for {field}: must be numeric"}), 400

            # Compare incoming data with saved record
            mismatched_fields = []
            for field in incoming_data:
                if field in saved_data and incoming_data[field] != saved_data[field]:
                    mismatched_fields.append({
                        "field": field,
                        "provided_value": incoming_data[field],
                        "saved_value": saved_data[field]
                    })

            if mismatched_fields:
                return jsonify({
                    "approval_status": 0,
                    "mismatch": 1,
                    "msg": (
                        "The provided information does not match our records. "
                        "Please ensure the details are correct or Submit your application for manual approval by bank representative."
                    ),
                    "mismatchFields": [
                        {
                            "field": m["field"],
                            "provided": m["provided_value"],
                            "expected": m["saved_value"]
                        } for m in mismatched_fields
                    ]
                }), 200

        # If all matches, proceed with prediction
        response = get_prediction(incoming_data)
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    

@app.route('/predict/<int:applicant_id>', methods=['GET'])
def predict_by_id(applicant_id):
    try:
        # Fetch the latest credit details for the applicant_id
        cursor = mysql.connection.cursor()
        query = """
            SELECT 
                credit_score, annual_income, self_reported_debt, self_reported_expenses,
                requested_amount, age, province, employment_status, months_employed,
                total_credit_limit, credit_utilization, num_open_accounts, num_credit_inquiries,
                payment_history
            FROM credit_details
            WHERE applicant_id = %s
            ORDER BY id DESC
            LIMIT 1
        """
        cursor.execute(query, (applicant_id,))
        row = cursor.fetchone()
        cursor.close()

        if not row:
            return jsonify({"error": "No credit details found for this applicant"}), 404

        # Map data to dictionary
        data = {
            "credit_score": int(row[0]),
            "annual_income": float(row[1]),
            "self_reported_debt": float(row[2]),
            "self_reported_expenses": float(row[3]),
            "requested_amount": float(row[4]),
            "age": int(row[5]),
            "province": int(row[6]),
            "employment_status": int(row[7]),
            "months_employed": int(row[8]),
            "total_credit_limit": float(row[9]),
            "credit_utilization": float(row[10]),
            "num_open_accounts": int(row[11]),
            "num_credit_inquiries": int(row[12]),
            "payment_history": int(row[13])
        }

        # Get prediction using shared function
        response = get_prediction(data)
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    
@app.route('/generatePDF', methods=['POST'])
def generate_pdf():
    data = request.get_json()
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.setFont("Helvetica", 12)
    p.drawString(100, 750, "Loan Approval Report")
    p.drawString(100, 730, f"Credit Score: {data['formData']['credit_score']}")
    p.drawString(100, 710, f"Annual Income: ${data['formData']['annual_income']}")
    p.drawString(100, 690, f"Requested Amount: ${data['formData']['requested_amount']}")
    p.drawString(100, 670, f"Approval Status: {'Approved' if data['result']['approval_status'] else 'Denied'}")
    if data['result']['approval_status']:
        p.drawString(100, 650, f"Approved Amount: ${data['result']['approved_amount']}")
        p.drawString(100, 630, f"Interest Rate: {data['result']['interest_rate']}%")
    p.showPage()
    p.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name='loan_approval.pdf', mimetype='application/pdf')

@app.route('/admin/applications', methods=['GET'])
def get_all_applications():
    try:
        cursor = mysql.connection.cursor()

        # Fetch all applications with user details
        query = """
            SELECT 
                la.application_id, la.applicant_id, la.approved_amount, la.approved, 
                la.interest_rate, la.dti, la.status, la.created_at, la.updated_at, 
                la.approved_at, la.rejected_reason, la.admin_notes,
                u.first_name, u.last_name, u.email, u.address, u.phone_number, u.username
            FROM loan_applications la
            LEFT JOIN users u ON la.applicant_id = u.applicant_id
        """
        cursor.execute(query)
        records = cursor.fetchall()
        cursor.close()

        # Format response as a list of combined objects
        applications = []
        for record in records:
            app_data = {
                "application_id": record[0],
                "applicant_id": record[1],
                "approved_amount": float(record[2]) if record[2] else None,
                "approved": record[3],
                "interest_rate": float(record[4]) if record[4] else None,
                "dti": float(record[5]) if record[5] else None,
                "status": record[6],
                "created_at": record[7].isoformat(),
                "updated_at": record[8].isoformat(),
                "approved_at": record[9].isoformat() if record[9] else None,
                "rejected_reason": record[10],
                "admin_notes": record[11],
                "first_name": record[12],
                "last_name": record[13],
                "email": record[14],
                "address": record[15],
                "phone_number": record[16],
                "username": record[17]
            }
            applications.append(app_data)

        return jsonify(applications), 200

    except Exception as e:
        return jsonify({"error": f"Failed to fetch applications: {str(e)}"}), 500
    
    
@app.route('/admin/users', methods=['GET'])
def get_admin_users():
    try:
        cursor = mysql.connection.cursor()
        # Fetch users with is_admin = FALSE and their credit details
        query = """
            SELECT 
                u.applicant_id, u.username, u.email,
                cd.credit_score, cd.annual_income, cd.self_reported_debt,
                cd.self_reported_expenses, cd.requested_amount, cd.age, cd.province,
                cd.employment_status, cd.months_employed, cd.total_credit_limit,
                cd.credit_utilization, cd.num_open_accounts, cd.num_credit_inquiries,
                cd.payment_history
            FROM users u
            LEFT JOIN (
                SELECT cd_inner.*
                FROM credit_details cd_inner
                INNER JOIN (
                    SELECT applicant_id, MAX(id) AS max_id
                    FROM credit_details
                    GROUP BY applicant_id
                ) cd_max ON cd_inner.applicant_id = cd_max.applicant_id AND cd_inner.id = cd_max.max_id
            ) cd ON u.applicant_id = cd.applicant_id
            WHERE u.is_admin = FALSE
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()

        print(rows[0])
        # Process results
        response = []
        for row in rows:
            user_data = {
                "applicant_id": row[0],
                "username": row[1],
                "email": row[2],
                "credit_score": row[3],
                "annual_income": float(row[4]) if row[4] else None,
                "self_reported_debt": float(row[5]) if row[5] else None,
                "self_reported_expenses": float(row[6]) if row[6] else None,
                "requested_amount": float(row[7]) if row[7] else None,
                "age": row[8],
                "province": row[9],
                "employment_status": row[10],
                "months_employed": row[11],
                "total_credit_limit": float(row[12]) if row[12] else None,
                "credit_utilization": float(row[13]) if row[13] else None,
                "num_open_accounts": row[14],
                "num_credit_inquiries": row[15],
                "payment_history": row[16]
            }
            
            # Convert to list for JSON response
            response.append(user_data)
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/application/<int:application_id>', methods=['DELETE'])
def delete_application(application_id):
    try:
        cursor = mysql.connection.cursor()
        query = "DELETE FROM loan_applications WHERE application_id = %s"
        cursor.execute(query, (application_id,))
        mysql.connection.commit()
        cursor.close()
        return jsonify({"message": "Application deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# @app.route('/predict/<int:applicant_id>', methods=['GET'])
# def predict_by_id(applicant_id):
#     try:
#         # Fetch the latest credit details for the applicant_id
#         cursor = mysql.connection.cursor()
#         query = """
#             SELECT 
#                 credit_score, annual_income, self_reported_debt, self_reported_expenses,
#                 requested_amount, age, province, employment_status, months_employed,
#                 total_credit_limit, credit_utilization, num_open_accounts, num_credit_inquiries,
#                 payment_history
#             FROM credit_details
#             WHERE applicant_id = %s
#             ORDER BY id DESC
#             LIMIT 1
#         """
#         cursor.execute(query, (applicant_id,))
#         row = cursor.fetchone()
#         cursor.close()

#         if not row:
#             return jsonify({"error": "No credit details found for this applicant"}), 404

#         # Map data to dictionary
#         data = {
#             "credit_score": row[0],
#             "annual_income": float(row[1]),
#             "self_reported_debt": float(row[2]),
#             "self_reported_expenses": float(row[3]),
#             "requested_amount": float(row[4]),
#             "age": row[5],
#             "province": row[6],
#             "employment_status": row[7],
#             "months_employed": row[8],
#             "total_credit_limit": float(row[9]),
#             "credit_utilization": float(row[10]),
#             "num_open_accounts": row[11],
#             "num_credit_inquiries": row[12],
#             "payment_history": row[13]
#         }

#         # Calculate derived fields
#         data["estimated_debt"] = data["total_credit_limit"] * (data["credit_utilization"] / 100) * 0.03
#         data["total_monthly_debt"] = data["self_reported_debt"] + data["estimated_debt"]
#         data["DTI"] = ((data["self_reported_debt"] + data["estimated_debt"] + (data["requested_amount"] * 0.03)) / 
#                    (data["annual_income"] / 12)) * 100

#         # Prepare DataFrame for prediction
#         df = pd.DataFrame([data])[expected_columns]
#         approval_data_imputed = pd.DataFrame(approval_imputer_X.transform(df), columns=df.columns)
#         approval_data_scaled = approval_scaler.transform(approval_data_imputed)
#         approval_pred = approval_model.predict(approval_data_scaled)[0]
#         approval_prob = approval_model.predict_proba(approval_data_scaled)[0][1]

#         # Prepare response
#         if approval_pred == 1:
#             terms_data_imputed = pd.DataFrame(terms_imputer_X.transform(df), columns=df.columns)
#             terms_data_scaled = terms_scaler.transform(terms_data_imputed)
#             terms_pred = terms_model.predict(terms_data_scaled)[0]
#             response = {
#                 "approval_status": int(approval_pred),
#                 "approval_probability": float(approval_prob),
#                 "approved_amount": float(terms_pred[0]),
#                 "interest_rate": float(terms_pred[1]),
#                 "DTI": float(data["DTI"])
#             }
#         else:
#             from app.extras.denial_reasons import get_denial_reasons, DENIAL_MESSAGES
#             denial_reason_keys = get_denial_reasons(data)
#             denial_reasons = [DENIAL_MESSAGES.get(key, DENIAL_MESSAGES["default"]) for key in denial_reason_keys] if denial_reason_keys else [DENIAL_MESSAGES["default"]]
#             response = {
#                 "approval_status": int(approval_pred),
#                 "approval_probability": float(approval_prob),
#                 "DTI": float(data["DTI"]),
#                 "denial_reasons": denial_reasons
#             }

#         return jsonify(response)

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)