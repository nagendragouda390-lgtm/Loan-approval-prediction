from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("loan_xgb_pipeline.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    data = pd.DataFrame([{
        "person_age": int(request.form["person_age"]),
        "person_income": float(request.form["person_income"]),
        "person_home_ownership": request.form["person_home_ownership"],
        "person_emp_length": float(request.form["person_emp_length"]),
        "loan_intent": request.form["loan_intent"],
        "loan_grade": request.form["loan_grade"],
        "loan_amnt": float(request.form["loan_amnt"]),
        "loan_int_rate": float(request.form["loan_int_rate"]),
        "loan_percent_income": float(request.form["loan_percent_income"]),
        "cb_person_default_on_file": request.form["cb_person_default_on_file"],
        "cb_person_cred_hist_length": int(request.form["cb_person_cred_hist_length"])
    }])

    prediction = model.predict(data)[0]

    result = "Loan Approved" if prediction == 1 else "Loan Rejected"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
