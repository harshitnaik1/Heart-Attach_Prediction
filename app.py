import pandas as pd
from flask import Flask,render_template,jsonify,request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl",'rb'))
preprocessor = pickle.load(open("preprocessor.pkl",'rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods = ["POST"])
def predict():
    data = {
        "Gender":int(request.form['Gender']),
        "age": float(request.form["age"]),
        "cigsPerDay": float(request.form["cigsPerDay"]),
        "tot cholesterol": float(request.form["cholesterol"]),
        "Systolic BP": float(request.form["sysBP"]),
        "Diastolic BP": float(request.form["diaBP"]),
        "BMI": float(request.form["bmi"]),
        "heartRate": float(request.form["heartRate"]),
        "glucose": float(request.form["glucose"]),
        "currentSmoker": int(request.form["currentSmoker"]),
        "education": int(request.form["education"]),
        "BP Meds": int(request.form["bpmeds"]),
        "prevalentStroke": int(request.form["stroke"]),
        "prevalentHyp": int(request.form["hyp"]),
        "diabetes": int(request.form["diabetes"])
    }
    new_data = pd.DataFrame([data])
    processed = preprocessor.transform(new_data)
    pred = model.predict(processed)[0]

    result = "Person is at High Risk"if pred==1 else "Person is at Low Risk"

    return render_template("index.html",prediction_text = result)

if __name__ == "__main__":
    app.run(debug=True)