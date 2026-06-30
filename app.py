from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    chol = float(request.form['chol'])
    bp = float(request.form['bp'])

    data = np.array([[age, chol, bp]])
    data = scaler.transform(data)

    prediction = model.predict(data)[0]

    if prediction == 1:
        result = "⚠️ High Risk of Heart Disease"
    else:
        result = "✅ Low Risk"

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)