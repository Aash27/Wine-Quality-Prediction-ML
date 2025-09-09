import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle

app = Flask(__name__)
model = pickle.load(open("winequality", "rb"))

@app.route("/")
def Home():
    return render_template("index1.html")

@app.route("/predict", methods = ["POST"])
def predict():
    input_data = [    float(request.form['fixed acidity']),
                      float(request.form['volatile acidity']),
                      float(request.form['citric acid']),
                      float(request.form['residual sugar']),
                      float(request.form['chlorides']),
                      float(request.form['free sulfur dioxide']),
                      float(request.form['total sulfur dioxide']),
                      float(request.form['density']),
                      float(request.form['pH']),
                      float(request.form['sulphates']),
                      float(request.form['alcohol']),
                      float(request.form['white'])]

    features = [np.array(input_data)]
    prediction = model.predict(features)
    return render_template("result.html", prediction = prediction[0])

if __name__ == "__main__":
    app.run(debug=True)


