# Import necessary libraries
from flask import Flask, render_template, request
import numpy as np
import joblib

# Create a Flask application
app = Flask(__name__)

# Load the pre-trained machine learning model
model = joblib.load('model.pkl')

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        input_data = [float(request.form['white']),
                      float(request.form['fixed acidity']),
                      float(request.form['volatile acidity']),
                      float(request.form['citric acid']),
                      float(request.form['residual sugar']),
                      float(request.form['chlorides']),
                      float(request.form['free sulfur dioxide']),
                      float(request.form['total sulfur dioxide']),
                      float(request.form['density']),
                      float(request.form['pH']),
                      float(request.form['sulphates']),
                      float(request.form['alcohol'])]

        # Ensure you have all 12 features and fill any missing ones with appropriate values
        if len(input_data) < 12:
            raise ValueError("Missing input features")

        # Convert input data to a numpy array
        input_data = np.array(input_data).reshape(1, -1)

        # Make a prediction
        prediction = model.predict(input_data)

        # Display the prediction result
        return render_template('result.html', prediction=prediction[0])

    except Exception as e:
        # Log the error
        print(f"Error: {str(e)}")

        # Return an error response
        return "An error occurred while processing your request.", 500



if __name__ == '__main__':
    app.run(debug=True)
