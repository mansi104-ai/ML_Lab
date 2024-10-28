from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'savedmodel.sav')
model = pickle.load(open(model_path, 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    result = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    return render_template('index.html', result=result)

# Run the app
if __name__ == "__main__":
    app.run(debug=True,port = 5001)
