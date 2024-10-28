from flask import Flask,render_template,request
import pickle

app = Flask(__name__)

#Load the model
model = pickle.load(open('savedmodel.sav','rb'))

@app.route('/')
def home():
    return render_template('index.html',**locals())

@app.route('/predict',methods = ['POST','GET'])
def predict():
    sepal_length= float(request.form['sepal_length'])#get it from a form
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    result = model.predict([[sepal_length,sepal_width,petal_length,petal_width]])
    return render_template('index.html',**locals())
    


#let's initialize a few things
if __name__ == "__main__":
    app.run(debug=True)
