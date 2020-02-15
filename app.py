import numpy as np

from flask import Flask, request, jsonify, render_template

import pickle

app = Flask(__name__)

model = pickle.load(open('taxi.pky','rb'))


#creating the Decorators so that we can link our pages 

@app.route('/')

def home():

  return render_template('index.html')

@app.route('/predict',methods =['POST'])
@app.route('/predict2',methods = ['GET','POST'])
def predict2():
#collecting the data from html page
 feature = []
 feature_list = ["Priceperweek", "Population", "Monthlyincome", "Averageparkingpermonth"]
 for i in feature_list:
     feature.append(int(request.values.get(i)))

 prediction = model.predict([feature])
 output = round(prediction[0], 2)

 return render_template('index.html', prediction_text="Weekly rides should be {}".format(int(output)))

 


if __name__ == "__main__":
    app.run(debug=True)

