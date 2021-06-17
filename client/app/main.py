import sys

from flask import Flask, render_template
from flask import request

import requests
 
app = Flask(__name__)

 
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/get_score', methods=['POST'])
def get_score():
    data = request.json
    res = requests.post("http://inference:5000/predict", json= data)
    score = res.text
    score = int(float(score))
    print(score)

    return str(score)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=6006, debug=True)
