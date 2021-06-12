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
    print(data)
    res = requests.post(f'http://127.0.0.1:5000/predict', json= data) # Set BentoML serve port to 5000(default)
    score = res.text
    print(score)
    score = int(float(score))
    print(score)

    return str(score)

# TODO Retrain by API request? or Cron Job by Airflow?

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=6006, debug=True)
