from flask import Flask, render_template
from flask import request

import inference

 
app = Flask(__name__)

 
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/get_score', methods=['POST'])
def get_score():
    data = request.json # 유저가 웹페이지에서 입력한 데이터
    processed_data = [] # question.csv 순서대로 유저가 입력한 데이터를 전처리해서 넣을 리스트
    print(data)
    for d in data:
        if 'answer' in d:
            row = [d['assess_id'], d['test_id'],d['tag'], d['answer']] # answer : 예(1)/아니오(0) 여부
            processed_data.append(row)
     
    print(processed_data)
    score = inference.inference(processed_data) # 전처리된 데이터를 가지고 추론하여 점수 받아오기
    score = int(score)
    return str(score)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=6006, debug=True)
