from flask import Flask, render_template
from flask import request

import inference
import sys
from args import parse_args
from datetime import datetime, timedelta
import sqlite3

app = Flask(__name__)

def insert_data(data):
    now = datetime.now()
    
    con = sqlite3.connect('data.db')
    cur = con.cursor()
    qry = 'SELECT userID FROM problems ORDER BY userID DESC LIMIT 1;'
    cur.execute(qry)
    rows = cur.fetchall()
    
    last_user_id = int(rows[0][0])
    current_user_id = last_user_id+10000 #파일에 있는 user id와 겹치지 않게 하기 위해 10000을 더해줍니다.
    for row in data :
    
        
        #row_index, userID, assessmentItemId, testId, answerCode, Timestamp, KnowledgeTag
        qry_values = "NULL,'{}','{}','{}', '{}', '{}', '{}'"\
            .format(current_user_id, row[0],row[1], row[3], now.strftime("%Y-%m-%d %H:%M:%S"), row[2])

    
        query = '''INSERT INTO problems VALUES ({})'''.format(qry_values)
    
        timestamp = now + timedelta(seconds=10)
        now = timestamp
        cur.execute(query)
    con.commit()
    con.close()

        
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/get_num_of_events', methods=['GET'])
def get_num_of_events():
    con = sqlite3.connect('data.db')
    cur = con.cursor()
    qry = 'SELECT row_index FROM problems ORDER BY row_index DESC LIMIT 1;'
    cur.execute(qry)
    rows = cur.fetchall()
    num_of_event = str(rows[0][0])
    return num_of_event

@app.route('/get_score', methods=['POST'])
def get_score():
    data = request.json
    user_data = []
    print(data)
    for d in data:
        if 'answer' in d:
            row = [d['assess_id'], d['test_id'],d['tag'], d['answer']]
            user_data.append(row)
     
    print(user_data)
    insert_data(user_data)
    score = inference.inference(user_data)
    score = int(score)
    return str(score)

if __name__ == '__main__':
    args = parse_args(mode='train')

    app.run(host="0.0.0.0", port=args.port, debug=True)
