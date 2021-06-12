## 훈련을 위한 데이터를 DB에서 추출을 해서 진행하는 데이터입니다.
## 가상의 시나리오를 위해 매 스케쥴마다 부분 부분 데이터를 가져오도록 하겠습니다.
import argparse
import sqlite3
import tqdm
import os
def extract_data():
    

    working_dir = os.getenv('WORKING_DIRECTORY', './')
    db_path = os.path.join(working_dir, 'data.db')
    data_path = os.path.join(working_dir, 'data.csv')
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    
    
    
    if os.path.isfile(data_path) and os.access(data_path, os.R_OK):
        print("File exists and is readable")
    else:
        print("Either the file is missing or not readable create new one")
        with open(data_path, "w") as f:
            f.write('userID,assessmentItemID,testId,answerCode,Timestamp,KnowledgeTag\n')

    qry="SELECT * FROM data_log"
    cur.execute(qry)
    rows = cur.fetchall()
    index = rows[0][0]
    
    
    qry="SELECT * FROM problems LIMIT {}, 10000".format(index)
    cur.execute(qry)

    rows = cur.fetchall()
    num_of_data  = len(rows)
    with open(data_path, "a") as f:
        for row in tqdm.tqdm(rows):
            values = [str(val) for val in row][1:]
            f.write(",".join(values)+"\n")    
    
    
    print("DATA IS BEING EXTRACTED FROM {} TO {}".format(index, index+num_of_data))
    qry = 'UPDATE data_log SET last_added_index = {}'.format(index+num_of_data)
    cur.execute(qry)
    con.commit()
    
    
    
    
    con.close()
    
    print("DONE!")



if __name__ == "__main__":
    
    
    extract_data()