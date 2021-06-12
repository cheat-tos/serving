## 최초로 데이터를 DB에 넣는 작업을 수행해주는 코드입니다
## 이 작업은 한번만 수행됩니다
import argparse
import sqlite3
import tqdm




def migrate(path=''):
    con = sqlite3.connect('data.db')
    cur = con.cursor()
    try:
        ## TABLE for user interaction
        cur.execute('''CREATE TABLE problems
                    (row_index INTEGER PRIMARY KEY, userID INTEGER,assessmentItemID TEXT,testId TEXT,answerCode INTEGER,Timestamp INTEGER,KnowledgeTag INTEGER)''')
        ## TABLE for data log
        ## 마지막으로 추출한 데이터가 어디인지 확인하기 위해
        cur.execute('''CREATE TABLE data_log
                    (last_added_index INTEGER)''')
        query ="INSERT INTO data_log VALUES (0)"
        cur.execute(query)
    except Exception as e:
        print(e)
        

    
    with open(path, 'r', encoding='utf-8') as f:
        #첫번째 줄(Header)은 Skip
        next(f)
        cnt = 0
        for line in tqdm.tqdm(f):
            line = line.strip()
            line = ','.join(["'"+a+"'" for a in line.split(',')[1:]])
            line = "NULL,"+line
            query = '''INSERT INTO problems VALUES ({})'''.format(line)
            cur.execute(query)
            if cnt%100 == 99:
                con.commit()
            
            
            cnt+=1
            
            
            
    con.commit()


    con.close()
    print("DONE!")
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True,
                        help='데이터 파일의 경로를 넣어주세요')
    
    args = parser.parse_args()
    migrate(args.path)