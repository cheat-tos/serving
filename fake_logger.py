""" Process 0에 해당하는 모듈입니다.
로그 Format,  참고 블로그: https://m.blog.naver.com/PostView.naver?blogId=innerbus_co&logNo=220807233038&proxyReferer=https:%2F%2Fwww.google.com%2F
"""

import time
import random
import logging
import argparse
import pandas as pd
from datetime import datetime


class FakeLogger(object):
    def __init__(self):
        train_path = "./train_data.csv"
        dtype = {"userID": "int16", "answerCode": "int8", "KnowledgeTag": "int16"}
        df = pd.read_csv(train_path, dtype=dtype, parse_dates=["Timestamp"])
        self.ass_set = list(set(df.assessmentItemID))
        self.test_set = list(set(df.testId))

    def _get_user_ip(self):
        """ return fake user ip like '000.000.000.000' """
        user_ip = [str(random.randint(0, 255)).zfill(3) for _ in range(4)]
        return ".".join(user_ip)

    def _get_asssesment_item_id(self):
        """ return fake ass item id like 'A050036002' """
        return random.choice(self.ass_set)

    def _get_test_id(self):
        """ return fake test id like 'A050000001' """
        return random.choice(self.test_set)

    def _get_fake_log(self, user_ip, test_id, ass_id, rfc931="-", user_id="-"):
        """ return fake log """
        timezone = datetime.now().strftime("[%d/%h/%Y:%H:%M:%S:%s]")
        log = f"{user_ip} {rfc931} {user_id} {timezone} 'GET /{test_id}/{ass_id}' HTTP/1.0 200 1043"
        return log

    def make_logs_chunk(self, user_num=10, log_cnt=1000):
        chunk_logs = []
        user_ips = list(set([self._get_user_ip() for _ in range(user_num)]))

        for _ in range(log_cnt):
            user_ip = random.choice(user_ips)
            ass_id = self._get_asssesment_item_id()
            test_id = self._get_asssesment_item_id()
            chunk_logs.append(self._get_fake_log(user_ip, test_id, ass_id))

            print(chunk_logs[-1])
            latency = random.randint(0, 2)
            time.sleep(latency)

        return chunk_logs


def main(args):
    #  config = open()

    logger = logging.getLogger("fake")
    file_handler = logging.FileHandler("fake.log")
    logger.addHandler(file_handler)

    log_helper = FakeLogger()

    if args.type == "real_time":
        while True:
            chunk_logs = log_helper.make_logs_chunk(user_num=10, log_cnt=1000)

            for log in chunk_logs:
                logger.info(log)

            time.sleep(10)

    raise argparse.ArgumentError(f"{args.type}은 존재하지 않는 Type입니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate fake log in two types")
    parser.add_argument("--type", type=str, default="real_time", help="choose 'real_time'")

    args = parser.parse_args()
    main(args)
