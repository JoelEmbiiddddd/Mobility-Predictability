'''

@File   : test.py
@Author : LiXin Huang, NWPU
@Date   : 2023/3/19
@Desc   : 
@Email  : iHuanglixin@outlook.com
'''

import argparse
import json
import datetime

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--gpu_id', default=1, type=int)
parser.add_argument('--size', default=32, type=int)
parser.add_argument('--root', default='/users/xxx', type=str)
args = parser.parse_args()
argsDict = args.__dict__

with open('setting.txt', 'a') as f:
    f.writelines(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n')
    for eachArg, value in argsDict.items():
        f.writelines(eachArg + ' : ' + str(value) + '\n')

    f.writelines('-'*50 + '\n')