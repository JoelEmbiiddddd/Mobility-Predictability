'''

@File   : TrajectoryDataset.py
@Author : LiXin Huang, NWPU
@Date   : 2023/3/27
@Desc   : 
@Email  : iHuanglixin@outlook.com
'''

import os
import json
import pandas as pd
import math
from tqdm import tqdm
import importlib
from logging import getLogger
import numpy as np
import time
from datetime import datetime, timedelta
from collections import defaultdict

parameter_list = ['dataset', 'min_session_len', 'min_sessions', "max_session_len",
                  'cut_method', 'window_size', 'min_checkins']


def parse_time(time_in, timezone_offset_in_minute=0):
    """
    将 json 中 time_format 格式的 time 转化为 local datatime
    """
    date = datetime.strptime(time_in, '%Y-%m-%dT%H:%M:%SZ')  # 这是 UTC 时间
    return date + timedelta(minutes=timezone_offset_in_minute)


def cal_timeoff(now_time, base_time):
    """
    计算两个时间之间的差值，返回值以小时为单位
    """
    # 先将 now 按小时对齐
    delta = now_time - base_time
    return delta.days * 24 + delta.seconds / 3600

class TrajectoryDataset:

    def __init__(self, config):
        self.config = config
        self.cut_data_cache = './output/cut_traj/'
        for param in parameter_list:
            self.cut_data_cache +=   '_' +  str(self.config[param])
        self.cut_data_cache += '.json'
        self.dataset = config['dataset']
        self.dyna_file = self.config.get('dyna_file', self.dataset)
        self.data_path = './output/{}/'.format(config['dataset_path'])
        self.data = None
        # 加载 encoder
        self.logger = getLogger()

    def get_data(self):
        """
        轨迹比较特殊，原子文件中存储的并不是轨迹而是一个一个点，因此需要先对轨迹进行切割
        """

        cut_data = self.cutter_filter()

        with open(self.cut_data_cache, 'w') as f:
            json.dump(cut_data, f)
        self.logger.info('finish cut data')

    def cutter_filter(self):
        """
        切割后的轨迹存储格式: (dict)
            {
                uid: [
                    [
                        checkin_record,
                        checkin_record,
                        ...
                    ],
                    [
                        checkin_record,
                        checkin_record,
                        ...
                    ],
                    ...
                ],
                ...
            }
        """
        # load data according to config
        traj = pd.read_csv(os.path.join(
            self.data_path, '{}.dyna'.format(self.dyna_file)))
        # filter inactive poi
        group_location = traj.groupby('location').count()
        filter_location = group_location[group_location['time'] >= self.config['min_checkins']]
        location_index = filter_location.index.tolist()
        traj = traj[traj['location'].isin(location_index)]
        user_set = pd.unique(traj['entity_id'])
        res = {}
        min_session_len = self.config['min_session_len']
        max_session_len = self.config['max_session_len']
        min_sessions = self.config['min_sessions']
        window_size = self.config['window_size']
        cut_method = self.config['cut_method']  # time_interval ,
        if cut_method == 'time_interval':
            # 按照时间窗口进行切割
            for uid in tqdm(user_set, desc="cut and filter trajectory"):
                usr_traj = traj[traj['entity_id'] == uid].to_numpy()
                sessions = []  # 存放该用户所有的 session
                session = []  # 单条轨迹
                for index, row in enumerate(usr_traj):
                    now_time = parse_time(row[2])
                    if index == 0:
                        session.append(row.tolist())
                        prev_time = now_time
                    else:
                        time_off = cal_timeoff(now_time, prev_time)
                        if time_off < window_size and time_off >= 0 and len(session) < max_session_len:
                            session.append(row.tolist())
                        else:
                            if len(session) >= min_session_len:
                                sessions.append(session)
                            session = []
                            session.append(row.tolist())
                    prev_time = now_time
                if len(session) >= min_session_len:
                    sessions.append(session)
                if len(sessions) >= min_sessions:
                    res[str(uid)] = sessions
        elif cut_method == 'same_date':
            # 将同一天的 check-in 划为一条轨迹
            for uid in tqdm(user_set, desc="cut and filter trajectory"):
                usr_traj = traj[traj['entity_id'] == uid].to_numpy()
                sessions = []  # 存放该用户所有的 session
                session = []  # 单条轨迹
                prev_date = None
                for index, row in enumerate(usr_traj):
                    now_time = parse_time(row[2])
                    now_date = now_time.day
                    if index == 0:
                        session.append(row.tolist())
                    else:
                        if prev_date == now_date and len(session) < max_session_len:
                            # 还是同一天
                            session.append(row.tolist())
                        else:
                            if len(session) >= min_session_len:
                                sessions.append(session)
                            session = []
                            session.append(row.tolist())
                    prev_date = now_date
                if len(session) >= min_session_len:
                    sessions.append(session)
                if len(sessions) >= min_sessions:
                    res[str(uid)] = sessions
        else:
            # cut by fix window_len used by STAN
            if max_session_len != window_size:
                raise ValueError('the fixed length window is not equal to max_session_len')
            for uid in tqdm(user_set, desc="cut and filter trajectory"):
                usr_traj = traj[traj['entity_id'] == uid].to_numpy()
                sessions = []  # 存放该用户所有的 session
                session = []  # 单条轨迹
                for index, row in enumerate(usr_traj):
                    if len(session) < window_size:
                        session.append(row.tolist())
                    else:
                        sessions.append(session)
                        session = []
                        session.append(row.tolist())
                if len(session) >= min_session_len:
                    sessions.append(session)
                if len(sessions) >= min_sessions:
                    res[str(uid)] = sessions
        return res

if __name__ == '__main__':
    with open('./config.json', 'r', encoding='utf-8') as fp:
        config = json.load(fp)

    TrajectoryDataset = TrajectoryDataset(config = config)
    TrajectoryDataset.get_data()