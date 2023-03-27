'''

@File   : DataProcess.py
@Author : LiXin Huang, NWPU
@Date   : 2023/3/18
@Desc   : 
@Email  : iHuanglixin@outlook.com
'''
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
import os
import random
import time
import json

def processing_csv():

    '''
    导入各个csv文件
    '''

    data_userinfo_nyc = pd.read_csv('./data/UserInfo/dataset_UbiComp2016_UserProfile_NYC.txt',sep='\t', header=None,
                           names = ['User ID','Gender',"TwitterFriends","TwitterFollower"],encoding='ISO-8859-1')

    data_userinfo_tky = pd.read_csv('./data/UserInfo/dataset_UbiComp2016_UserProfile_TKY.txt',sep='\t', header=None,
                           names = ['User ID','Gender',"TwitterFriends","TwitterFollower"],encoding='ISO-8859-1')

    checkIn = pd.read_csv('./data/dataset_TIST2015/dataset_TIST2015_Checkins.txt',sep='\t', header=None,
                           names = ['User ID', 'Venue ID', 'UTC time', 'Timezone offset in minutes'],encoding='ISO-8859-1')

    POI = pd.read_csv('./data/dataset_TIST2015/dataset_TIST2015_POIs.txt',sep='\t',header=None,
                      names = ['Venue ID',"Latitude","Longitude","Venue category name","Country code"],encoding='ISO-8859-1')

    city = pd.read_csv('./data/dataset_TIST2015/dataset_TIST2015_Cities.txt',sep='\t', header=None,
                           names = ['City name', 'Latitude', 'Longitude', 'Country code','Country name','City type'],encoding='ISO-8859-1')

    city = city.drop(labels=['Latitude',"Longitude"],axis=1)

    '''
    合并dataframe
    '''
    df1 = pd.merge(POI,city,how='inner',on=['Country code'])

    df_nyc = pd.merge(df1.loc[df1['City name'] == 'New York'],checkIn,how='inner',on=["Venue ID"])

    df_tky = pd.merge(df1.loc[df1['City name'] == 'Tokyo'], checkIn, how='inner', on=["Venue ID"])

    df_nyc = pd.merge(df_nyc,data_userinfo_nyc,how='inner',on=["User ID"])

    df_tky = pd.merge(df_tky, data_userinfo_tky, how='inner', on=["User ID"])

    '''
    筛选出其中的0.4，因为数量实在是太多了
    '''

    userinfo_nyc = list(set(df_nyc["User ID"].tolist()))
    user_choice_nyc = random.sample(userinfo_nyc, int(len(userinfo_nyc) * 0.5))
    df_nyc = df_nyc[(df_nyc['User ID']).isin(user_choice_nyc)]

    userinfo_tky = list(set(df_tky["User ID"].tolist()))
    user_choice_tky = random.sample(userinfo_tky, int(len(userinfo_tky) * 0.5))
    df_tky = df_tky[(df_tky['User ID']).isin(user_choice_tky)]


    df_nyc.to_csv('./output/data_nyc.csv')

    df_tky.to_csv('./output/data_tky.csv')

def libcity_dataProcessing():
    raw_data = pd.read_csv("./output/data_nyc.csv",index_col=0)
    # raw_data = pd.read_csv('./dataset_TIST2015/data.csv', sep='\t', header=None,
    #                        names=['User ID', 'Venue ID', 'Venue category name', 'Latitude',
    #                               'Longitude', 'Timezone offset in minutes', 'UTC time'], encoding='ISO-8859-1')
    output_folder = './output/nyc'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # 先处理 geo
    poi = raw_data.filter(items=['Venue ID', 'Latitude', 'Longitude'])
    poi = poi.groupby('Venue ID').mean()
    poi.reset_index(inplace=True)
    poi['geo_id'] = poi.index

    # 计算 coordinates
    coordinates = []
    for index, row in poi.iterrows():
        coordinates.append('[{},{}]'.format(row['Longitude'], row['Latitude']))

    poi['coordinates'] = coordinates
    poi['type'] = 'Point'
    poi = poi.drop(['Latitude', 'Longitude'], axis=1)
    category_info = raw_data.filter(items=['Venue ID', 'Venue category ID', 'Venue category name'])
    category_info = category_info.rename(columns={'Venue category name': 'venue_category_name',
                                                  'Venue category ID': 'venue_category_id'})
    category_info = category_info.drop_duplicates(['Venue ID'])
    poi = pd.merge(poi, category_info, on='Venue ID')
    loc_hash2ID = poi.filter(items=['Venue ID', 'geo_id'])
    poi = poi.reindex(columns=['geo_id', 'type', 'coordinates', 'venue_category_id', 'venue_category_name'])
    poi.to_csv(output_folder + '/foursquare_nyc.geo', index=False)

    # 处理 usr
    user = pd.unique(raw_data['User ID'])
    user = pd.DataFrame(user, columns=['User ID'])
    user['usr_id'] = user.index

    # 处理 dyna
    dyna = raw_data.filter(items=['User ID', 'Venue ID', 'Timezone offset in minutes', 'UTC time'])
    dyna = pd.merge(dyna, loc_hash2ID, on='Venue ID')
    dyna = pd.merge(dyna, user, on='User ID')

    def parse_time(time_in, timezone_offset_in_minute=0):
        """
        将 json 中 time_format 格式的 time 转化为 local datatime
        """
        date = datetime.strptime(time_in, '%a %b %d %H:%M:%S %z %Y')  # 这是 UTC 时间
        return date + timedelta(minutes=timezone_offset_in_minute)

    new_time = []
    for index, row in tqdm(dyna.iterrows()):
        date = parse_time(row['UTC time'], int(row['Timezone offset in minutes']))
        new_time.append(date.strftime('%Y-%m-%dT%H:%M:%SZ'))

    dyna['time'] = new_time
    dyna['type'] = 'trajectory'
    dyna = dyna.rename(columns={'geo_id': 'location', 'usr_id': 'entity_id'})
    dyna = dyna.sort_values(by='time')
    dyna['dyna_id'] = dyna.index
    dyna = dyna.reindex(columns=['dyna_id', 'type', 'time', 'entity_id', 'location'])
    dyna.to_csv(output_folder + '/foursquare_nyc.dyna', index=False)

    user = user.drop(['User ID'], axis=1)
    user.to_csv(output_folder + '/foursquare_nyc.usr', index=False)

def timestamp(date):
    """时间戳"""
    time_stamp = time.mktime(time.strptime(date, '%Y-%m-%d%H:%M:%S'))

    return int(time_stamp)


def restruct_the_dataset():
    with open('./output/cut_traj/_foursquare_nyc_5_2_50_same_date_12_5.json', 'r', encoding='utf-8') as fp:
        data = json.load(fp)

    process_data = pd.DataFrame(columns=("session_id","user_id","item_id","timeframe","eventdate"))

    session_id = 0
    user_id = 'NA'
    for uid in tqdm(data.keys()):
        # 遍历每个用户
        for Traj_List in data[uid]:
            session_id+=1
            # 遍历每个用户的每条轨迹
            for Traj in Traj_List:
                date,time= Traj[2].strip('Z').split('T')[0],Traj[2].strip('Z').split('T')[1]
                process_data = process_data.append({'session_id': session_id, 'user_id': user_id,
                                                'item_id': Traj[-1], 'timeframe': timestamp(date+time),'eventdate': date}, ignore_index=True)

    #process_data = process_data.drop(labels='Unnamed: 0',axis=1)
    process_data.to_csv('./output/processed/_foursquare_nyc_5_2_50_same_date_12_5.csv')

def analyse_dataset(data_path):
    with open(data_path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)

    location_max = 0

    session_len = 0
    check_in_len = 0
    location_list = []
    user_len = len(data.keys())
    for uid in tqdm(data.keys()):
        # 遍历每个用户
        for Traj_List in data[uid]:
            # 遍历每个用户的每条轨迹
            for Traj in Traj_List:
                check_in_len+=1
                if Traj[-1] not in location_list:
                    location_list.append(Traj[-1])

                location_max = max(location_max,Traj[-1])
            session_len+=1

    print(session_len)
    print(user_len)
    print(len(location_list))
    print(check_in_len)

if __name__ == '__main__':
    # processing_csv()
    restruct_the_dataset()
    # my_test()
    # libcity_dataProcessing()
    # analyse_dataset('./output/cut_traj/_foursquare_nyc_5_2_50_same_date_12_5.json')