# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 19:07:28 2018

@author: Administrator
"""

import numpy as np
import pandas as pd


# 时间转化为秒
def  t2s(t):
    h, m = t.strip().split(":")
    return int(h) * 3600 + int(m) * 60 

# 单车道流量修复
def  fill_missing_data(lane_name, time_array):
    # 读取原始流量存入flow_array第一列为时间，第二列是流量
    flow_file_name = lane_name+'.txt'
    f = open(flow_file_name, 'rb')
    flow = f.readline()
    flow_list = []
    while flow:
        strs=flow.split()
        flow_list.append(strs)
        flow=f.readline()
    f.close()
    flow_array = np.array(flow_list)
    flow_array = flow_array[:, [1, 2]]
        
    # 找出数据缺失的行号,流量设为0
    result=np.setdiff1d(time_array,flow_array[:,0])
    comple_set=np.zeros(result.shape)
    comple_set=np.c_[result,comple_set]
        
    # 补全数据
    new_flow_array=np.concatenate((flow_array, comple_set), axis=0)
    
    # 时间转化为秒,然后排序
    for i in range(new_flow_array.shape[0]):
        new_flow_array[i][0]=t2s(new_flow_array[i][0])
    new_flow_array=np.array(new_flow_array, dtype='float')
    new_flow_array=new_flow_array[new_flow_array[:,0].argsort()]
    return new_flow_array

# Input:数据日期字符串，如:'0722'
# 需要准备的文件：
# 1.time.txt
# 2.lane2routes.xlsx(把车道映射为方向，共计12个方向）
# 3.各个车道的流量数据（例如e_w_1.txt，即东向西第一车道的流量）
# Output: 0722flow.xlsx，即每个车道的流量
#         0722route.xlsx，即每个方向的流量
def batch_fill_missing_data(date):
    # 读取lane_name和route_id
    lane_route_df = pd.DataFrame(pd.read_excel('lanes2routes.xlsx'))
    lane_name = np.array(lane_route_df.lane_list, dtype='<U10')
    route_name = lane_route_df['route_id'].unique()
    
    # 读取时间戳，存入time_array
    f = open('time.txt')
    line = f.readline()
    data_list = []
    while line:
        times = line.split()
        data_list.append(times)
        line = f.readline()
    f.close()
    time_array = np.array(data_list)
    time_array = time_array[:,1]
    
    # 时间戳转化为秒，存入新建的DataFrame
    new_time_array = np.zeros(time_array.shape[0])
    for j in range(time_array.shape[0]):
        new_time_array[j] = t2s(time_array[j])
    # 新建dataFrame，用于存储修复后的数据,先存入时间戳
    modified_flow=pd.DataFrame(new_time_array, columns=['time'])
    
    # 逐车道读取流量数据并修复
    for lane in lane_name:
        # 修复当前车道的流量
        new_flow = fill_missing_data(lane, time_array)
        df_temp = pd.DataFrame(new_flow[:, 1], columns=[lane])
        # 插入流量数据表
        modified_flow = pd.concat([modified_flow, df_temp], axis=1)
    flow_file_name = date+'flow.xlsx'
    modified_flow.to_excel(flow_file_name)
    
    # 生成route_df
    route_df=pd.DataFrame(modified_flow.iloc[:,0])
    
    for r in route_name:
        temp_lane=lane_route_df[lane_route_df.route_id==r]
        temp_route=pd.Series(np.zeros(288))
        for l in temp_lane.lane_list:
            temp_route+=modified_flow[l]
        temp_df=pd.DataFrame(temp_route.values,columns=[r])
        route_df=pd.concat([route_df, temp_df], axis=1)
    route_file_name=date+'route.xlsx'
    route_df.to_excel(route_file_name)
     
if __name__ == "__main__":
    batch_fill_missing_data("0728")