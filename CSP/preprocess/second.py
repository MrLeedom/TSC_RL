'''
   @author:leedom

   Created on Fri Jan 18 12:06:57 2019
   description:将不同路口的数据区分开来,拿出上午七点到九点的数据
'''
import pandas as pd
import datetime
#分别将几个路口的数据分开放
junction = {'airport_road':200430,'lihua_road':200277,'zhengning_road':200276,'jianshe4_road':200235,'jianshe3_road':200270,'jianshe2_road':200269,'jianshe1_road':200268}
data = pd.read_csv('./output/0722.csv')
# print(data.head())
for key,value in junction.items():   
   data_road = data[data['device_ID'] == value]


   #想要测的目标时间段,切换到时分秒
   start = '2018-07-22 07:00:00'
   end = '2018-07-22 09:00:00'
   start = datetime.datetime.strptime(start,'%Y-%m-%d %H:%M:%S')
   start = start.time()
   end = datetime.datetime.strptime(end,'%Y-%m-%d %H:%M:%S')
   end = end.time()

   data_road['date'] = [datetime.datetime.strptime(data_road.iloc[i,:]['t_s'],'%Y-%m-%d %H:%M:%S') for i in range(data_road.iloc[:,0].size)]
   data_road['time'] = [i.time() for i in data_road['date']]
  
   data_road = data_road[ (data_road['time'] >= start) & (data_road['time'] <= end)]
   #输出到指定文件中
   data_road.to_csv('./output/' + str(key) + '.csv',columns=['device_ID','intersection_name','camerapostion','turnID','t_s','t_e','cameraID','roadID','flow'],sep=',',index=False)