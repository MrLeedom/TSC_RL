'''
   @author:leedom

   Created on Fri Jan 18 12:22:45 2019
   description:将较大的文件抽取其中有效的部分
'''
import pandas as pd

#将某一天的数据大致提取出来
data = pd.read_csv('./output/video_flow180821.csv',nrows=580123)  #文件过大,直接抽取前n行数据
print(data.columns)
data[['t_s','t_e']] = data[['t_s','t_e']].astype(str)
data.to_csv('./output/0722.csv',columns=['device_ID','intersection_name','camerapostion','turnID','t_s','t_e','cameraID','roadID','flow'],sep=',',index=False)
