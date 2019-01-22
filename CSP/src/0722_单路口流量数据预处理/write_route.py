# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 10:21:56 2018

@author: Administrator
"""
import numpy as np
import pandas as pd

#Input:日期(如'0722')、route_file文件(如'0722route.xlsx')
#OutPut:rou.xml文件，文件名：'shixin_shanyin'+date+'.rou.xml'

def write_route(date,route_file):
    route_df = pd.DataFrame(pd.read_excel(route_file))
    file_name='shixin_shanyin_'+date+'.rou.xml'
    with open(file_name, "a") as routes:
        print("""<routes>
<route id="e_s" edges="gneE1 -gneE0"/>
<route id="e_w" edges="gneE1 -gneE3"/>
<route id="e_n" edges="gneE1 -gneE2"/>
<route id="w_s" edges="gneE3 -gneE0"/>
<route id="w_e" edges="gneE3 -gneE1"/>
<route id="w_n" edges="gneE3 -gneE2"/>
<route id="s_e" edges="gneE0 -gneE1"/>
<route id="s_w" edges="gneE0 -gneE3"/>
<route id="s_n" edges="gneE0 -gneE2"/>
<route id="n_e" edges="gneE2 -gneE1"/>
<route id="n_w" edges="gneE2 -gneE3"/>
<route id="n_s" edges="gneE2 -gneE0"/>""", file=routes)
   
        for i in range(route_df.shape[0]):
            begin_time=i*300
            end_time=(i+1)*300
            for j in range(1,route_df.shape[1]):
                flow_id=i*12+j
                route_id=route_df.columns[j]
                number=route_df.iat[i,j]
                print('<flow id="shixin_shanyin_%s" route="%s" begin="%d" end="%d" number="%d" departlane="random"/>'\
                      %(flow_id,route_id,begin_time,end_time,number), file=routes)
        print("""</routes>""", file=routes)
if __name__ == "__main__":
    write_route('low', 'low.xlsx')